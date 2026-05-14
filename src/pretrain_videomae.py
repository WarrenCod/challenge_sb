"""
Stage 1 — VideoMAE self-supervised pretraining on unlabeled video clips.

Tubelet embedding (joint space-time patches) + tube masking + asymmetric
encoder/decoder + per-tubelet normalized-pixel MSE.

Run (from repo root):

    python src/pretrain_videomae.py experiment=videomae_pretrain

Smoke (foreground, <2 min — and pass training.resume=false so it doesn't poison
the real run's _last.pt on config-hash mismatch):

    python src/pretrain_videomae.py experiment=videomae_pretrain \\
        dataset.max_samples=64 training.epochs=1 training.batch_size=4 \\
        training.resume=false

The encoder state is saved to ``checkpoints/videomae_stage1.pt`` (under
``encoder_state_dict``) for Stage 2 loading. Full resumable state lives in
``..._last.pt`` next to it.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_ssl_dataset import (
    UnlabeledVideoClipDataset,
    collect_clip_dirs_deep,
)
from models.videomae import VideoMAEModel
from utils import (
    atomic_torch_save,
    capture_rng_state,
    finish_wandb,
    init_wandb,
    make_resume_hash,
    restore_rng_state,
    set_seed,
    wandb_log,
)


class _ClipConsistentTransform:
    """RandomResizedCrop + ColorJitter, with all random params shared across
    the clip's frames. Then per-frame ToTensor + Normalize.

    No flips (SSv2 labels are direction-sensitive). No RandAugment / RandomErasing
    — VideoMAE relies on masking, not heavy photometric aug.
    """

    def __init__(
        self,
        image_size: int = 224,
        scale=(0.5, 1.0),
        ratio=(3 / 4, 4 / 3),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
    ) -> None:
        from torchvision.transforms import ColorJitter, RandomResizedCrop

        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._RRC = RandomResizedCrop
        self._CJ = ColorJitter
        self._mean = [0.5, 0.5, 0.5]
        self._std = [0.5, 0.5, 0.5]

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        i, j, h, w = self._RRC.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        frames = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]

        b_range = (max(0.0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
        c_range = (max(0.0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
        s_range = (max(0.0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
        fn_idx, b, c, s, _ = self._CJ.get_params(b_range, c_range, s_range, None)
        jittered = []
        for img in frames:
            for fid in fn_idx:
                if fid == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fid == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fid == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
            jittered.append(img)

        tensors = [TF.normalize(TF.to_tensor(img), mean=self._mean, std=self._std) for img in jittered]
        return torch.stack(tensors, dim=0)


def build_optimizer(model: nn.Module, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; no decay on 1-D params, mask_token, or pos_embed (which is a buffer)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "mask_token" in name or "pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))


def make_lr_lambda(total_steps: int, warmup_steps: int):
    def _fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return _fn


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- data --------------------------------------------------------------------
    pretrain_dirs = cfg.dataset.get("pretrain_dirs")
    if pretrain_dirs:
        roots = [Path(d).resolve() for d in pretrain_dirs]
    else:
        roots = [Path(cfg.dataset.train_dir).resolve()]
    print(f"[videomae] pooling clips from {[str(r) for r in roots]}")
    clip_dirs = collect_clip_dirs_deep(roots)
    if cfg.dataset.get("max_samples"):
        clip_dirs = clip_dirs[: int(cfg.dataset.max_samples)]
    print(f"[videomae] {len(clip_dirs)} clips after pooling")

    clip_transform = _ClipConsistentTransform(
        image_size=int(cfg.videomae.image_size),
        scale=tuple(cfg.videomae.get("rrc_scale", (0.5, 1.0))),
        ratio=tuple(cfg.videomae.get("rrc_ratio", (3 / 4, 4 / 3))),
        brightness=float(cfg.videomae.get("brightness", 0.4)),
        contrast=float(cfg.videomae.get("contrast", 0.4)),
        saturation=float(cfg.videomae.get("saturation", 0.4)),
    )

    dataset = UnlabeledVideoClipDataset(
        video_dirs=clip_dirs,
        num_frames=int(cfg.videomae.num_frames),
        clip_transform=clip_transform,
        seed=int(cfg.dataset.seed),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(int(cfg.training.num_workers) > 0),
    )

    # --- model -------------------------------------------------------------------
    model = VideoMAEModel(
        num_frames=int(cfg.videomae.num_frames),
        img_size=int(cfg.videomae.image_size),
        tubelet_time=int(cfg.videomae.tubelet_time),
        tubelet_size=int(cfg.videomae.tubelet_size),
        embed_dim=int(cfg.videomae.embed_dim),
        depth=int(cfg.videomae.depth),
        num_heads=int(cfg.videomae.num_heads),
        mlp_ratio=float(cfg.videomae.mlp_ratio),
        decoder_embed_dim=int(cfg.videomae.decoder_embed_dim),
        decoder_depth=int(cfg.videomae.decoder_depth),
        decoder_num_heads=int(cfg.videomae.decoder_num_heads),
        mask_ratio=float(cfg.videomae.mask_ratio),
        norm_pix_loss=bool(cfg.videomae.norm_pix_loss),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[videomae] model = {n_params:.1f}M params, grid=(t={model.t_grid}, "
        f"h={model.hw_grid}, w={model.hw_grid}); {model.num_tokens} tokens, "
        f"mask_ratio={model.mask_ratio}"
    )

    # --- optim / sched -----------------------------------------------------------
    epochs = int(cfg.training.epochs)
    steps_per_epoch = len(loader)
    limit_batches: Optional[int] = cfg.training.get("limit_batches", None)
    if limit_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(limit_batches))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(cfg.training.get("warmup_epochs", 0)) * steps_per_epoch

    batch_size = int(cfg.training.batch_size)
    grad_accum = max(1, int(cfg.training.get("grad_accum", 1)))
    effective_batch = batch_size * grad_accum
    base_lr = float(cfg.training.base_lr) * effective_batch / 256.0
    optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=float(cfg.training.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warmup_steps))

    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16  # bf16: no GradScaler needed.
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    # --- checkpoint / resume ------------------------------------------------------
    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_every = int(cfg.training.get("snapshot_every", 50))
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)

    init_wandb(cfg, default_run_name=ckpt_path.stem)

    cfg_hash = make_resume_hash(cfg, keys=("dataset", "training", "videomae"))
    start_epoch = 0
    resume_enabled = bool(cfg.training.get("resume", True))
    if resume_enabled and last_path.exists():
        print(f"[videomae] found resume file: {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("cfg_hash") != cfg_hash:
            raise RuntimeError(
                f"Resume aborted: config hash mismatch.\n"
                f"  saved:   {state.get('cfg_hash')}\n"
                f"  current: {cfg_hash}\n"
                f"To start fresh, either delete {last_path} or pass training.resume=false."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[videomae] resuming from epoch {start_epoch + 1}/{epochs}")
    elif not resume_enabled and last_path.exists():
        print(f"[videomae] training.resume=false → ignoring {last_path}")

    # --- train -------------------------------------------------------------------
    avg: Optional[float] = None
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running, seen = 0.0, 0
        # Health probes: pred std signals decoder/encoder collapse (drops to ~0 if
        # encoder memorises a constant). Grad norm of patch_embed signals encoder
        # is still learning (vanishes ⇒ dead).
        last_pred_std: float = 0.0
        last_grad_norm_pe: float = 0.0
        t0 = time.time()
        bar = tqdm(
            loader,
            desc=f"[{epoch + 1}/{epochs}] videomae",
            leave=False,
            dynamic_ncols=True,
            total=steps_per_epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        for step_in_epoch, video in enumerate(bar):
            if limit_batches is not None and step_in_epoch >= limit_batches:
                break
            video = video.to(device, non_blocking=True)

            if amp_enabled:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    loss, pred, _ = model(video)
                    loss_scaled = loss / grad_accum
                loss_scaled.backward()
            else:
                loss, pred, _ = model(video)
                (loss / grad_accum).backward()

            accum_count += 1
            if accum_count >= grad_accum:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                pe_grad = model.patch_embed.proj.weight.grad
                if pe_grad is not None:
                    last_grad_norm_pe = float(pe_grad.detach().norm().item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1

            last_pred_std = float(pred.detach().float().std().item())
            running += float(loss.item()) * video.size(0)
            seen += video.size(0)
            bar.set_postfix(loss=running / max(1, seen), lr=optimizer.param_groups[0]["lr"])

        dt = (time.time() - t0) / 60.0
        avg = running / max(1, seen)
        print(
            f"Epoch {epoch + 1}/{epochs} | videomae loss {avg:.4f} | "
            f"pred_std {last_pred_std:.4f} | pe_grad_norm {last_grad_norm_pe:.4f} | "
            f"{dt:.2f} min"
        )
        wandb_log(
            {
                "videomae/train_loss": avg,
                "videomae/lr": optimizer.param_groups[0]["lr"],
                "videomae/epoch_minutes": dt,
                "videomae/pred_std": last_pred_std,
                "videomae/grad_norm_patch_embed": last_grad_norm_pe,
                "epoch": epoch + 1,
            },
            step=epoch + 1,
        )

        # Per-epoch resumable snapshot.
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "cfg_hash": cfg_hash,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "rng": capture_rng_state(),
                "videomae_train_loss": avg,
            },
            last_path,
        )

        # Periodic encoder-only snapshot for Stage 2 / probe consumption.
        if (epoch + 1) % snapshot_every == 0 or (epoch + 1) == epochs:
            atomic_torch_save(
                {
                    "encoder_state_dict": model.encoder_state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "videomae_train_loss": avg,
                },
                ckpt_path,
            )
            print(f"  Saved encoder to {ckpt_path} (epoch {epoch + 1})")

    if avg is None:
        print(f"Nothing to do: resume position ({start_epoch}) is already at epochs ({epochs}).")
    else:
        print(f"Done. Final videomae loss: {avg:.4f}")
    finish_wandb()


if __name__ == "__main__":
    main()
