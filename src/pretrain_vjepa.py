"""
V-JEPA self-supervised pretraining on unlabeled video clips.

Usage (from repo root, in a tmux session so SSH drops don't kill it):

    tmux new -s vjepa
    python src/pretrain_vjepa.py experiment=vjepa_pretrain

Smoke-test (a handful of steps, one epoch) before the overnight run:

    python src/pretrain_vjepa.py experiment=vjepa_pretrain \\
        training.epochs=1 training.limit_batches=30

Loss alone doesn't prove SSL worked — watch ``emb_std`` in the log: if it
collapses toward zero, the representations are trivial. Also run a linear
probe after training (separate experiment).
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_ssl_dataset import UnlabeledVideoClipDataset
from models.vjepa import VJEPA
from models.vjepa_masking import sample_mask
from utils import set_seed


def build_ssl_transforms(image_size: int):
    """Light augmentation — JEPA relies on masking, not heavy photometric aug.

    No horizontal flip (SSv2 labels are direction-sensitive). Small colour
    jitter to reduce low-level shortcut learning.
    """
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_optimizer(
    model: nn.Module, base_lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW, no decay on 1-D params / mask_token / pos_embed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or "mask_token" in name
            or "pos_embed" in name
        ):
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


def ema_momentum(step: int, total_steps: int, m_start: float, m_end: float) -> float:
    """Cosine schedule from m_start at step 0 to m_end at total_steps."""
    if total_steps <= 0:
        return m_end
    t = min(step / total_steps, 1.0)
    return m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * t))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- data ---
    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()
    canonical_classes = {p.name for p in val_dir.iterdir() if p.is_dir()}
    transform = build_ssl_transforms(image_size=int(cfg.vjepa.image_size))

    dataset = UnlabeledVideoClipDataset(
        root_dir=train_dir,
        num_frames=int(cfg.vjepa.num_frames),
        transform=transform,
        canonical_classes=canonical_classes,
        seed=int(cfg.dataset.seed),
    )
    print(f"[vjepa] {len(dataset)} clips in {train_dir}")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(int(cfg.training.num_workers) > 0),
    )

    # --- model ---
    model = VJEPA(
        num_frames=int(cfg.vjepa.num_frames),
        img_size=int(cfg.vjepa.image_size),
        tubelet_time=int(cfg.vjepa.tubelet_time),
        tubelet_size=int(cfg.vjepa.tubelet_size),
        embed_dim=int(cfg.vjepa.embed_dim),
        depth=int(cfg.vjepa.depth),
        num_heads=int(cfg.vjepa.num_heads),
        mlp_ratio=float(cfg.vjepa.mlp_ratio),
        predictor_embed_dim=int(cfg.vjepa.predictor_embed_dim),
        predictor_depth=int(cfg.vjepa.predictor_depth),
        predictor_num_heads=int(cfg.vjepa.predictor_num_heads),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[vjepa] model = {n_trainable:.1f}M trainable / {n_total:.1f}M total "
        f"(target encoder frozen)"
    )
    print(
        f"[vjepa] grid = ({model.context_encoder.t_grid}, "
        f"{model.context_encoder.h_grid}, {model.context_encoder.w_grid}); "
        f"{model.context_encoder.num_tokens} tokens"
    )

    # --- optim / sched / amp ---
    epochs = int(cfg.training.epochs)
    steps_per_epoch = len(loader)
    limit_batches: Optional[int] = cfg.training.get("limit_batches", None)
    if limit_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(limit_batches))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg.training.get("warmup_epochs", 0)) * steps_per_epoch

    batch_size = int(cfg.training.batch_size)
    base_lr = float(cfg.training.base_lr) * batch_size / 256.0
    optimizer = build_optimizer(
        model, base_lr=base_lr, weight_decay=float(cfg.training.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(total_steps, warmup_steps)
    )

    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16  # bf16: no grad scaler needed, more stable for SSL

    # --- masking RNG ---
    mask_rng = random.Random(int(cfg.dataset.seed) + 7919)
    t_grid = model.context_encoder.t_grid
    h_grid = model.context_encoder.h_grid
    w_grid = model.context_encoder.w_grid

    # --- ema schedule ---
    m_start = float(cfg.training.ema_momentum_start)
    m_end = float(cfg.training.ema_momentum_end)

    # --- misc ---
    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_every = int(cfg.training.get("snapshot_every", 10))
    diag_every = int(cfg.training.get("diag_every", 50))
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    global_step = 0
    strat_counts = {"spatial": 0, "temporal": 0}

    for epoch in range(epochs):
        model.context_encoder.train()
        model.predictor.train()
        # Target encoder always in eval: no dropout, no stochastic depth.
        model.target_encoder.eval()

        running, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(
            loader,
            desc=f"[{epoch + 1}/{epochs}] vjepa",
            leave=False,
            dynamic_ncols=True,
            total=steps_per_epoch,
        )
        for step_in_epoch, video in enumerate(bar):
            if limit_batches is not None and step_in_epoch >= limit_batches:
                break
            video = video.to(device, non_blocking=True)

            # Sample one mask for the whole batch (simpler than per-sample padding).
            context_ids, target_ids, strategy = sample_mask(
                t_grid,
                h_grid,
                w_grid,
                spatial_ratio=float(cfg.vjepa.spatial_mask_ratio),
                spatial_n_blocks=int(cfg.vjepa.spatial_mask_n_blocks),
                temporal_keep_first=int(cfg.vjepa.temporal_keep_first),
                p_spatial=float(cfg.vjepa.p_spatial),
                rng=mask_rng,
            )
            strat_counts[strategy] += 1
            context_ids = context_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    z_pred, z_tgt = model(video, context_ids, target_ids)
                    loss = model.loss(z_pred, z_tgt)
                loss.backward()
            else:
                z_pred, z_tgt = model(video, context_ids, target_ids)
                loss = model.loss(z_pred, z_tgt)
                loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            # EMA update for target encoder.
            m = ema_momentum(global_step, total_steps, m_start, m_end)
            model.update_target(m)

            running += float(loss.item()) * video.size(0)
            seen += video.size(0)
            postfix = {
                "loss": running / max(1, seen),
                "lr": optimizer.param_groups[0]["lr"],
                "ema": m,
                "mask": strategy[:4],
            }

            if global_step % diag_every == 0:
                with torch.no_grad():
                    emb_std = model.embedding_std(video[:4])
                postfix["emb_std"] = emb_std
                if emb_std < 0.01:
                    tqdm.write(
                        f"[vjepa][step {global_step}] WARN emb_std={emb_std:.4f} — "
                        f"possible collapse."
                    )

            bar.set_postfix(postfix)
            global_step += 1

        dt = (time.time() - t0) / 60.0
        avg = running / max(1, seen)
        print(
            f"Epoch {epoch + 1}/{epochs} | vjepa loss {avg:.4f} | "
            f"{dt:.2f} min | "
            f"masks: spatial={strat_counts['spatial']} temporal={strat_counts['temporal']}"
        )

        if (epoch + 1) % snapshot_every == 0 or (epoch + 1) == epochs:
            torch.save(
                {
                    "context_encoder_state_dict": model.context_encoder_state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "vjepa_train_loss": avg,
                },
                ckpt_path,
            )
            print(f"  Saved context encoder to {ckpt_path} (epoch {epoch + 1})")

    print(f"Done. Final vjepa loss: {avg:.4f}")


if __name__ == "__main__":
    main()
