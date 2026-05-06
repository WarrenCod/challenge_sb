"""
Stage 1 — iBOT (Image BERT pretraining with online tokenizer).

Self-supervised on individual frames. Two losses:
  - DINO: cross-entropy between student CLS softmax and teacher CLS softmax,
    across all (student_crop, teacher_global) pairs except matching ones.
  - MIM:  cross-entropy between student patch logits at masked positions and
    teacher patch logits (computed on the full, unmasked global crop).

Teacher = EMA of student. Heads are 3-MLP + L2-norm + weight-norm linear (DINO).

Saves an `encoder_state_dict` in the same key layout as MAE's, so Stage 2 can
load it via the existing ViTMAEEncoder without renaming.

Usage:
    python src/pretrain_ibot.py experiment=ibot_pretrain
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.frame_dataset import collect_frame_paths, collect_frame_paths_deep
from dataset.ibot_dataset import MultiCropAug, ibot_collate, iBOTFrameDataset
from models.ibot import MultiCropEncoder, ema_update, iBOTHead, iBOTViT
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


def build_optimizer(model: nn.Module, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no decay on 1-D params (bias, LayerNorm, CLS / mask tokens, pos-embeds)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "cls_token" in name or "mask_token" in name or "pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))


def cosine_schedule(base: float, final: float, n_epochs: int, niter_per_ep: int, warmup_epochs: int = 0, start_warmup: float = 0.0) -> np.ndarray:
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = n_epochs * niter_per_ep
    if warmup_iters > 0:
        warmup = np.linspace(start_warmup, base, warmup_iters)
    else:
        warmup = np.array([])
    iters = np.arange(total_iters - warmup_iters)
    cosine = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
    return np.concatenate([warmup, cosine])


def teacher_temp_schedule(warmup_temp: float, final_temp: float, warmup_epochs: int, n_epochs: int) -> np.ndarray:
    warm = np.linspace(warmup_temp, final_temp, max(1, warmup_epochs))
    rest = np.ones(max(0, n_epochs - warmup_epochs)) * final_temp
    return np.concatenate([warm, rest])


class iBOTLoss(nn.Module):
    """DINO + MIM joint loss with EMA-tracked centers (one per loss)."""

    def __init__(self, out_dim: int, n_global_crops: int = 2, n_local_crops: int = 4, center_momentum: float = 0.9, student_temp: float = 0.1, mim_weight: float = 1.0) -> None:
        super().__init__()
        self.n_global = n_global_crops
        self.n_local = n_local_crops
        self.center_momentum = center_momentum
        self.student_temp = student_temp
        self.mim_weight = mim_weight
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def update_centers(self, teacher_cls: torch.Tensor, teacher_patch: torch.Tensor) -> None:
        # teacher_cls:    (n_global * B, D)
        # teacher_patch:  (n_global * B, N, D)
        cls_mean = teacher_cls.mean(dim=0, keepdim=True)
        patch_mean = teacher_patch.mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(cls_mean, alpha=1.0 - self.center_momentum)
        self.center_patch.mul_(self.center_momentum).add_(patch_mean, alpha=1.0 - self.center_momentum)

    def forward(
        self,
        student_cls: torch.Tensor,         # ((n_global+n_local) * B, D)
        student_patch: torch.Tensor,       # (n_global * B, N, D)
        teacher_cls: torch.Tensor,         # (n_global * B, D)        — detached
        teacher_patch: torch.Tensor,       # (n_global * B, N, D)     — detached
        masks: torch.Tensor,               # (n_global * B, N) bool
        teacher_temp: float,
    ) -> dict:
        n_total = self.n_global + self.n_local
        b = teacher_cls.shape[0] // self.n_global

        # --- DINO loss ---
        student_cls_logits = student_cls / self.student_temp                       # ((n_g+n_l)*B, D)
        teacher_cls_softmax = F.softmax((teacher_cls - self.center) / teacher_temp, dim=-1)
        teacher_cls_softmax = teacher_cls_softmax.detach()
        student_cls_log = F.log_softmax(student_cls_logits, dim=-1)

        # Reshape: (n_views, B, D)
        s_cls = student_cls_log.view(n_total, b, -1)
        t_cls = teacher_cls_softmax.view(self.n_global, b, -1)

        dino_loss = 0.0
        n_pairs = 0
        for ti in range(self.n_global):
            for si in range(n_total):
                if si == ti:
                    continue  # skip same-crop self-pair
                dino_loss = dino_loss - (t_cls[ti] * s_cls[si]).sum(dim=-1).mean()
                n_pairs += 1
        dino_loss = dino_loss / max(1, n_pairs)

        # --- MIM loss (only at masked positions of the global crops) ---
        teacher_patch_softmax = F.softmax((teacher_patch - self.center_patch) / teacher_temp, dim=-1).detach()
        student_patch_log = F.log_softmax(student_patch / self.student_temp, dim=-1)
        mim_per = -(teacher_patch_softmax * student_patch_log).sum(dim=-1)        # (n_global*B, N)
        mask_f = masks.float()
        mim_loss = (mim_per * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        total = dino_loss + self.mim_weight * mim_loss

        # Update centers from THIS step's teacher outputs.
        self.update_centers(teacher_cls, teacher_patch)

        return {"loss": total, "dino": dino_loss.detach(), "mim": mim_loss.detach()}


def cancel_last_layer_gradients(student: nn.Module, freeze_epochs: int, current_epoch: int) -> None:
    """Zero out gradients on the iBOT head's last linear for the first N epochs (DINO trick)."""
    if current_epoch >= freeze_epochs:
        return
    for n, p in student.named_parameters():
        if "last_layer" in n and p.grad is not None:
            p.grad = None


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
    pretrain_dirs = cfg.dataset.get("pretrain_dirs")
    if pretrain_dirs:
        roots = [Path(d).resolve() for d in pretrain_dirs]
        frame_paths = collect_frame_paths_deep(roots)
        print(f"[ibot] {len(frame_paths)} frames pooled from {len(roots)} roots:")
        for r in roots:
            print(f"  - {r}")
    else:
        train_dir = Path(cfg.dataset.train_dir).resolve()
        frame_paths = collect_frame_paths(train_dir)
        print(f"[ibot] {len(frame_paths)} frames in {train_dir}")
    if cfg.dataset.get("max_samples") is not None:
        frame_paths = frame_paths[: int(cfg.dataset.max_samples)]
        print(f"[ibot] truncated to max_samples={cfg.dataset.max_samples}")

    aug = MultiCropAug(
        global_size=int(cfg.ibot.global_size),
        local_size=int(cfg.ibot.local_size),
        n_local=int(cfg.ibot.n_local_crops),
        global_scale=tuple(cfg.ibot.global_scale),
        local_scale=tuple(cfg.ibot.local_scale),
    )
    grid = int(cfg.ibot.global_size) // int(cfg.ibot.patch_size)
    dataset = iBOTFrameDataset(
        frame_paths=frame_paths,
        aug=aug,
        global_grid_size=grid,
        mask_ratio_min=float(cfg.ibot.mask_ratio_min),
        mask_ratio_max=float(cfg.ibot.mask_ratio_max),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(int(cfg.training.num_workers) > 0),
        collate_fn=ibot_collate,
    )

    # --- models: student / teacher / heads ---
    # Build student and teacher independently and copy state_dicts. deepcopy doesn't
    # work on modules using nn.utils.weight_norm (pytorch/pytorch#103001).
    out_dim = int(cfg.ibot.out_dim)

    def _make_backbone() -> iBOTViT:
        return iBOTViT(
            img_size=int(cfg.ibot.global_size),
            patch_size=int(cfg.ibot.patch_size),
            embed_dim=int(cfg.ibot.embed_dim),
            depth=int(cfg.ibot.depth),
            num_heads=int(cfg.ibot.num_heads),
            mlp_ratio=float(cfg.ibot.mlp_ratio),
            drop_path_rate=float(cfg.ibot.drop_path),
        )

    def _make_head() -> iBOTHead:
        return iBOTHead(in_dim=int(cfg.ibot.embed_dim), out_dim=out_dim)

    student_backbone = _make_backbone().to(device)
    teacher_backbone = _make_backbone().to(device)
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad = False

    student_cls_head = _make_head().to(device)
    student_patch_head = _make_head().to(device)
    teacher_cls_head = _make_head().to(device)
    teacher_patch_head = _make_head().to(device)
    teacher_cls_head.load_state_dict(student_cls_head.state_dict())
    teacher_patch_head.load_state_dict(student_patch_head.state_dict())
    for p in list(teacher_cls_head.parameters()) + list(teacher_patch_head.parameters()):
        p.requires_grad = False

    student = MultiCropEncoder(student_backbone, student_cls_head, student_patch_head)
    teacher = MultiCropEncoder(teacher_backbone, teacher_cls_head, teacher_patch_head)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
    print(f"[ibot] student trainable params = {n_params:.1f}M")

    # --- optim / sched / amp ---
    epochs = int(cfg.training.epochs)
    n_iter = len(loader)
    base_lr = float(cfg.training.base_lr) * int(cfg.training.batch_size) / 256.0
    optimizer = build_optimizer(student, base_lr=base_lr, weight_decay=float(cfg.training.weight_decay))

    lr_sched = cosine_schedule(
        base=base_lr,
        final=float(cfg.training.final_lr),
        n_epochs=epochs,
        niter_per_ep=n_iter,
        warmup_epochs=int(cfg.training.warmup_epochs),
        start_warmup=0.0,
    )
    wd_sched = cosine_schedule(
        base=float(cfg.training.weight_decay),
        final=float(cfg.training.final_weight_decay),
        n_epochs=epochs,
        niter_per_ep=n_iter,
    )
    momentum_sched = cosine_schedule(
        base=float(cfg.training.ema_momentum),
        final=1.0,
        n_epochs=epochs,
        niter_per_ep=n_iter,
    )
    teacher_temps = teacher_temp_schedule(
        warmup_temp=float(cfg.training.warmup_teacher_temp),
        final_temp=float(cfg.training.teacher_temp),
        warmup_epochs=int(cfg.training.warmup_teacher_temp_epochs),
        n_epochs=epochs,
    )

    loss_fn = iBOTLoss(
        out_dim=out_dim,
        n_global_crops=2,
        n_local_crops=int(cfg.ibot.n_local_crops),
        center_momentum=float(cfg.training.center_momentum),
        student_temp=float(cfg.training.student_temp),
        mim_weight=float(cfg.training.mim_weight),
    ).to(device)

    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device=device.type) if amp_enabled else None

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)
    snapshot_every = int(cfg.training.get("snapshot_every", 25))

    init_wandb(cfg, default_run_name=ckpt_path.stem)

    # --- resume ---
    cfg_hash = make_resume_hash(cfg, keys=("model", "dataset", "training", "ibot"))
    start_epoch = 0
    resume_enabled = bool(cfg.training.get("resume", True))
    if resume_enabled and last_path.exists():
        print(f"[ibot] found resume file: {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("cfg_hash") != cfg_hash:
            raise RuntimeError(
                f"Resume aborted: config hash mismatch.\n"
                f"  saved:   {state.get('cfg_hash')}\n"
                f"  current: {cfg_hash}\n"
                f"To start fresh, either delete {last_path} or pass training.resume=false."
            )
        student_backbone.load_state_dict(state["student_backbone"])
        teacher_backbone.load_state_dict(state["teacher_backbone"])
        student_cls_head.load_state_dict(state["student_cls_head"])
        student_patch_head.load_state_dict(state["student_patch_head"])
        teacher_cls_head.load_state_dict(state["teacher_cls_head"])
        teacher_patch_head.load_state_dict(state["teacher_patch_head"])
        loss_fn.center.copy_(state["loss_center"])
        loss_fn.center_patch.copy_(state["loss_center_patch"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[ibot] resuming from epoch {start_epoch + 1}/{epochs}")
    elif not resume_enabled and last_path.exists():
        print(f"[ibot] training.resume=false → ignoring {last_path}")

    grad_clip = float(cfg.training.get("grad_clip", 3.0))
    freeze_last_epochs = int(cfg.training.get("freeze_last_layer_epochs", 1))

    avg_loss: Optional[float] = None
    for epoch in range(start_epoch, epochs):
        student.train()
        teacher.eval()
        running = {"loss": 0.0, "dino": 0.0, "mim": 0.0}
        seen = 0
        t0 = time.time()
        bar = tqdm(loader, desc=f"[{epoch + 1}/{epochs}] ibot", leave=False, dynamic_ncols=True)
        for it, (crops, masks) in enumerate(bar):
            global_step = epoch * n_iter + it
            for pg in optimizer.param_groups:
                pg["lr"] = lr_sched[global_step]
                if pg["weight_decay"] != 0.0:
                    pg["weight_decay"] = wd_sched[global_step]

            crops = [c.to(device, non_blocking=True) for c in crops]
            masks = [m.to(device, non_blocking=True) for m in masks]

            optimizer.zero_grad(set_to_none=True)

            stepped = True
            with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                # Student: all crops + masks for globals, get patches for globals.
                student_masks = masks + [None] * int(cfg.ibot.n_local_crops)
                s_cls, s_patch = student(crops, masks=student_masks, return_patch_for_globals=True)
                # Teacher: only the 2 global crops, NO mask, NO grad.
                with torch.no_grad():
                    t_cls, t_patch = teacher(crops[:2], masks=None, return_patch_for_globals=True)
                    t_cls = t_cls.detach()
                    t_patch = t_patch.detach()

                masks_stack = torch.cat(masks, dim=0)  # (n_global*B, N)
                losses = loss_fn(
                    student_cls=s_cls,
                    student_patch=s_patch,
                    teacher_cls=t_cls,
                    teacher_patch=t_patch,
                    masks=masks_stack,
                    teacher_temp=float(teacher_temps[epoch]),
                )
                loss = losses["loss"]

            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                cancel_last_layer_gradients(student, freeze_last_epochs, epoch)
                prev = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= prev
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                cancel_last_layer_gradients(student, freeze_last_epochs, epoch)
                optimizer.step()

            if stepped:
                ema_update(student_backbone, teacher_backbone, momentum_sched[global_step])
                ema_update(student_cls_head, teacher_cls_head, momentum_sched[global_step])
                ema_update(student_patch_head, teacher_patch_head, momentum_sched[global_step])

            b = crops[0].shape[0]
            for k in running:
                running[k] += float(losses[k].item()) * b
            seen += b
            bar.set_postfix(loss=running["loss"] / max(1, seen), lr=optimizer.param_groups[0]["lr"])

        dt = (time.time() - t0) / 60.0
        avg_loss = running["loss"] / max(1, seen)
        avg_dino = running["dino"] / max(1, seen)
        avg_mim = running["mim"] / max(1, seen)
        print(f"Epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | dino {avg_dino:.4f} | mim {avg_mim:.4f} | {dt:.2f} min")
        wandb_log(
            {
                "ibot/loss": avg_loss,
                "ibot/dino": avg_dino,
                "ibot/mim": avg_mim,
                "ibot/lr": optimizer.param_groups[0]["lr"],
                "ibot/teacher_temp": float(teacher_temps[epoch]),
                "ibot/ema_momentum": float(momentum_sched[(epoch + 1) * n_iter - 1]),
                "ibot/epoch_minutes": dt,
                "epoch": epoch + 1,
            },
            step=epoch + 1,
        )

        # Per-epoch resumable snapshot.
        atomic_torch_save(
            {
                "student_backbone": student_backbone.state_dict(),
                "teacher_backbone": teacher_backbone.state_dict(),
                "student_cls_head": student_cls_head.state_dict(),
                "student_patch_head": student_patch_head.state_dict(),
                "teacher_cls_head": teacher_cls_head.state_dict(),
                "teacher_patch_head": teacher_patch_head.state_dict(),
                "loss_center": loss_fn.center.detach().cpu(),
                "loss_center_patch": loss_fn.center_patch.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "cfg_hash": cfg_hash,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "rng": capture_rng_state(),
                "ibot_train_loss": avg_loss,
            },
            last_path,
        )

        # Periodic encoder-only snapshot for Stage 2 (use the EMA *teacher* — better
        # downstream features per DINO/iBOT recipe).
        if (epoch + 1) % snapshot_every == 0 or (epoch + 1) == epochs:
            atomic_torch_save(
                {
                    "encoder_state_dict": teacher_backbone.encoder_state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "ibot_train_loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"  Saved teacher encoder to {ckpt_path} (epoch {epoch + 1})")

    if avg_loss is None:
        print(f"Nothing to do: resume position ({start_epoch}) is already at epochs ({epochs}).")
    else:
        print(f"Done. Final iBOT loss: {avg_loss:.4f}")
    finish_wandb()


if __name__ == "__main__":
    main()
