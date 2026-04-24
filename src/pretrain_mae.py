"""
Stage 1 — Masked Autoencoder pretraining on individual frames.

Usage (from repo root, after `uv sync`):

    python src/pretrain_mae.py experiment=mae_pretrain

No validation split: MAE reconstruction loss tracks training loss closely, and
the only meaningful transfer signal is the Stage-2 linear probe. Instead, we
log train recon loss per epoch and periodically snapshot the encoder.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.frame_dataset import FrameDataset
from models.mae_vit import MaskedAutoencoderViT
from utils import (
    atomic_torch_save,
    build_transforms,
    capture_rng_state,
    make_resume_hash,
    restore_rng_state,
    set_seed,
)


def build_mae_transforms(image_size: int = 224) -> "object":
    """RandomResizedCrop + ColorJitter + ToTensor + 0.5/0.5 norm. No hflip (SSv2 labels are direction-sensitive)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_optimizer(model: nn.Module, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no decay on 1-D params (bias, LayerNorm, CLS/mask tokens, pos-embeds)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "cls_token" in name or "mask_token" in name:
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

    # --- data ---
    train_dir = Path(cfg.dataset.train_dir).resolve()
    transform = build_mae_transforms(image_size=int(cfg.mae.image_size))
    dataset = FrameDataset(
        root_dir=train_dir,
        transform=transform,
        max_samples=cfg.dataset.get("max_samples"),
    )
    print(f"[mae] {len(dataset)} frames in {train_dir}")

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
    model = MaskedAutoencoderViT(
        img_size=int(cfg.mae.image_size),
        patch_size=int(cfg.mae.patch_size),
        embed_dim=int(cfg.mae.embed_dim),
        depth=int(cfg.mae.depth),
        num_heads=int(cfg.mae.num_heads),
        decoder_embed_dim=int(cfg.mae.decoder_embed_dim),
        decoder_depth=int(cfg.mae.decoder_depth),
        decoder_num_heads=int(cfg.mae.decoder_num_heads),
        mask_ratio=float(cfg.mae.mask_ratio),
        norm_pix_loss=bool(cfg.mae.norm_pix_loss),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[mae] model = {n_params:.1f}M params")

    # --- optim / sched / amp ---
    epochs = int(cfg.training.epochs)
    total_steps = epochs * len(loader)
    warmup_steps = int(cfg.training.get("warmup_epochs", 0)) * len(loader)

    batch_size = int(cfg.training.batch_size)
    base_lr = float(cfg.training.base_lr) * batch_size / 256.0
    optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=float(cfg.training.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warmup_steps))

    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(device=device.type) if amp_enabled else None

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_every = int(cfg.training.get("snapshot_every", 50))
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)

    # --- resume from last.pt if present and config matches ---
    cfg_hash = make_resume_hash(cfg)
    start_epoch = 0
    resume_enabled = bool(cfg.training.get("resume", True))
    if resume_enabled and last_path.exists():
        print(f"[mae] found resume file: {last_path}")
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
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[mae] resuming from epoch {start_epoch + 1}/{epochs}")
    elif not resume_enabled and last_path.exists():
        print(f"[mae] training.resume=false → ignoring {last_path}")

    # --- train ---
    avg: Optional[float] = None
    for epoch in range(start_epoch, epochs):
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(loader, desc=f"[{epoch + 1}/{epochs}] mae", leave=False, dynamic_ncols=True)
        for imgs in bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            stepped = True
            if amp_enabled and scaler is not None:
                with autocast(device_type=device.type, dtype=torch.float16):
                    loss, _, _ = model(imgs)
                scaler.scale(loss).backward()
                prev = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= prev
            else:
                loss, _, _ = model(imgs)
                loss.backward()
                optimizer.step()

            if stepped:
                scheduler.step()

            running += float(loss.item()) * imgs.size(0)
            seen += imgs.size(0)
            bar.set_postfix(loss=running / max(1, seen), lr=optimizer.param_groups[0]["lr"])

        dt = (time.time() - t0) / 60.0
        avg = running / max(1, seen)
        print(f"Epoch {epoch + 1}/{epochs} | mae loss {avg:.4f} | {dt:.2f} min")

        # Per-epoch resumable snapshot (full state, atomic, overwritten).
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "cfg_hash": cfg_hash,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "rng": capture_rng_state(),
                "mae_train_loss": avg,
            },
            last_path,
        )

        # Periodic snapshot + always keep latest encoder-only file.
        if (epoch + 1) % snapshot_every == 0 or (epoch + 1) == epochs:
            atomic_torch_save(
                {
                    "encoder_state_dict": model.encoder_state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "mae_train_loss": avg,
                },
                ckpt_path,
            )
            print(f"  Saved encoder to {ckpt_path} (epoch {epoch + 1})")

    if avg is None:
        print(f"Nothing to do: resume position ({start_epoch}) is already at epochs ({epochs}).")
    else:
        print(f"Done. Final MAE loss: {avg:.4f}")


if __name__ == "__main__":
    main()
