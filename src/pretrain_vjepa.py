"""
Stage 1 — V-JEPA Tier-0 (frame-pair JEPA on 4-frame clips).

Per step:
  - sample one pair (i, j), i != j, per clip
  - context encoder f_theta on frame i; EMA target encoder f_theta_bar on frame j
  - narrow predictor g_phi(z_i, delta_t) -> z_hat_j
  - smooth_L1(LN(z_hat_j), LN(z_j_target).detach())
  - backprop into f_theta + g_phi; EMA update of f_theta_bar

Saves the EMA target encoder under `encoder_state_dict` in the same key layout
as MAE / iBOT, so Stage 2 loads it via the existing ViTMAEEncoder.

Usage:
    python src/pretrain_vjepa.py experiment=vjepa_pretrain
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.vjepa_dataset import VJEPAClipAug
from dataset.video_dataset import VideoFrameDataset
from models.vjepa import VJEPA
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
        if (
            p.ndim <= 1
            or name.endswith(".bias")
            or "cls_token" in name
            or "mask_token" in name
            or "pos_embed" in name
            or "dt_embed" in name
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.95))


def cosine_schedule(
    base: float,
    final: float,
    n_epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup: float = 0.0,
) -> np.ndarray:
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = n_epochs * niter_per_ep
    warmup = np.linspace(start_warmup, base, warmup_iters) if warmup_iters > 0 else np.array([])
    iters = np.arange(total_iters - warmup_iters)
    cosine = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
    return np.concatenate([warmup, cosine])


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
    aug = VJEPAClipAug(image_size=int(cfg.vjepa.image_size))
    dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.vjepa.num_frames),
        clip_transform=aug,
    )
    if cfg.dataset.get("max_samples") is not None:
        dataset.samples = dataset.samples[: int(cfg.dataset.max_samples)]
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
        variant=str(cfg.vjepa.variant),
        image_size=int(cfg.vjepa.image_size),
        predictor_dim=int(cfg.vjepa.predictor_dim),
        predictor_depth=int(cfg.vjepa.predictor_depth),
        predictor_heads=int(cfg.vjepa.predictor_heads),
        num_frames=int(cfg.vjepa.num_frames),
        loss_beta=float(cfg.training.loss_beta),
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[vjepa] trainable = {n_train:.1f}M  | total = {n_total:.1f}M")

    # --- optim / sched ---
    epochs = int(cfg.training.epochs)
    n_iter = len(loader)
    base_lr = float(cfg.training.base_lr) * int(cfg.training.batch_size) / 256.0
    optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=float(cfg.training.weight_decay))

    lr_sched = cosine_schedule(
        base=base_lr,
        final=float(cfg.training.get("final_lr", 1.0e-6)),
        n_epochs=epochs,
        niter_per_ep=n_iter,
        warmup_epochs=int(cfg.training.warmup_epochs),
        start_warmup=0.0,
    )
    momentum_sched = cosine_schedule(
        base=float(cfg.training.ema_momentum_start),
        final=float(cfg.training.ema_momentum_end),
        n_epochs=epochs,
        niter_per_ep=n_iter,
    )

    amp_enabled = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    # bf16 has fp32-equivalent dynamic range (8 exponent bits), so no GradScaler
    # is needed and overflow → Inf → NaN is not a concern under normal regimes.
    amp_dtype = torch.bfloat16

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)

    max_nonfinite_streak = int(cfg.training.get("max_nonfinite_streak", 50))
    es_patience = int(cfg.training.get("early_stop_patience", 25))
    es_min_delta = float(cfg.training.get("early_stop_min_delta", 1.0e-4))
    es_min_epochs = int(cfg.training.get("early_stop_min_epochs", 60))

    init_wandb(cfg, default_run_name=ckpt_path.stem)

    # --- resume ---
    cfg_hash = make_resume_hash(cfg, keys=("dataset", "training", "vjepa"))
    start_epoch = 0
    best_loss = float("inf")
    epochs_since_improve = 0
    resume_enabled = bool(cfg.training.get("resume", True))
    if resume_enabled and last_path.exists():
        print(f"[vjepa] found resume file: {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("cfg_hash") != cfg_hash:
            raise RuntimeError(
                f"Resume aborted: config hash mismatch.\n"
                f"  saved:   {state.get('cfg_hash')}\n"
                f"  current: {cfg_hash}\n"
                f"To start fresh, delete {last_path} or pass training.resume=false."
            )
        model.context_encoder.load_state_dict(state["context_encoder"])
        model.target_encoder.load_state_dict(state["target_encoder"])
        model.predictor.load_state_dict(state["predictor"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best_loss = float(state.get("best_loss", float("inf")))
        epochs_since_improve = int(state.get("epochs_since_improve", 0))
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(
            f"[vjepa] resuming from epoch {start_epoch + 1}/{epochs} "
            f"(best_loss={best_loss:.4f}, no_improve={epochs_since_improve})"
        )
    elif not resume_enabled and last_path.exists():
        print(f"[vjepa] training.resume=false → ignoring {last_path}")

    grad_clip = float(cfg.training.get("grad_clip", 3.0))

    avg_loss: Optional[float] = None
    nonfinite_streak = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        model.target_encoder.eval()
        running = {"loss": 0.0, "cos": 0.0, "tgt_std": 0.0}
        seen = 0
        nonfinite_this_epoch = 0
        t0 = time.time()
        bar = tqdm(loader, desc=f"[{epoch + 1}/{epochs}] vjepa", leave=False, dynamic_ncols=True)
        for it, batch in enumerate(bar):
            clip, _ = batch                                   # ignore labels
            clip = clip.to(device, non_blocking=True)         # (B, T, 3, H, W)

            global_step = epoch * n_iter + it
            for pg in optimizer.param_groups:
                pg["lr"] = lr_sched[global_step]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(clip)
                loss = out["loss"]

            # Guard: skip backward + optimizer + EMA on non-finite loss so a
            # single bad batch can't poison the EMA target encoder permanently.
            if not torch.isfinite(loss):
                nonfinite_this_epoch += 1
                nonfinite_streak += 1
                if nonfinite_streak >= max_nonfinite_streak:
                    raise RuntimeError(
                        f"V-JEPA training aborted: {nonfinite_streak} non-finite steps in a row "
                        f"(epoch {epoch + 1}, iter {it})."
                    )
                continue
            nonfinite_streak = 0

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.ema_step(float(momentum_sched[global_step]))

            b = clip.shape[0]
            running["loss"] += float(loss.item()) * b
            running["cos"] += float(out["cos_sim"].item()) * b
            running["tgt_std"] += float(out["target_std"].item()) * b
            seen += b
            bar.set_postfix(
                loss=running["loss"] / max(1, seen),
                cos=running["cos"] / max(1, seen),
                lr=optimizer.param_groups[0]["lr"],
            )

        dt = (time.time() - t0) / 60.0
        avg_loss = running["loss"] / max(1, seen) if seen > 0 else float("nan")
        avg_cos = running["cos"] / max(1, seen) if seen > 0 else float("nan")
        avg_tgt = running["tgt_std"] / max(1, seen) if seen > 0 else float("nan")
        loss_finite = bool(np.isfinite(avg_loss))

        improved = loss_finite and (avg_loss < best_loss - es_min_delta)
        if improved:
            best_loss = avg_loss
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | cos {avg_cos:.4f} "
            f"| tgt_std {avg_tgt:.4f} | {dt:.2f} min "
            f"| best {best_loss:.4f} (no_improve={epochs_since_improve}) "
            f"| nonfinite {nonfinite_this_epoch}"
        )
        wandb_log(
            {
                "vjepa/loss": avg_loss,
                "vjepa/cos_sim": avg_cos,
                "vjepa/target_std": avg_tgt,
                "vjepa/lr": optimizer.param_groups[0]["lr"],
                "vjepa/ema_momentum": float(momentum_sched[(epoch + 1) * n_iter - 1]),
                "vjepa/epoch_minutes": dt,
                "vjepa/best_loss": best_loss,
                "vjepa/nonfinite_steps": nonfinite_this_epoch,
                "epoch": epoch + 1,
            },
            step=epoch + 1,
        )

        # Per-epoch resumable snapshot. Skip if the epoch's loss was non-finite,
        # so a single corrupted epoch can't overwrite a clean resume state.
        if loss_finite:
            atomic_torch_save(
                {
                    "context_encoder": model.context_encoder.state_dict(),
                    "target_encoder": model.target_encoder.state_dict(),
                    "predictor": model.predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "cfg_hash": cfg_hash,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "rng": capture_rng_state(),
                    "vjepa_train_loss": avg_loss,
                    "best_loss": best_loss,
                    "epochs_since_improve": epochs_since_improve,
                },
                last_path,
            )
        else:
            print(f"  Skipping {last_path.name} update (non-finite epoch loss).")

        # Best-by-loss encoder snapshot for Stage 2 (EMA target).
        if improved:
            atomic_torch_save(
                {
                    "encoder_state_dict": model.target_encoder.encoder_state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch + 1,
                    "vjepa_train_loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"  New best loss {avg_loss:.4f} → saved target encoder to {ckpt_path}")

        # Early stop on plateau, gated by a minimum number of epochs so we don't
        # bail before warmup + the early plateau have settled.
        if (epoch + 1) >= es_min_epochs and epochs_since_improve >= es_patience:
            print(
                f"Early stop at epoch {epoch + 1}: no improvement > {es_min_delta:g} "
                f"for {epochs_since_improve} epochs (best={best_loss:.4f})."
            )
            break

    if avg_loss is None:
        print(f"Nothing to do: resume position ({start_epoch}) is already at epochs ({epochs}).")
    else:
        print(f"Done. Best V-JEPA loss: {best_loss:.4f} | last epoch loss: {avg_loss:.4f}")
    finish_wandb()


if __name__ == "__main__":
    main()
