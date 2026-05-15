"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.multi_clip_dataset import MultiClipVideoDataset
from dataset.pseudo_label_dataset import PseudoLabelVideoDataset
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cmt import CMT
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.aux.pair_direction import PairDirectionAuxHead
from models.aux.predict_next_cls import PredictNextCLSAuxHead
from models.aux.predict_prev_cls import PredictPrevCLSAuxHead
from models.modular import attach_aux_head, build_modular_model
from utils import (
    ModelEMA,
    atomic_torch_save,
    build_llrd_param_groups,
    build_strong_clip_transform,
    build_transforms,
    build_two_group_param_groups,
    capture_rng_state,
    cutmix_batch,
    finish_wandb,
    init_wandb,
    make_resume_hash,
    mixup_batch,
    restore_rng_state,
    set_seed,
    split_train_val,
    wandb_log,
)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=cfg.model.pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=cfg.model.pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "modular":
        return build_modular_model(cfg.model, num_classes=num_classes)
    if name == "cmt":
        return CMT(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=bool(cfg.model.pretrained),
            c_prime=int(cfg.model.c_prime),
            motion_widths=list(cfg.model.motion_widths),
            d=int(cfg.model.d),
            set_num_blocks=int(cfg.model.set_num_blocks),
            set_num_heads=int(cfg.model.set_num_heads),
            set_ffn_mult=int(cfg.model.set_ffn_mult),
            head_hidden=int(cfg.model.head_hidden),
            dropout=float(cfg.model.dropout),
        )

    raise ValueError(f"Unknown model.name: {name}")


def load_teacher(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Rebuild a frozen teacher from a checkpoint produced by this script.

    The teacher's architecture is taken from the embedded Hydra config in the
    checkpoint, so it can differ from the student. The model is set to eval()
    and ``requires_grad_(False)`` so it contributes no gradients and its
    BN/dropout stay deterministic.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"Teacher checkpoint missing 'config': {checkpoint_path}")
    teacher_cfg = OmegaConf.create(ckpt["config"])
    teacher = build_model(teacher_cfg)
    # Use strict=False so we tolerate aux-head keys (predict_next_cls_head,
    # predict_prev_cls_head, aux_head) that EMA may have mirrored into the
    # saved state_dict but that aren't part of the rebuilt eval-time model.
    missing, unexpected = teacher.load_state_dict(ckpt["model_state_dict"], strict=False)
    aux_like = lambda k: any(k.startswith(p) for p in ("predict_next_cls_head.", "predict_prev_cls_head.", "aux_head."))
    real_missing = [k for k in missing if not aux_like(k)]
    real_unexpected = [k for k in unexpected if not aux_like(k)]
    if real_missing or real_unexpected:
        print(
            f"[train] teacher load_state_dict: "
            f"missing={real_missing[:3]} ({len(real_missing)}), "
            f"unexpected={real_unexpected[:3]} ({len(real_unexpected)})"
        )
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(device)
    return teacher


class EnsembleTeacher(nn.Module):
    """Wraps several frozen teachers; returns logits-space tensor for KD.

    Two combine modes (the downstream code in `_forward_loss` does
    ``softmax(teacher_out / T)``, so we always return something on a
    logits-like scale):

    - ``mean_logits``    : average per-teacher logits.
    - ``softmax_mean``   : average per-teacher softmax probabilities, then
                           ``log()`` so the downstream ``softmax(/T)`` divides
                           the (log-)probabilities by T as expected.
    """

    def __init__(self, teachers: list[nn.Module], combine: str = "softmax_mean") -> None:
        super().__init__()
        if len(teachers) == 0:
            raise ValueError("EnsembleTeacher needs at least one teacher.")
        if combine not in ("mean_logits", "softmax_mean"):
            raise ValueError(f"unknown combine mode: {combine}")
        self.teachers = nn.ModuleList(teachers)
        self.combine = combine
        for t in self.teachers:
            t.eval()
            t.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [t(x) for t in self.teachers]
        if self.combine == "mean_logits":
            return torch.stack(outs, dim=0).mean(dim=0)
        # softmax_mean: average probabilities, return as log-probs.
        probs = torch.stack([F.softmax(o, dim=-1) for o in outs], dim=0).mean(dim=0)
        return torch.log(probs.clamp_min(1e-8))


def _sample_shared_mix(
    clip_a: torch.Tensor,
    clip_b: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup OR CutMix to both clips with a SHARED (lam, perm, box).

    Returns (clip_a_mixed, clip_b_mixed, y_a, y_b, lam). When neither lever is
    enabled, returns the originals with lam=1.0 and y_a==y_b==labels.
    """
    if mixup_alpha > 0 and cutmix_alpha > 0:
        use_cutmix = random.random() >= 0.5
    elif mixup_alpha > 0:
        use_cutmix = False
    elif cutmix_alpha > 0:
        use_cutmix = True
    else:
        return clip_a, clip_b, labels, labels, 1.0

    B = clip_a.size(0)
    if use_cutmix:
        lam_raw = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        perm = torch.randperm(B, device=clip_a.device)
        h, w = clip_a.shape[-2], clip_a.shape[-1]
        cut_ratio = math.sqrt(1.0 - lam_raw)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        if cut_h <= 0 or cut_w <= 0:
            return clip_a, clip_b, labels, labels, 1.0
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)
        if y2 == y1 or x2 == x1:
            return clip_a, clip_b, labels, labels, 1.0
        ca = clip_a.clone()
        cb = clip_b.clone()
        ca[:, :, :, y1:y2, x1:x2] = clip_a[perm][:, :, :, y1:y2, x1:x2]
        cb[:, :, :, y1:y2, x1:x2] = clip_b[perm][:, :, :, y1:y2, x1:x2]
        lam_adj = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
        return ca, cb, labels, labels[perm], lam_adj

    lam_raw = float(np.random.beta(mixup_alpha, mixup_alpha))
    perm = torch.randperm(B, device=clip_a.device)
    ca = lam_raw * clip_a + (1.0 - lam_raw) * clip_a[perm]
    cb = lam_raw * clip_b + (1.0 - lam_raw) * clip_b[perm]
    return ca, cb, labels, labels[perm], lam_raw


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip: float = 0.0,
    epoch_label: str = "",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    ema: Optional[ModelEMA] = None,
    aux_loss_weight: float = 0.0,
    predict_next_cls_weight: float = 0.0,
    predict_prev_cls_weight: float = 0.0,
    pair_direction_weight: float = 0.0,
    multi_clip_enabled: bool = False,
    consistency_weight: float = 0.0,
    teacher: Optional[nn.Module] = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 1.0,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch.

    With mix augmentation, accuracy is reported against the dominant label of each
    mixed pair (y_a). It's only a rough proxy for training fit; trust val/acc instead.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_aux = aux_loss_weight > 0.0 and getattr(model, "aux_head", None) is not None
    use_predict_next_cls = (
        predict_next_cls_weight > 0.0
        and getattr(model, "predict_next_cls_head", None) is not None
    )
    use_predict_prev_cls = (
        predict_prev_cls_weight > 0.0
        and getattr(model, "predict_prev_cls_head", None) is not None
    )
    use_cls_aux = use_predict_next_cls or use_predict_prev_cls
    use_pair_direction = (
        pair_direction_weight > 0.0
        and getattr(model, "pair_direction_head", None) is not None
    )
    use_distill = teacher is not None and distill_alpha > 0.0
    use_consistency = multi_clip_enabled and consistency_weight > 0.0

    def _forward_loss(video, y_a, y_b, lam, half_size: Optional[int] = None):
        if use_pair_direction:
            logits, frame_cls = model.forward_with_cls(video)   # (Btot, K), (Btot, T, D)
            main = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            pair_loss = model.pair_direction_head(frame_cls, y_a, y_b, lam, loss_fn)
            loss = main + pair_direction_weight * pair_loss
        elif use_aux:
            logits, aux_logits = model.forward_with_aux(video)  # (B,K), (B,T,K)
            main = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            B, T, K = aux_logits.shape
            flat = aux_logits.reshape(B * T, K)
            ya_rep = y_a.repeat_interleave(T)
            yb_rep = y_b.repeat_interleave(T)
            aux = lam * loss_fn(flat, ya_rep) + (1.0 - lam) * loss_fn(flat, yb_rep)
            loss = main + aux_loss_weight * aux
        elif use_cls_aux:
            logits, frame_cls = model.forward_with_cls(video)  # (B,K), (B,T,D)
            main = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            loss = main
            if use_predict_next_cls:
                loss = loss + predict_next_cls_weight * model.predict_next_cls_head(frame_cls)
            if use_predict_prev_cls:
                loss = loss + predict_prev_cls_weight * model.predict_prev_cls_head(frame_cls)
        else:
            logits = model(video)
            loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
        if use_consistency and half_size is not None:
            logits_a = logits[:half_size]
            logits_b = logits[half_size:]
            log_p_a = F.log_softmax(logits_a, dim=-1)
            log_p_b = F.log_softmax(logits_b, dim=-1)
            kl_ab = F.kl_div(log_p_a, log_p_b.detach().exp(), reduction="batchmean")
            kl_ba = F.kl_div(log_p_b, log_p_a.detach().exp(), reduction="batchmean")
            loss = loss + consistency_weight * 0.5 * (kl_ab + kl_ba)
        if use_distill:
            with torch.no_grad():
                teacher_logits = teacher(video)
            tk = distill_temperature
            student_logp = F.log_softmax(logits / tk, dim=-1)
            teacher_p = F.softmax(teacher_logits / tk, dim=-1)
            kl = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (tk * tk)
            loss = (1.0 - distill_alpha) * loss + distill_alpha * kl
        return logits, loss

    bar = tqdm(data_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    nan_streak = 0
    MAX_NAN_STREAK = 20
    for batch in bar:
        if multi_clip_enabled:
            # Dataset yields (clip_a, clip_b, label). Two augmented views per video.
            clip_a, clip_b, labels = batch
            clip_a = clip_a.to(device, non_blocking=True)
            clip_b = clip_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clip_a, clip_b, y_a_half, y_b_half, lam = _sample_shared_mix(
                clip_a, clip_b, labels, mixup_alpha, cutmix_alpha
            )
            half_size = labels.size(0)
            video_batch = torch.cat([clip_a, clip_b], dim=0)         # (2B, T, C, H, W)
            y_a = torch.cat([y_a_half, y_a_half], dim=0)             # (2B,)
            y_b = torch.cat([y_b_half, y_b_half], dim=0)
        else:
            # Standard path: dataset yields (video, label).
            video_batch, labels = batch
            video_batch = video_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            half_size = None
            # Alternate Mixup / CutMix per batch when both are enabled.
            if mixup_alpha > 0 and cutmix_alpha > 0:
                if random.random() < 0.5:
                    video_batch, (y_a, y_b, lam) = mixup_batch(video_batch, labels, mixup_alpha)
                else:
                    video_batch, (y_a, y_b, lam) = cutmix_batch(video_batch, labels, cutmix_alpha)
            elif mixup_alpha > 0:
                video_batch, (y_a, y_b, lam) = mixup_batch(video_batch, labels, mixup_alpha)
            elif cutmix_alpha > 0:
                video_batch, (y_a, y_b, lam) = cutmix_batch(video_batch, labels, cutmix_alpha)
            else:
                y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        stepped = True
        if amp and scaler is not None:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits, loss = _forward_loss(video_batch, y_a, y_b, lam, half_size=half_size)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {nan_streak} consecutive non-finite losses."
                    )
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Scaler lowers the scale when it skipped the step due to inf/NaN grads.
            stepped = scaler.get_scale() >= prev_scale
        elif amp:
            # bf16 path: no grad scaler needed.
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits, loss = _forward_loss(video_batch, y_a, y_b, lam, half_size=half_size)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {nan_streak} consecutive non-finite losses."
                    )
                continue
            nan_streak = 0
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            logits, loss = _forward_loss(video_batch, y_a, y_b, lam, half_size=half_size)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(
                        f"Training diverged: {nan_streak} consecutive non-finite losses."
                    )
                continue
            nan_streak = 0
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and stepped:
            scheduler.step()
        if ema is not None and stepped:
            ema.update(model)

        # In multi_clip mode, video_batch has 2*B rows but `labels` is still B —
        # use video_batch.size(0) as the effective sample count for averaging.
        nsamples = video_batch.size(0)
        running_loss += float(loss.item()) * nsamples
        predictions = logits.argmax(dim=1)
        correct += int((predictions == y_a).sum().item())
        total += nsamples
        bar.set_postfix(
            loss=running_loss / total,
            acc=correct / total,
            lr=optimizer.param_groups[0]["lr"],
        )

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def _cycle(iterable):
    """Yield from `iterable` forever (re-create iterator after exhaustion)."""
    while True:
        for x in iterable:
            yield x


def _ce_soft(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets: -sum(y_soft * log_softmax(x)) averaged
    over the batch. Equivalent to KL(soft_target || softmax(logits)) up to a
    constant (the soft-target entropy)."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_target * log_p).sum(dim=-1).mean()


def train_one_epoch_pseudo(
    model: nn.Module,
    labeled_loader: DataLoader,
    pseudo_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip: float = 0.0,
    epoch_label: str = "",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    ema: Optional[ModelEMA] = None,
    pair_direction_weight: float = 0.0,
    pseudo_weight: float = 1.0,
    teacher: Optional[nn.Module] = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 1.0,
) -> Tuple[float, float]:
    """Joint training step: labeled CE/KD/PDH + pseudo CE-soft on test data.

    Each step pulls one labeled batch (size L) and one pseudo batch (size P)
    in parallel; the pseudo loader is cycled because it has fewer samples.
    Two separate forwards keep the PDH/KD code paths unchanged on the
    labeled half and avoid running them on noisy pseudo samples.

    Combined loss is sample-weighted: total = (L * loss_l + P * loss_p) / (L+P),
    optionally scaled by `pseudo_weight` on the pseudo half.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_pair_direction = (
        pair_direction_weight > 0.0
        and getattr(model, "pair_direction_head", None) is not None
    )
    use_distill = teacher is not None and distill_alpha > 0.0

    pseudo_iter = iter(_cycle(pseudo_loader))

    def _forward_loss(video_l, y_a, y_b, lam, video_p, soft_p):
        if use_pair_direction:
            logits_l, frame_cls_l = model.forward_with_cls(video_l)
            main_l = lam * loss_fn(logits_l, y_a) + (1.0 - lam) * loss_fn(logits_l, y_b)
            pair_l = model.pair_direction_head(frame_cls_l, y_a, y_b, lam, loss_fn)
            loss_l = main_l + pair_direction_weight * pair_l
        else:
            logits_l = model(video_l)
            loss_l = lam * loss_fn(logits_l, y_a) + (1.0 - lam) * loss_fn(logits_l, y_b)
        if use_distill:
            with torch.no_grad():
                teacher_logits = teacher(video_l)
            tk = distill_temperature
            student_logp = F.log_softmax(logits_l / tk, dim=-1)
            teacher_p = F.softmax(teacher_logits / tk, dim=-1)
            kd = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (tk * tk)
            loss_l = (1.0 - distill_alpha) * loss_l + distill_alpha * kd

        logits_p = model(video_p)
        loss_p = _ce_soft(logits_p, soft_p)

        L = video_l.size(0)
        P = video_p.size(0)
        total_loss = (L * loss_l + pseudo_weight * P * loss_p) / float(L + P)
        return logits_l, logits_p, total_loss

    bar = tqdm(labeled_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    nan_streak = 0
    MAX_NAN_STREAK = 20
    for video_l, labels in bar:
        video_l = video_l.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_alpha > 0 and cutmix_alpha > 0:
            if random.random() < 0.5:
                video_l, (y_a, y_b, lam) = mixup_batch(video_l, labels, mixup_alpha)
            else:
                video_l, (y_a, y_b, lam) = cutmix_batch(video_l, labels, cutmix_alpha)
        elif mixup_alpha > 0:
            video_l, (y_a, y_b, lam) = mixup_batch(video_l, labels, mixup_alpha)
        elif cutmix_alpha > 0:
            video_l, (y_a, y_b, lam) = cutmix_batch(video_l, labels, cutmix_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        video_p, soft_p = next(pseudo_iter)
        video_p = video_p.to(device, non_blocking=True)
        soft_p = soft_p.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        stepped = True
        if amp and scaler is not None:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits_l, logits_p, loss = _forward_loss(video_l, y_a, y_b, lam, video_p, soft_p)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(f"Training diverged: {nan_streak} consecutive non-finite losses.")
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= prev_scale
        elif amp:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits_l, logits_p, loss = _forward_loss(video_l, y_a, y_b, lam, video_p, soft_p)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(f"Training diverged: {nan_streak} consecutive non-finite losses.")
                continue
            nan_streak = 0
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            logits_l, logits_p, loss = _forward_loss(video_l, y_a, y_b, lam, video_p, soft_p)
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= MAX_NAN_STREAK:
                    raise RuntimeError(f"Training diverged: {nan_streak} consecutive non-finite losses.")
                continue
            nan_streak = 0
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and stepped:
            scheduler.step()
        if ema is not None and stepped:
            ema.update(model)

        nsamples_l = video_l.size(0)
        running_loss += float(loss.item()) * nsamples_l
        # Train acc reported on labeled half only (pseudo has no hard label).
        predictions = logits_l.argmax(dim=1)
        correct += int((predictions == y_a).sum().item())
        total += nsamples_l
        bar.set_postfix(
            loss=running_loss / total,
            acc=correct / total,
            lr=optimizer.param_groups[0]["lr"],
        )

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    epoch_label: str = "",
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(data_loader, desc=f"{epoch_label} val", leave=False, dynamic_ncols=True)
    for video_batch, labels in bar:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if amp:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(video_batch)
                loss = loss_fn(logits, labels)
        else:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)
        bar.set_postfix(loss=running_loss / total, acc=correct / total)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()

    # Keep only class folders whose names match the validation split — train/ contains
    # 23 extra folders under an older numeric scheme that duplicate canonical clips
    # under wrong labels; leaving them in poisons the CE signal.
    canonical_classes = {p.name for p in val_dir.iterdir() if p.is_dir()}

    train_samples = collect_video_samples(train_dir)
    train_samples = [s for s in train_samples if s[0].parent.name in canonical_classes]
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        train_samples = train_samples[: int(max_samples)]
        val_samples = val_samples[: int(max_samples)]
    print(
        f"[train] {len(train_samples)} train videos from {train_dir.name}/, "
        f"{len(val_samples)} val videos from {val_dir.name}/ (real held-out set)",
        flush=True,
    )

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    if cfg.model.name == "modular":
        use_imagenet_norm = bool(cfg.model.spatial.get("pretrained", False))
    else:
        use_imagenet_norm = bool(cfg.model.get("pretrained", False))
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    # Strong clip-aware augmentation (RandomResizedCrop + ColorJitter + RandAugment +
    # RandomErasing, all consistent across frames). Used when training a temporal model
    # like TSM where per-frame independent crops would scramble the motion signal.
    use_strong_aug = bool(cfg.training.get("strong_clip_aug", False))
    random_temporal_offset = bool(cfg.dataset.get("random_temporal_offset", False))

    # Optional multi-clip training (exp3a): for each video, sample two
    # independently augmented clips per item. The training loop applies the
    # SAME (mixup/cutmix lam, perm) to both halves and adds a symmetric KL
    # consistency loss between their predictions. Requires random_offset_frac
    # so the two clips don't collapse to identical inputs.
    mc_cfg = cfg.training.get("multi_clip", None)
    multi_clip_enabled = mc_cfg is not None and bool(mc_cfg.get("enabled", False))
    if multi_clip_enabled and not random_temporal_offset:
        raise ValueError(
            "training.multi_clip.enabled=true requires dataset.random_temporal_offset=true"
        )
    train_dataset_cls = MultiClipVideoDataset if multi_clip_enabled else VideoFrameDataset
    if use_strong_aug:
        train_clip_transform = build_strong_clip_transform(
            image_size=224, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = train_dataset_cls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
            random_offset_frac=random_temporal_offset,
        )
    else:
        train_dataset = train_dataset_cls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            transform=train_transform,
            sample_list=train_samples,
            random_offset_frac=random_temporal_offset,
        )
    if random_temporal_offset:
        print("[train] random_temporal_offset=True — train clips use Uniform[0,1) offset_frac per item.", flush=True)
    if multi_clip_enabled:
        print(
            f"[train] multi_clip enabled (2 clips/video, consistency KL weight "
            f"{float(mc_cfg.get('consistency_weight', 0.2))})",
            flush=True,
        )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    # Pseudo-label (noisy-student) parallel loader. When enabled, each training
    # step consumes `pseudo_per_batch` test videos with ensemble soft targets
    # alongside `batch_size - pseudo_per_batch` labeled train videos — the
    # total per-step forward stays at `batch_size`.
    pseudo_cfg = cfg.training.get("pseudo_labels", None)
    pseudo_enabled = pseudo_cfg is not None and bool(pseudo_cfg.get("enabled", False))
    pseudo_per_batch = int(pseudo_cfg.get("pseudo_per_batch", 4)) if pseudo_enabled else 0
    pseudo_weight = float(pseudo_cfg.get("weight", 1.0)) if pseudo_enabled else 0.0
    pseudo_loader: Optional[DataLoader] = None
    if pseudo_enabled:
        if multi_clip_enabled:
            raise ValueError("training.pseudo_labels is not compatible with training.multi_clip")
        labeled_bs = int(cfg.training.batch_size) - pseudo_per_batch
        if labeled_bs <= 0:
            raise ValueError(
                f"pseudo_per_batch={pseudo_per_batch} must be < batch_size={cfg.training.batch_size}"
            )
        pseudo_file = Path(str(pseudo_cfg.path)).resolve()
        if not pseudo_file.is_file():
            raise FileNotFoundError(f"pseudo_labels.path not found: {pseudo_file}")
        if use_strong_aug:
            pseudo_dataset = PseudoLabelVideoDataset(
                test_dir=Path(cfg.dataset.test_dir).resolve(),
                pseudo_file=pseudo_file,
                num_frames=int(cfg.dataset.num_frames),
                clip_transform=train_clip_transform,
                random_offset_frac=True,
            )
        else:
            pseudo_dataset = PseudoLabelVideoDataset(
                test_dir=Path(cfg.dataset.test_dir).resolve(),
                pseudo_file=pseudo_file,
                num_frames=int(cfg.dataset.num_frames),
                transform=train_transform,
                random_offset_frac=True,
            )
        print(
            f"[train] pseudo-label noisy-student enabled: {len(pseudo_dataset)} pseudo videos, "
            f"labeled_bs={labeled_bs} + pseudo_bs={pseudo_per_batch} per step, "
            f"weight={pseudo_weight}, file={pseudo_file.name}",
            flush=True,
        )
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=pseudo_per_batch,
            shuffle=True,
            num_workers=max(2, int(cfg.training.num_workers) // 2),
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    else:
        labeled_bs = int(cfg.training.batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=labeled_bs,
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optional per-frame deep-supervision head. Attached on the live model
    # before EMA so the EMA copy mirrors it (eval still uses main classifier).
    aux_cfg = cfg.training.get("aux_loss", None)
    aux_loss_weight = 0.0
    if aux_cfg is not None and bool(aux_cfg.get("enabled", False)):
        if cfg.model.name != "modular":
            raise ValueError("training.aux_loss only supported for model.name=modular")
        attach_aux_head(model, num_classes=int(cfg.model.num_classes))
        model.aux_head = model.aux_head.to(device)
        aux_loss_weight = float(aux_cfg.get("weight", 0.3))
        print(f"[train] aux per-frame loss enabled (weight={aux_loss_weight})")

    # Optional auxiliary "predict next CLS" head: from frames 1..T-1 spatial CLS,
    # predict frame-T CLS via a small MLP, cosine loss vs stopgrad target. The
    # head is attached as an attribute so LLRD/EMA pick it up automatically.
    pnc_cfg = cfg.training.get("predict_next_cls", None)
    predict_next_cls_weight = 0.0
    if pnc_cfg is not None and bool(pnc_cfg.get("enabled", False)):
        if cfg.model.name != "modular":
            raise ValueError("training.predict_next_cls only supported for model.name=modular")
        embed_dim = int(model.spatial.out_dim)
        num_input_frames = int(cfg.dataset.num_frames) - 1
        hidden_dim = int(pnc_cfg.get("hidden_dim", 512))
        model.predict_next_cls_head = PredictNextCLSAuxHead(
            embed_dim=embed_dim,
            num_input_frames=num_input_frames,
            hidden_dim=hidden_dim,
        ).to(device)
        predict_next_cls_weight = float(pnc_cfg.get("weight", 0.1))
        print(
            f"[train] predict-next-CLS aux enabled (weight={predict_next_cls_weight}, "
            f"hidden={hidden_dim}, T_in={num_input_frames})"
        )

    # Optional auxiliary "predict prev CLS" head: mirror of predict-next with
    # input=CLS at frames 1..T-1, target=stopgrad(CLS at frame 0). Same zero-init
    # fc2, same fp32 cosine loss. Bidirectional aux signal.
    ppc_cfg = cfg.training.get("predict_prev_cls", None)
    predict_prev_cls_weight = 0.0
    if ppc_cfg is not None and bool(ppc_cfg.get("enabled", False)):
        if cfg.model.name != "modular":
            raise ValueError("training.predict_prev_cls only supported for model.name=modular")
        embed_dim = int(model.spatial.out_dim)
        num_input_frames = int(cfg.dataset.num_frames) - 1
        hidden_dim = int(ppc_cfg.get("hidden_dim", 512))
        model.predict_prev_cls_head = PredictPrevCLSAuxHead(
            embed_dim=embed_dim,
            num_input_frames=num_input_frames,
            hidden_dim=hidden_dim,
        ).to(device)
        predict_prev_cls_weight = float(ppc_cfg.get("weight", 0.1))
        print(
            f"[train] predict-prev-CLS aux enabled (weight={predict_prev_cls_weight}, "
            f"hidden={hidden_dim}, T_in={num_input_frames})"
        )

    # Optional pairwise-direction head (exp3a): for the 6 ordered (i<j) pairs
    # of frame CLS tokens, classify the action from
    # concat[cls_i, cls_j, cls_j-cls_i, cls_i+cls_j] via a 2-layer MLP. Averaged
    # logits supervised by mixup-aware CE with weight λ_pair. Zero-init fc_out.
    pdh_cfg = cfg.training.get("pair_direction", None)
    pair_direction_weight = 0.0
    if pdh_cfg is not None and bool(pdh_cfg.get("enabled", False)):
        if cfg.model.name != "modular":
            raise ValueError("training.pair_direction only supported for model.name=modular")
        embed_dim = int(model.spatial.out_dim)
        hidden_dim = int(pdh_cfg.get("hidden_dim", 768))
        model.pair_direction_head = PairDirectionAuxHead(
            embed_dim=embed_dim,
            num_frames=int(cfg.dataset.num_frames),
            num_classes=int(cfg.model.num_classes),
            hidden_dim=hidden_dim,
        ).to(device)
        pair_direction_weight = float(pdh_cfg.get("weight", 0.3))
        print(
            f"[train] pair-direction aux enabled (weight={pair_direction_weight}, "
            f"hidden={hidden_dim})"
        )

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    amp_dtype_str = str(cfg.training.get("amp_dtype", "fp16")).lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    # Optimizer: AdamW (default, with optional LLRD or backbone/head split), or SGD-momentum.
    optimizer_name = str(cfg.training.get("optimizer", "adamw")).lower()
    llrd = float(cfg.training.get("llrd", 0.0))
    backbone_lr = cfg.training.get("backbone_lr", None)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr),
            momentum=float(cfg.training.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.training.get("nesterov", False)),
        )
        print(f"[train] SGD optimizer (lr={cfg.training.lr}, momentum={cfg.training.get('momentum', 0.9)})")
    elif llrd > 0.0:
        groups = build_llrd_param_groups(
            model,
            base_lr=float(cfg.training.lr),
            decay_rate=llrd,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(groups)
        print(f"[train] AdamW + LLRD (decay={llrd}): {len(groups)} param groups")
    elif backbone_lr is not None:
        groups = build_two_group_param_groups(
            model,
            head_lr=float(cfg.training.lr),
            backbone_lr=float(backbone_lr),
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(groups)
        print(
            f"[train] AdamW + 2-group LR: backbone={backbone_lr}, head={cfg.training.lr}, "
            f"{len(groups)} param groups"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=weight_decay,
        )

    total_steps = int(cfg.training.epochs) * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    lr_schedule = str(cfg.training.get("lr_schedule", "cosine")).lower()
    if lr_schedule not in ("cosine", "constant"):
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r} (expected 'cosine' or 'constant')")

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if lr_schedule == "constant":
            return 1.0
        remaining = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / remaining
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    # GradScaler is only useful for fp16 — bf16 has the same exponent range as fp32.
    scaler = (
        GradScaler(device=device.type)
        if (amp_enabled and amp_dtype == torch.float16)
        else None
    )

    # Optional EMA of weights — typically +0.5–2 pt on video classifiers.
    ema: Optional[ModelEMA] = None
    if bool(cfg.training.get("ema", False)):
        ema_decay = float(cfg.training.get("ema_decay", 0.9999))
        ema = ModelEMA(model, decay=ema_decay)
        print(f"[train] EMA enabled (decay={ema_decay})")

    # Optional SWA: average end-of-epoch (EMA, if present, else live) weights
    # across a plateau window. Saved at end-of-training as a separate ckpt.
    # Not included in _last.pt resume state — on a crash mid-window, SWA
    # restarts from scratch when training resumes.
    swa_cfg = cfg.training.get("swa", None)
    swa_enabled = swa_cfg is not None and bool(swa_cfg.get("enabled", False))
    swa_start_epoch_1idx = int(swa_cfg.get("start_epoch", 0)) if swa_enabled else 0
    swa_model: Optional[torch.optim.swa_utils.AveragedModel] = None

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last" + checkpoint_path.suffix)
    swa_path = checkpoint_path.with_name(checkpoint_path.stem + "_swa" + checkpoint_path.suffix)
    if swa_enabled:
        print(
            f"[train] SWA enabled (start at epoch {swa_start_epoch_1idx}, "
            f"source={'EMA' if ema is not None else 'live'}, save to {swa_path})"
        )

    init_wandb(cfg, default_run_name=checkpoint_path.stem)

    # --- resume from last.pt if present and config matches ---
    cfg_hash = make_resume_hash(cfg)
    start_epoch = 0
    resume_enabled = bool(cfg.training.get("resume", True))
    if resume_enabled and last_path.exists():
        print(f"[train] found resume file: {last_path}")
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
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"])
        best_val_accuracy = float(state.get("best_val_accuracy", 0.0))
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[train] resuming from epoch {start_epoch + 1}/{cfg.training.epochs} (best so far {best_val_accuracy:.4f})")
    elif not resume_enabled and last_path.exists():
        print(f"[train] training.resume=false → ignoring {last_path}")

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.training.get("cutmix_alpha", 0.0))
    consistency_weight = (
        float(mc_cfg.get("consistency_weight", 0.0)) if multi_clip_enabled else 0.0
    )

    # Optional self/cross-distillation: KL between student and a frozen teacher
    # on the same (already-mixed) input. alpha splits the loss; T softens both
    # distributions. Teacher is rebuilt from the checkpoint's embedded Hydra cfg.
    teacher: Optional[nn.Module] = None
    distill_alpha = 0.0
    distill_temperature = 1.0
    distill_cfg = cfg.training.get("distill", None)
    if distill_cfg is not None and distill_cfg.get("teacher_ckpt", None):
        raw_ckpt = distill_cfg.teacher_ckpt
        # Accept either a single path or a list of paths (ensemble teacher).
        if isinstance(raw_ckpt, (list, tuple)) or OmegaConf.is_list(raw_ckpt):
            teacher_paths = [Path(str(p)).resolve() for p in raw_ckpt]
        else:
            teacher_paths = [Path(str(raw_ckpt)).resolve()]
        for tp in teacher_paths:
            if not tp.is_file():
                raise FileNotFoundError(f"distill.teacher_ckpt not found: {tp}")
        loaded = []
        for tp in teacher_paths:
            print(f"[train] loading teacher from {tp}")
            loaded.append(load_teacher(tp, device))
        if len(loaded) == 1:
            teacher = loaded[0]
        else:
            combine = str(distill_cfg.get("combine", "softmax_mean"))
            teacher = EnsembleTeacher(loaded, combine=combine).to(device)
            print(f"[train] ensemble teacher: {len(loaded)} models, combine={combine}")
        distill_alpha = float(distill_cfg.get("alpha", 0.5))
        distill_temperature = float(distill_cfg.get("temperature", 4.0))
        print(
            f"[train] distillation enabled (alpha={distill_alpha}, T={distill_temperature})"
        )

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        label = f"[{epoch + 1}/{cfg.training.epochs}]"
        if pseudo_enabled:
            assert pseudo_loader is not None
            train_loss, train_acc = train_one_epoch_pseudo(
                model,
                train_loader,
                pseudo_loader,
                loss_fn,
                optimizer,
                device,
                scheduler=scheduler,
                scaler=scaler,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip=grad_clip,
                epoch_label=label,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                ema=ema,
                pair_direction_weight=pair_direction_weight,
                pseudo_weight=pseudo_weight,
                teacher=teacher,
                distill_alpha=distill_alpha,
                distill_temperature=distill_temperature,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                scheduler=scheduler,
                scaler=scaler,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip=grad_clip,
                epoch_label=label,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                ema=ema,
                aux_loss_weight=aux_loss_weight,
                predict_next_cls_weight=predict_next_cls_weight,
                predict_prev_cls_weight=predict_prev_cls_weight,
                pair_direction_weight=pair_direction_weight,
                multi_clip_enabled=multi_clip_enabled,
                consistency_weight=consistency_weight,
                teacher=teacher,
                distill_alpha=distill_alpha,
                distill_temperature=distill_temperature,
            )
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc = evaluate_epoch(
            eval_model,
            val_loader,
            loss_fn,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            epoch_label=label,
        )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        wandb_log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            },
            step=epoch + 1,
        )

        # Guard: refuse to overwrite _last.pt with a poisoned state. exp2k
        # ran 50 epochs of NaN before being killed because nothing in the
        # loop noticed. Exiting non-zero lets train_robust.sh re-launch and
        # resume from the previous (healthy) _last.pt.
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            print(
                f"[train] non-finite loss at epoch {epoch + 1} "
                f"(train_loss={train_loss}, val_loss={val_loss}). "
                f"Not overwriting {last_path}. Exiting 2 for watchdog respawn.",
                flush=True,
            )
            try:
                wandb_log({"train/diverged_epoch": epoch + 1})
                finish_wandb()
            except Exception:
                pass
            raise SystemExit(2)

        # SWA update: snapshot current (EMA or live) weights into the running
        # average once per epoch from swa_start onward. Done after the
        # divergence guard above so we never average a NaN epoch.
        if swa_enabled and (epoch + 1) >= swa_start_epoch_1idx:
            swa_source = ema.module if ema is not None else model
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(swa_source)
                print(f"[train] SWA: started averaging at epoch {epoch + 1}", flush=True)
            swa_model.update_parameters(swa_source)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # When EMA is on, save the EMA weights (those are what just got val'd).
            best_state_dict = (ema.module if ema is not None else model).state_dict()
            payload: Dict[str, Any] = {
                "model_state_dict": best_state_dict,
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": use_imagenet_norm,
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            atomic_torch_save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

        # Per-epoch resumable snapshot (full state, atomic, overwritten).
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "ema": ema.module.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "cfg_hash": cfg_hash,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "rng": capture_rng_state(),
                "best_val_accuracy": best_val_accuracy,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            last_path,
        )

    if swa_enabled and swa_model is not None:
        n_avg = int(swa_model.n_averaged.item())
        swa_payload: Dict[str, Any] = {
            "model_state_dict": swa_model.module.state_dict(),
            "model_name": cfg.model.name,
            "num_classes": int(cfg.model.num_classes),
            "pretrained": use_imagenet_norm,
            "num_frames": int(cfg.dataset.num_frames),
            "val_accuracy": -1.0,  # SWA wasn't val'd inline; run evaluate.py offline
            "config": OmegaConf.to_container(cfg, resolve=True),
            "swa_start_epoch": swa_start_epoch_1idx,
            "swa_n_averaged": n_avg,
        }
        if cfg.model.name == "cnn_lstm":
            swa_payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))
        atomic_torch_save(swa_payload, swa_path)
        print(f"[train] saved SWA checkpoint ({n_avg} averages) to {swa_path}")
        wandb_log({"swa/n_averaged": n_avg, "swa/start_epoch": swa_start_epoch_1idx})

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb_log({"val/best_acc": best_val_accuracy})
    finish_wandb()


if __name__ == "__main__":
    main()
