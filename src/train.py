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
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.multi_clip_dataset import MultiClipVideoDataset
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.aux.pair_direction import PairDirectionAuxHead
from models.cmt import CMT
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.modular import build_modular_model
from torch.optim.swa_utils import AveragedModel
from utils import (
    ModelEMA,
    apply_mix_shared,
    atomic_torch_save,
    build_llrd_param_groups,
    build_soft_clip_transform,
    build_strong_clip_transform,
    build_transforms,
    build_two_group_param_groups,
    build_videomae_clip_transform,
    capture_rng_state,
    cutmix_batch,
    finish_wandb,
    init_wandb,
    make_resume_hash,
    mixup_batch,
    restore_rng_state,
    sample_mix_params,
    set_backbone_frozen,
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
    teacher: Optional[ModelEMA] = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 1.0,
    pair_direction_head: Optional[PairDirectionAuxHead] = None,
    pair_direction_weight: float = 0.0,
    multi_clip_consistency_weight: float = 0.0,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch.

    With mix augmentation, accuracy is reported against the dominant label of each
    mixed pair (y_a). It's only a rough proxy for training fit; trust val/acc instead.

    When ``teacher`` is provided and ``distill_alpha > 0``, the EMA teacher's softmax
    is used as a soft target alongside the hard CE loss:
        loss = (1 - α) * CE_mix + α * T² * KL(softmax(student/T) || softmax(teacher/T))
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(data_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    multi_clip_active = multi_clip_consistency_weight > 0.0
    aux_active = pair_direction_head is not None and pair_direction_weight > 0.0
    for batch in bar:
        # batch is (video, label) or (clip_a, clip_b, label) when multi-clip is on.
        if multi_clip_active and len(batch) == 3:
            clip_a, clip_b, labels = batch
            clip_a = clip_a.to(device, non_blocking=True)
            clip_b = clip_b.to(device, non_blocking=True)
        else:
            clip_a, labels = batch
            clip_a = clip_a.to(device, non_blocking=True)
            clip_b = None
        labels = labels.to(device, non_blocking=True)

        # Sample mix params ONCE so both halves are mixed identically.
        h, w = clip_a.shape[-2], clip_a.shape[-1]
        if clip_b is not None:
            mix = sample_mix_params(clip_a.size(0), mixup_alpha, cutmix_alpha, h, w, device)
            clip_a = apply_mix_shared(clip_a, mix)
            clip_b = apply_mix_shared(clip_b, mix)
            if mix["kind"] == "none":
                y_a, y_b, lam = labels, labels, 1.0
            else:
                y_a, y_b, lam = labels, labels[mix["perm"]], float(mix["lam"])
            video_batch = clip_a
        else:
            # Single-clip path (legacy).
            if mixup_alpha > 0 and cutmix_alpha > 0:
                if random.random() < 0.5:
                    clip_a, (y_a, y_b, lam) = mixup_batch(clip_a, labels, mixup_alpha)
                else:
                    clip_a, (y_a, y_b, lam) = cutmix_batch(clip_a, labels, cutmix_alpha)
            elif mixup_alpha > 0:
                clip_a, (y_a, y_b, lam) = mixup_batch(clip_a, labels, mixup_alpha)
            elif cutmix_alpha > 0:
                clip_a, (y_a, y_b, lam) = cutmix_batch(clip_a, labels, cutmix_alpha)
            else:
                y_a, y_b, lam = labels, labels, 1.0
            video_batch = clip_a

        optimizer.zero_grad(set_to_none=True)

        kd_active = teacher is not None and distill_alpha > 0.0
        T = float(distill_temperature)

        def _forward(video):
            """Single forward.

            Returns ``(logits, frame_cls or None)``. When the aux head is
            active and the spatial encoder returns 4-D ``(B, T', N, D)``
            features, we unpack the model so the per-tubelet mean is exposed
            to the aux head. Otherwise just call ``model(video)``.
            """
            if not aux_active:
                return model(video), None
            features = model.spatial(video)
            if features.dim() == 4:
                frame_cls = features.mean(dim=2)         # (B, T', D)
            else:
                frame_cls = features                     # already (B, T', D)
            temporal_out = model.temporal(features)
            logits_ = model.classifier(temporal_out)
            return logits_, frame_cls

        def _compose_loss(logits_: torch.Tensor, frame_cls_a, logits_b=None, frame_cls_b=None):
            ce_a = lam * loss_fn(logits_, y_a) + (1.0 - lam) * loss_fn(logits_, y_b)
            total = ce_a
            if logits_b is not None:
                # Secondary clip CE (same labels under shared mix).
                ce_b = lam * loss_fn(logits_b, y_a) + (1.0 - lam) * loss_fn(logits_b, y_b)
                total = 0.5 * (ce_a + ce_b)
                # Symmetric KL consistency. Detach the target side so each KL
                # term pulls one set of logits toward a soft moving target.
                p_a = F.log_softmax(logits_, dim=-1)
                p_b = F.log_softmax(logits_b, dim=-1)
                with torch.no_grad():
                    q_a = F.softmax(logits_.detach().float(), dim=-1)
                    q_b = F.softmax(logits_b.detach().float(), dim=-1)
                kl_ab = F.kl_div(p_a, q_b, reduction="batchmean")
                kl_ba = F.kl_div(p_b, q_a, reduction="batchmean")
                total = total + multi_clip_consistency_weight * 0.5 * (kl_ab + kl_ba)
            if aux_active and frame_cls_a is not None:
                aux_loss = pair_direction_head(frame_cls_a, y_a, y_b, lam, loss_fn)
                if frame_cls_b is not None:
                    aux_loss_b = pair_direction_head(frame_cls_b, y_a, y_b, lam, loss_fn)
                    aux_loss = 0.5 * (aux_loss + aux_loss_b)
                total = total + pair_direction_weight * aux_loss
            if kd_active:
                with torch.no_grad():
                    teacher_logits = teacher.module(video_batch)
                kd = F.kl_div(
                    F.log_softmax(logits_ / T, dim=-1),
                    F.softmax(teacher_logits.float() / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                total = (1.0 - distill_alpha) * total + distill_alpha * kd
            return total

        def _full_forward():
            logits_a, fcls_a = _forward(clip_a)
            if clip_b is not None:
                logits_b, fcls_b = _forward(clip_b)
            else:
                logits_b, fcls_b = None, None
            return logits_a, fcls_a, logits_b, fcls_b

        stepped = True
        if amp and scaler is not None:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits, fcls_a, logits_b, fcls_b = _full_forward()
                loss = _compose_loss(logits, fcls_a, logits_b, fcls_b)
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
                logits, fcls_a, logits_b, fcls_b = _full_forward()
                loss = _compose_loss(logits, fcls_a, logits_b, fcls_b)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            logits, fcls_a, logits_b, fcls_b = _full_forward()
            loss = _compose_loss(logits, fcls_a, logits_b, fcls_b)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and stepped:
            scheduler.step()

        if ema is not None and stepped:
            ema.update(model)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == y_a).sum().item())
        total += labels.size(0)
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
    all_samples = collect_video_samples(train_dir)

    # Keep only class folders whose names match the validation split — train/ contains
    # 23 extra folders under an older numeric scheme that duplicate canonical clips
    # under wrong labels; leaving them in poisons the CE signal.
    val_dir = Path(cfg.dataset.val_dir).resolve()
    canonical_classes = {p.name for p in val_dir.iterdir() if p.is_dir()}
    all_samples = [s for s in all_samples if s[0].parent.name in canonical_classes]

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
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

    # Strong / soft clip-aware augmentation (RandomResizedCrop + ColorJitter +
    # RandAugment + RandomErasing, all consistent across frames). The "soft" preset
    # uses milder ranges — useful for ViT FT on small datasets where the strong
    # preset can overwhelm the gradient signal.
    aug_strength = str(cfg.training.get("aug_strength", "")).lower()
    use_strong_aug = bool(cfg.training.get("strong_clip_aug", False)) or aug_strength == "strong"
    use_soft_aug = aug_strength == "soft"
    use_videomae_aug = aug_strength == "videomae"
    # Multi-clip wraps VideoFrameDataset to yield (clip_a, clip_b, label) so the
    # train loop can compute a sym-KL consistency loss between two augmented views.
    multi_clip_cfg = cfg.training.get("multi_clip", None)
    multi_clip_enabled = bool(multi_clip_cfg is not None and multi_clip_cfg.get("enabled", False))
    TrainDatasetCls = MultiClipVideoDataset if multi_clip_enabled else VideoFrameDataset
    if multi_clip_enabled:
        print(f"[train] multi-clip enabled (num_clips=2, consistency_weight={float(multi_clip_cfg.get('consistency_weight', 0.2))})")
    if use_videomae_aug:
        train_clip_transform = build_videomae_clip_transform(
            image_size=224, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = TrainDatasetCls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
        )
    elif use_soft_aug:
        train_clip_transform = build_soft_clip_transform(
            image_size=224, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = TrainDatasetCls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
        )
    elif use_strong_aug:
        train_clip_transform = build_strong_clip_transform(
            image_size=224, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = TrainDatasetCls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
        )
    else:
        train_dataset = TrainDatasetCls(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            transform=train_transform,
            sample_list=train_samples,
        )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
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

    # Optional auxiliary head: pair-direction CE over per-tubelet summary tokens.
    # When enabled, the head is attached as a submodule of the model so it is
    # saved/loaded via the model state_dict and picked up by the optimizer
    # param-group walker (LLRD puts it in the head groups by exclusion).
    pair_cfg = cfg.training.get("pair_direction", None)
    pair_direction_head: Optional[PairDirectionAuxHead] = None
    pair_direction_weight = 0.0
    if pair_cfg is not None and bool(pair_cfg.get("enabled", False)):
        if not hasattr(model, "spatial"):
            raise ValueError("pair_direction requires model.name=modular (model.spatial.out_dim)")
        pair_num_frames = int(pair_cfg.get(
            "num_frames",
            getattr(getattr(model.spatial, "encoder", None), "t_grid", cfg.dataset.num_frames),
        ))
        pair_direction_head = PairDirectionAuxHead(
            embed_dim=int(model.spatial.out_dim),
            num_frames=pair_num_frames,
            num_classes=int(cfg.model.num_classes),
            hidden_dim=int(pair_cfg.get("hidden_dim", 768)),
        ).to(device)
        pair_direction_weight = float(pair_cfg.get("weight", 0.3))
        model.pair_direction_head = pair_direction_head
        print(
            f"[train] PairDirectionAux: T={pair_num_frames}, "
            f"weight={pair_direction_weight}, hidden={pair_direction_head.fc1.out_features}"
        )

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
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

    # Optional SWA — averages weights over the plateau window for a final
    # consolidation step. AveragedModel handles the averaging math; we save the
    # averaged weights separately at end-of-training as `<run>_swa.pt`.
    swa_cfg = cfg.training.get("swa", None)
    swa: Optional[AveragedModel] = None
    swa_start_epoch = -1
    if swa_cfg is not None and bool(swa_cfg.get("enabled", False)):
        swa = AveragedModel(model)
        swa_start_epoch = int(swa_cfg.get("start_epoch", max(1, int(cfg.training.epochs) // 2)))
        print(f"[train] SWA enabled (start_epoch={swa_start_epoch})")

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last" + checkpoint_path.suffix)

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
        if swa is not None and state.get("swa") is not None:
            try:
                swa.module.load_state_dict(state["swa"]["module"])
                swa.n_averaged.fill_(int(state["swa"]["n_averaged"]))
                print(f"[train] resumed SWA (n_averaged={int(swa.n_averaged.item())})")
            except Exception as e:
                print(f"[train] WARNING: SWA resume failed ({e}); starting fresh SWA")
        start_epoch = int(state["epoch"])
        best_val_accuracy = float(state.get("best_val_accuracy", 0.0))
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[train] resuming from epoch {start_epoch + 1}/{cfg.training.epochs} (best so far {best_val_accuracy:.4f})")
    elif not resume_enabled and last_path.exists():
        print(f"[train] training.resume=false → ignoring {last_path}")

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.training.get("cutmix_alpha", 0.0))
    freeze_backbone_epochs = int(cfg.training.get("freeze_backbone_epochs", 0))
    distill_alpha = float(cfg.training.get("distill_alpha", 0.0))
    distill_temperature = float(cfg.training.get("distill_temperature", 1.0))
    multi_clip_consistency_weight = float(
        (multi_clip_cfg or {}).get("consistency_weight", 0.0)
    ) if multi_clip_enabled else 0.0
    # When EMA is used as a distillation teacher, evaluating on it would just
    # measure the teacher — eval the live student instead. The flag still lets
    # legacy configs keep eval-on-EMA when no distillation is active.
    default_eval_on_ema = ema is not None and distill_alpha == 0.0
    eval_on_ema = bool(cfg.training.get("eval_on_ema", default_eval_on_ema))
    if distill_alpha > 0.0 and ema is None:
        raise ValueError("distill_alpha > 0 requires training.ema=true (the EMA acts as the teacher).")
    if distill_alpha > 0.0:
        print(f"[train] EMA-teacher distillation enabled (alpha={distill_alpha}, T={distill_temperature})")
    if freeze_backbone_epochs > 0:
        print(f"[train] LP-FT: spatial trunk frozen for first {freeze_backbone_epochs} epochs")

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        # LP-FT toggle (idempotent; correct after resume too).
        set_backbone_frozen(model, frozen=epoch < freeze_backbone_epochs)
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            print(f"[train] unfreezing spatial trunk at epoch {epoch + 1}/{cfg.training.epochs}")
        label = f"[{epoch + 1}/{cfg.training.epochs}]"
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
            teacher=ema if distill_alpha > 0.0 else None,
            distill_alpha=distill_alpha,
            distill_temperature=distill_temperature,
            pair_direction_head=pair_direction_head,
            pair_direction_weight=pair_direction_weight,
            multi_clip_consistency_weight=multi_clip_consistency_weight,
        )

        # SWA update: add this epoch's weights to the running average.
        if swa is not None and (epoch + 1) >= swa_start_epoch:
            swa.update_parameters(model)
        eval_model = ema.module if (ema is not None and eval_on_ema) else model
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

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Save whichever model was actually evaluated (live student when distill is
            # on, EMA when EMA is the eval target, plain model otherwise).
            best_state_dict = eval_model.state_dict()
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
        snapshot = {
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
        }
        if swa is not None:
            snapshot["swa"] = {
                "module": swa.module.state_dict(),
                "n_averaged": int(swa.n_averaged.item()),
            }
        atomic_torch_save(snapshot, last_path)

    # End-of-training: write the SWA-averaged weights to a separate
    # checkpoint so submission scripts can load them in place of the live
    # student. We save it under a "*_swa.pt" sibling of the best checkpoint.
    if swa is not None and int(swa.n_averaged.item()) > 0:
        swa_path = checkpoint_path.with_name(checkpoint_path.stem + "_swa" + checkpoint_path.suffix)
        swa_payload: Dict[str, Any] = {
            "model_state_dict": swa.module.state_dict(),
            "model_name": cfg.model.name,
            "num_classes": int(cfg.model.num_classes),
            "pretrained": use_imagenet_norm,
            "num_frames": int(cfg.dataset.num_frames),
            "n_averaged": int(swa.n_averaged.item()),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        atomic_torch_save(swa_payload, swa_path)
        print(f"  Saved SWA model to {swa_path} (n_averaged={int(swa.n_averaged.item())})")

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb_log({"val/best_acc": best_val_accuracy})
    finish_wandb()


if __name__ == "__main__":
    main()
