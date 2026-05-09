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
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cmt import CMT
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.modular import attach_aux_head, build_modular_model
from models.two_stream import build_two_stream_model
from utils import (
    ModelEMA,
    atomic_torch_save,
    build_flow_clip_transforms,
    build_flow_only_clip_transforms,
    build_llrd_param_groups,
    build_strong_clip_transform,
    build_transforms,
    build_two_group_param_groups,
    capture_rng_state,
    cutmix_batch,
    cutmix_batch_two_stream,
    finish_wandb,
    init_wandb,
    make_resume_hash,
    mixup_batch,
    mixup_batch_two_stream,
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
    if name == "two_stream":
        return build_two_stream_model(cfg.model, num_classes=num_classes)

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
    aux_loss_weight: float = 0.0,
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

    def _forward_loss(video, y_a, y_b, lam):
        if use_aux:
            logits, aux_logits = model.forward_with_aux(video)  # (B,K), (B,T,K)
            main = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            B, T, K = aux_logits.shape
            flat = aux_logits.reshape(B * T, K)
            ya_rep = y_a.repeat_interleave(T)
            yb_rep = y_b.repeat_interleave(T)
            aux = lam * loss_fn(flat, ya_rep) + (1.0 - lam) * loss_fn(flat, yb_rep)
            return logits, main + aux_loss_weight * aux
        logits = model(video)
        loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
        return logits, loss

    bar = tqdm(data_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    for video_batch, labels in bar:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
                logits, loss = _forward_loss(video_batch, y_a, y_b, lam)
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
                logits, loss = _forward_loss(video_batch, y_a, y_b, lam)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            logits, loss = _forward_loss(video_batch, y_a, y_b, lam)
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


def train_one_epoch_two_stream(
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
) -> Tuple[float, float]:
    """Two-stream variant: data loader yields ((rgb, flow), labels)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_aux = aux_loss_weight > 0.0 and getattr(model, "aux_head", None) is not None

    def _aux_ce(aux_logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        B, T, K = aux_logits.shape
        flat = aux_logits.reshape(B * T, K)
        ya_rep = y_a.repeat_interleave(T)
        yb_rep = y_b.repeat_interleave(T)
        return lam * loss_fn(flat, ya_rep) + (1.0 - lam) * loss_fn(flat, yb_rep)

    def _forward_loss(rgb, flow, y_a, y_b, lam):
        if use_aux:
            logits, (aux_rgb, aux_flow) = model.forward_with_aux(rgb, flow)
            main = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            aux = _aux_ce(aux_rgb, y_a, y_b, lam) + _aux_ce(aux_flow, y_a, y_b, lam)
            return logits, main + aux_loss_weight * aux
        logits = model(rgb, flow)
        loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
        return logits, loss

    bar = tqdm(data_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    for (rgb_batch, flow_batch), labels in bar:
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        flow_batch = flow_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_alpha > 0 and cutmix_alpha > 0:
            if random.random() < 0.5:
                rgb_batch, flow_batch, (y_a, y_b, lam) = mixup_batch_two_stream(
                    rgb_batch, flow_batch, labels, mixup_alpha
                )
            else:
                rgb_batch, flow_batch, (y_a, y_b, lam) = cutmix_batch_two_stream(
                    rgb_batch, flow_batch, labels, cutmix_alpha
                )
        elif mixup_alpha > 0:
            rgb_batch, flow_batch, (y_a, y_b, lam) = mixup_batch_two_stream(
                rgb_batch, flow_batch, labels, mixup_alpha
            )
        elif cutmix_alpha > 0:
            rgb_batch, flow_batch, (y_a, y_b, lam) = cutmix_batch_two_stream(
                rgb_batch, flow_batch, labels, cutmix_alpha
            )
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        stepped = True
        if amp and scaler is not None:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits, loss = _forward_loss(rgb_batch, flow_batch, y_a, y_b, lam)
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
                logits, loss = _forward_loss(rgb_batch, flow_batch, y_a, y_b, lam)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            logits, loss = _forward_loss(rgb_batch, flow_batch, y_a, y_b, lam)
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
            gate=float(torch.sigmoid(model.gate if hasattr(model, "gate") else getattr(model, "module", model).gate).item()),
        )

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch_two_stream(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    epoch_label: str = "",
) -> Tuple[float, float]:
    """Two-stream eval: data loader yields ((rgb, flow), labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(data_loader, desc=f"{epoch_label} val", leave=False, dynamic_ncols=True)
    for (rgb_batch, flow_batch), labels in bar:
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        flow_batch = flow_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if amp:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(rgb_batch, flow_batch)
                loss = loss_fn(logits, labels)
        else:
            logits = model(rgb_batch, flow_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)
        bar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / max(total, 1), correct / max(total, 1)


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
    is_two_stream = cfg.model.name == "two_stream"
    modality = str(cfg.dataset.get("modality", "rgb")).lower()
    is_flow_only = modality == "flow" and not is_two_stream
    if cfg.model.name == "modular":
        use_imagenet_norm = bool(cfg.model.spatial.get("pretrained", False))
    elif is_two_stream:
        use_imagenet_norm = bool(cfg.model.rgb.spatial.get("pretrained", False))
    else:
        use_imagenet_norm = bool(cfg.model.get("pretrained", False))
    image_size = int(cfg.dataset.get("image_size", 224))

    if is_flow_only:
        # Flow-only single-stream: data loader yields (flow_tensor, label).
        # Reuses FlowAwareClipAug/EvalTransform and discards the RGB output.
        train_clip_transform = build_flow_only_clip_transforms(
            image_size=image_size, is_training=True, use_imagenet_norm=use_imagenet_norm
        )
        eval_clip_transform = build_flow_only_clip_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
        )
        val_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=eval_clip_transform,
            sample_list=val_samples,
        )
    elif is_two_stream:
        # Two-stream: data loader yields ((rgb, flow), label). Train uses
        # FlowAwareClipAug (RRC + ColorJitter + RandomErasing + Farneback);
        # val uses deterministic resize + flow.
        train_clip_transform = build_flow_clip_transforms(
            image_size=image_size, is_training=True, use_imagenet_norm=use_imagenet_norm
        )
        eval_clip_transform = build_flow_clip_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=use_imagenet_norm
        )
        train_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=train_clip_transform,
            sample_list=train_samples,
        )
        val_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=int(cfg.dataset.num_frames),
            clip_transform=eval_clip_transform,
            sample_list=val_samples,
        )
    else:
        train_transform = build_transforms(
            image_size=image_size, is_training=True, use_imagenet_norm=use_imagenet_norm
        )
        eval_transform = build_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=use_imagenet_norm
        )
        # Strong clip-aware augmentation (RandomResizedCrop + ColorJitter + RandAugment +
        # RandomErasing, all consistent across frames). Used when training a temporal model
        # like TSM where per-frame independent crops would scramble the motion signal.
        use_strong_aug = bool(cfg.training.get("strong_clip_aug", False))
        if use_strong_aug:
            train_clip_transform = build_strong_clip_transform(
                image_size=image_size, use_imagenet_norm=use_imagenet_norm
            )
            train_dataset = VideoFrameDataset(
                root_dir=train_dir,
                num_frames=int(cfg.dataset.num_frames),
                clip_transform=train_clip_transform,
                sample_list=train_samples,
            )
        else:
            train_dataset = VideoFrameDataset(
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
    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optional per-frame deep-supervision head. Attached on the live model
    # before EMA so the EMA copy mirrors it (eval still uses main classifier).
    aux_cfg = cfg.training.get("aux_loss", None)
    aux_loss_weight = 0.0
    if aux_cfg is not None and bool(aux_cfg.get("enabled", False)):
        if cfg.model.name == "modular":
            attach_aux_head(model, num_classes=int(cfg.model.num_classes))
            model.aux_head = model.aux_head.to(device)
        elif is_two_stream:
            # Mirror exp9: per-frame deep-supervision on each substream.
            attach_aux_head(model.rgb_model, num_classes=int(cfg.model.num_classes))
            attach_aux_head(model.flow_model, num_classes=int(cfg.model.num_classes))
            model.rgb_model.aux_head = model.rgb_model.aux_head.to(device)
            model.flow_model.aux_head = model.flow_model.aux_head.to(device)
        else:
            raise ValueError("training.aux_loss only supported for model.name in {modular, two_stream}")
        aux_loss_weight = float(aux_cfg.get("weight", 0.3))
        print(f"[train] aux per-frame loss enabled (weight={aux_loss_weight})")

    # Optional RGB-stream hot-start from a single-stream checkpoint (e.g. exp9).
    # Only fires for two_stream and only on a fresh run (no resume file).
    last_path_probe = Path(cfg.training.checkpoint_path).resolve()
    last_path_probe = last_path_probe.with_name(last_path_probe.stem + "_last" + last_path_probe.suffix)
    rgb_init_path = cfg.training.get("rgb_init_from_checkpoint", None)
    resume_will_happen = bool(cfg.training.get("resume", True)) and last_path_probe.exists()
    if is_two_stream and rgb_init_path and not resume_will_happen:
        rgb_init_path = Path(str(rgb_init_path)).resolve()
        print(f"[train] hot-starting RGB stream from {rgb_init_path}")
        ckpt = torch.load(rgb_init_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        # Drop training-only keys (aux head etc) before loading.
        missing, unexpected = model.rgb_model.load_state_dict(sd, strict=False)
        # Aux-head weights from the source ckpt would land under "aux_head.*"; ok if missing.
        meaningful_missing = [k for k in missing if not k.startswith("aux_head")]
        if meaningful_missing:
            print(f"[train] RGB hot-start missing keys: {meaningful_missing[:5]}{'...' if len(meaningful_missing) > 5 else ''}")
        if unexpected:
            print(f"[train] RGB hot-start unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

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
        # Two-stream uses the RGB backbone as the hot-started slow group; flow
        # stream (fresh) and heads/gate train at head_lr.
        backbone_prefix = "rgb_model.spatial." if is_two_stream else "spatial."
        groups = build_two_group_param_groups(
            model,
            head_lr=float(cfg.training.lr),
            backbone_lr=float(backbone_lr),
            weight_decay=weight_decay,
            backbone_prefix=backbone_prefix,
        )
        optimizer = torch.optim.AdamW(groups)
        print(
            f"[train] AdamW + 2-group LR: backbone={backbone_lr}, head={cfg.training.lr}, "
            f"backbone_prefix='{backbone_prefix}', {len(groups)} param groups"
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
        start_epoch = int(state["epoch"])
        best_val_accuracy = float(state.get("best_val_accuracy", 0.0))
        if "rng" in state:
            restore_rng_state(state["rng"])
        print(f"[train] resuming from epoch {start_epoch + 1}/{cfg.training.epochs} (best so far {best_val_accuracy:.4f})")
    elif not resume_enabled and last_path.exists():
        print(f"[train] training.resume=false → ignoring {last_path}")

    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.training.get("cutmix_alpha", 0.0))

    train_fn = train_one_epoch_two_stream if is_two_stream else train_one_epoch
    eval_fn = evaluate_epoch_two_stream if is_two_stream else evaluate_epoch

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        label = f"[{epoch + 1}/{cfg.training.epochs}]"
        train_loss, train_acc = train_fn(
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
        )
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc = eval_fn(
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
        log_metrics = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
        }
        if is_two_stream:
            log_metrics["gate/rgb_weight"] = float(torch.sigmoid(model.gate).item())
        wandb_log(log_metrics, step=epoch + 1)

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

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb_log({"val/best_acc": best_val_accuracy})
    finish_wandb()


if __name__ == "__main__":
    main()
