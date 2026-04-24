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
from models.modular import build_modular_model
from utils import build_llrd_param_groups, build_transforms, set_seed, split_train_val


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
    grad_clip: float = 0.0,
    epoch_label: str = "",
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(data_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
    for video_batch, labels in bar:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        stepped = True
        if amp and scaler is not None:
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(video_batch)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Scaler lowers the scale when it skipped the step due to inf/NaN grads.
            stepped = scaler.get_scale() >= prev_scale
        else:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and stepped:
            scheduler.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
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
            with autocast(device_type=device.type, dtype=torch.float16):
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

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    # Layer-wise LR decay (for ViT-MAE fine-tuning). Falls back to a single group otherwise.
    llrd = float(cfg.training.get("llrd", 0.0))
    if llrd > 0.0:
        groups = build_llrd_param_groups(
            model,
            base_lr=float(cfg.training.lr),
            decay_rate=llrd,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(groups)
        print(f"[train] LLRD enabled (decay={llrd}): {len(groups)} param groups")
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
    scaler = GradScaler(device=device.type) if amp_enabled else None

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
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
            grad_clip=grad_clip,
            epoch_label=label,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, amp=amp_enabled, epoch_label=label
        )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
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

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
