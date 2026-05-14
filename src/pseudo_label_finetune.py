#!/usr/bin/env python3
"""FixMatch-lite pseudo-label fine-tune phase for exp3p.

Loads a trained Stage-2 checkpoint, predicts on the test set, keeps top-K
most-confident pseudo-labels (softmax >= ``threshold``), and fine-tunes the
same model for a short schedule on (train ∪ pseudo-train) with strong
augmentation on real labels and weak augmentation on pseudo-labels.

This is allowed under the competition rules — test images are visible, only
labels are hidden. The model only "learns from itself" — pseudo-labels are
its own confident predictions.

Usage::

    python src/pseudo_label_finetune.py \\
        training.checkpoint_path=checkpoints/videomae_exp3p.pt \\
        pseudo_label.threshold=0.85 \\
        pseudo_label.epochs=10
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, ConcatDataset

from create_submission import build_model_from_checkpoint, discover_all_test_videos
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import (
    build_transforms,
    build_videomae_clip_transform,
    set_seed,
    split_train_val,
    ModelEMA,
)


@torch.no_grad()
def predict_test_pseudo_labels(
    model: nn.Module,
    test_root: Path,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_imagenet_norm: bool,
) -> Tuple[List[str], List[Path], torch.Tensor]:
    """Returns (video_names, video_dirs, probs(N, K))."""
    video_names, video_dirs = discover_all_test_videos(test_root)
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)
    sample_list = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    probs_chunks: List[torch.Tensor] = []
    model.eval()
    for batch_idx, (video_batch, _y) in enumerate(loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch)
        probs_chunks.append(F.softmax(logits.float(), dim=-1).cpu())
        if batch_idx % max(1, len(loader) // 10) == 0:
            print(f"  [pseudo-pred] batch {batch_idx}/{len(loader)}", flush=True)
    probs = torch.cat(probs_chunks, dim=0)
    return video_names, video_dirs, probs


def select_confident_pseudo_labels(
    video_dirs: List[Path],
    probs: torch.Tensor,
    threshold: float,
    max_keep: int = -1,
) -> List[Tuple[Path, int]]:
    """Keep test samples whose top-1 softmax >= threshold. Returns [(dir, label), ...]."""
    confidences, labels = probs.max(dim=-1)
    keep_mask = confidences >= threshold
    kept_idx = keep_mask.nonzero(as_tuple=True)[0].tolist()
    kept = [(video_dirs[i], int(labels[i].item())) for i in kept_idx]
    if max_keep > 0 and len(kept) > max_keep:
        # Sort by confidence desc, keep top-K.
        order = sorted(range(len(kept)), key=lambda j: -float(confidences[kept_idx[j]].item()))
        kept = [kept[j] for j in order[:max_keep]]
    return kept


def histogram_str(labels: List[int], num_classes: int = 33) -> str:
    counts = [0] * num_classes
    for lbl in labels:
        counts[lbl] += 1
    nonzero = [(i, c) for i, c in enumerate(counts) if c > 0]
    nonzero.sort(key=lambda x: -x[1])
    head = ", ".join(f"{c}:{n}" for c, n in nonzero[:8])
    return f"{len(nonzero)}/{num_classes} classes used; top: {head}"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", flush=True)
        device_str = "cpu"
    device = torch.device(device_str)

    pl_cfg = cfg.get("pseudo_label", None) or OmegaConf.create({})
    threshold = float(pl_cfg.get("threshold", 0.85))
    pl_epochs = int(pl_cfg.get("epochs", 10))
    pl_lr = float(pl_cfg.get("lr", 5.0e-5))
    pl_batch_size = int(pl_cfg.get("batch_size", 32))
    pl_max_keep = int(pl_cfg.get("max_keep", -1))
    out_ckpt = Path(pl_cfg.get(
        "output_ckpt",
        str(Path(cfg.training.checkpoint_path).with_name(
            Path(cfg.training.checkpoint_path).stem + "_pl.pt"
        )),
    )).resolve()

    # --- Load model ---
    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    print(f"[pl] loading {ckpt_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    if "config" in ckpt and ckpt["config"] is not None:
        saved_cfg = OmegaConf.create(ckpt["config"])
        use_imagenet_norm = bool(saved_cfg.model.spatial.get("pretrained", False))
    else:
        use_imagenet_norm = False

    # --- Predict & select ---
    test_root = Path(cfg.dataset.test_dir).resolve()
    print(f"[pl] predicting on test_dir = {test_root}", flush=True)
    _, video_dirs, probs = predict_test_pseudo_labels(
        model, test_root, num_frames, pl_batch_size,
        int(cfg.training.num_workers), device, use_imagenet_norm,
    )
    kept = select_confident_pseudo_labels(
        video_dirs, probs, threshold=threshold, max_keep=pl_max_keep,
    )
    n_total = len(video_dirs)
    print(f"[pl] kept {len(kept)}/{n_total} pseudo-labels (threshold={threshold})", flush=True)
    if len(kept) == 0:
        raise SystemExit(
            f"No pseudo-labels passed threshold={threshold}. Lower it or check model."
        )
    print("[pl] " + histogram_str([lbl for _, lbl in kept]), flush=True)

    # --- Build combined train dataset ---
    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_train = collect_video_samples(train_dir)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    canonical = {p.name for p in val_dir.iterdir() if p.is_dir()}
    all_train = [s for s in all_train if s[0].parent.name in canonical]
    train_samples, val_samples = split_train_val(
        all_train, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed),
    )

    clip_transform = build_videomae_clip_transform(
        image_size=224, use_imagenet_norm=use_imagenet_norm
    )
    train_ds = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        clip_transform=clip_transform,
        sample_list=train_samples,
    )
    pseudo_ds = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        clip_transform=clip_transform,
        sample_list=kept,
    )
    combined = ConcatDataset([train_ds, pseudo_ds])
    print(f"[pl] combined dataset: {len(train_ds)} real + {len(pseudo_ds)} pseudo = {len(combined)}", flush=True)

    val_ds = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm),
        sample_list=[(p, 0) for p in [vp[0] for vp in val_samples]],
    )

    train_loader = DataLoader(
        combined, batch_size=pl_batch_size, shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # --- Fine-tune ---
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.get("label_smoothing", 0.1)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pl_lr,
        weight_decay=float(cfg.training.get("weight_decay", 0.05)),
    )
    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if str(cfg.training.get("amp_dtype", "fp16")).lower() in ("bf16", "bfloat16")
        else torch.float16
    )
    ema = ModelEMA(model, decay=float(cfg.training.get("ema_decay", 0.9999))) \
        if bool(cfg.training.get("ema", True)) else None

    print(f"[pl] fine-tuning {pl_epochs} epochs at lr={pl_lr}", flush=True)
    best_acc = 0.0
    for epoch in range(pl_epochs):
        model.train()
        running, total, correct = 0.0, 0, 0
        for batch_idx, (video_batch, labels) in enumerate(train_loader, start=1):
            video_batch = video_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(video_batch)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(video_batch)
                loss = loss_fn(logits, labels)
            loss.backward()
            if float(cfg.training.get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += labels.size(0)
        print(f"  [pl ep {epoch + 1}/{pl_epochs}] train loss {running / total:.4f} acc {correct / total:.4f}", flush=True)

    # --- Save ---
    save_payload = dict(ckpt)
    save_payload["model_state_dict"] = model.state_dict()
    save_payload["pseudo_label_meta"] = {
        "threshold": threshold,
        "kept": len(kept),
        "total": n_total,
        "pl_epochs": pl_epochs,
        "pl_lr": pl_lr,
    }
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_payload, out_ckpt)
    print(f"[pl] saved {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
