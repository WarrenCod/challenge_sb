#!/usr/bin/env python3
"""
Run a trained checkpoint on the test split and write a submission CSV::

    video_name,predicted_class

Uses the same Hydra layout as ``train.py`` / ``evaluate.py``. Paths and checkpoint
come from the composed config (see ``configs/data/default.yaml`` and
``configs/train/default.yaml``).

Example (from ``src/``)::

    python create_submission.py
    python create_submission.py training.checkpoint_path=/path/to/best_model.pt
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset
from train import build_model
from utils import (
    build_flow_clip_transforms,
    build_flow_only_clip_transforms,
    build_transforms,
    set_seed,
)


def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader]


def _index_video_folders(test_root: Path) -> Dict[str, Path]:
    """
    Walk ``test_root`` **once** and map each ``video_<id>`` folder name -> path.

    Prunes search at each ``video_*`` directory (frames live there; no need to
    descend), so we avoid scanning every JPEG.

    This replaces ``glob(f'**/{name}')`` per manifest row, which re-walked the
    full tree thousands of times.
    """
    test_root = test_root.resolve()
    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(
                    f"Duplicate video folder name {name!r}: {index[name]} and {p}"
                )
            index[name] = p
            dirs.remove(name)
    return index


def resolve_video_dirs(test_root: Path, video_names: List[str]) -> List[Path]:
    """Map each ``video_<id>`` folder name to a path using a pre-built index."""
    index = _index_video_folders(test_root)
    out: List[Path] = []
    missing: List[str] = []
    for name in video_names:
        p = index.get(name)
        if p is None:
            missing.append(name)
        else:
            out.append(p)
    if missing:
        sample = ", ".join(repr(m) for m in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}"
        )
    return out


def discover_all_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    """
    Discover all ``video_*`` folders under ``test_root`` and return them sorted.

    Returns:
        (video_names, video_dirs) sorted by video folder name.
    """
    index = _index_video_folders(test_root)
    video_names = sorted(index.keys())
    video_dirs = [index[name] for name in video_names]
    return video_names, video_dirs


def build_model_from_checkpoint(ckpt: Dict[str, Any]) -> torch.nn.Module:
    """Rebuild the model using the saved Hydra config when available."""
    if "config" in ckpt and ckpt["config"] is not None:
        cfg = OmegaConf.create(ckpt["config"])
        return build_model(cfg)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": ckpt.get("model_name", "cnn_baseline"),
                "num_classes": int(ckpt["num_classes"]),
                "pretrained": bool(ckpt.get("pretrained", True)),
                "lstm_hidden_size": int(ckpt.get("lstm_hidden_size", 512)),
            }
        }
    )
    return build_model(cfg)


def _forward(model: torch.nn.Module, batch_inputs, device: torch.device) -> torch.Tensor:
    """Single forward; handles both single-tensor and (rgb, flow) tuple inputs."""
    if isinstance(batch_inputs, (list, tuple)):
        rgb, flow = batch_inputs
        rgb = rgb.to(device)
        flow = flow.to(device)
        return model(rgb, flow)
    return model(batch_inputs.to(device))


def _batch_size(batch_inputs) -> int:
    if isinstance(batch_inputs, (list, tuple)):
        return batch_inputs[0].size(0)
    return batch_inputs.size(0)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
) -> List[int]:
    """Run the model on the loader; print batch progress to stdout."""
    model.eval()
    preds: List[int] = []
    n_batches = len(loader)
    # About 10 progress lines for long runs; at least every batch if tiny
    log_interval = max(1, n_batches // 10)
    processed = 0
    for batch_idx, (batch_inputs, _labels) in enumerate(loader, start=1):
        logits = _forward(model, batch_inputs, device)
        batch_pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(int(p) for p in batch_pred)
        bs = _batch_size(batch_inputs)
        processed += bs
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(
                f"  Inference batch {batch_idx}/{n_batches} "
                f"({processed}/{total_videos} clips)",
                flush=True,
            )
    return preds


@torch.no_grad()
def accumulate_softmax(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
    num_classes: int,
    label: str,
) -> torch.Tensor:
    """Run inference and return summed softmax (B_total, num_classes) on CPU.

    Used by multi-clip TTA: caller calls this once per temporal offset and
    averages the accumulated softmax before argmaxing.
    """
    model.eval()
    out = torch.zeros(total_videos, num_classes, dtype=torch.float32)
    cursor = 0
    n_batches = len(loader)
    log_interval = max(1, n_batches // 5)
    for batch_idx, (batch_inputs, _labels) in enumerate(loader, start=1):
        logits = _forward(model, batch_inputs, device)
        probs = logits.softmax(dim=1).cpu()
        bs = probs.size(0)
        out[cursor : cursor + bs] = probs
        cursor += bs
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(
                f"  [{label}] batch {batch_idx}/{n_batches} ({cursor}/{total_videos})",
                flush=True,
            )
    return out


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint missing required keys: {missing}")
    if unexpected:
        print(f"[load] dropping training-only keys: {unexpected}", flush=True)
    model.to(device)
    print(f"Model on device: {device}", flush=True)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.get("pretrained", False)))
    saved_cfg = ckpt.get("config") or {}
    saved_dataset = saved_cfg.get("dataset", {}) if isinstance(saved_cfg, dict) else {}
    image_size = int(saved_dataset.get("image_size", cfg.dataset.get("image_size", 224)))
    saved_model_name = (saved_cfg.get("model", {}) or {}).get("name") if isinstance(saved_cfg, dict) else None
    is_two_stream = saved_model_name == "two_stream"
    modality = str(saved_dataset.get("modality", "rgb")).lower() if isinstance(saved_dataset, dict) else "rgb"
    is_flow_only = modality == "flow" and not is_two_stream
    if is_two_stream:
        eval_transform = None  # signal: use clip_transform path below
        eval_clip_transform = build_flow_clip_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=pretrained
        )
        print("[load] two-stream model: using flow-aware eval transform.", flush=True)
    elif is_flow_only:
        eval_transform = None
        eval_clip_transform = build_flow_only_clip_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=pretrained
        )
        print("[load] flow-only model: using flow eval clip transform.", flush=True)
    else:
        eval_clip_transform = None
        eval_transform = build_transforms(
            image_size=image_size, is_training=False, use_imagenet_norm=pretrained
        )

    test_root = Path(cfg.dataset.test_dir).resolve()
    output_path = Path(cfg.dataset.submission_output).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")

    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f"Reading manifest: {manifest_path}", flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
        print(
            f"Resolved {len(video_dirs)} video folders from manifest for inference.",
            flush=True,
        )
    else:
        print(
            "No dataset.test_manifest provided; using all video_* folders found in test_dir.",
            flush=True,
        )
        video_names, video_dirs = discover_all_test_videos(test_root)
        print(
            f"Discovered {len(video_dirs)} video folders (sorted by video name).",
            flush=True,
        )
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    batch_size = int(cfg.training.batch_size)
    tta_clips = int(cfg.training.get("tta_clips", 1))
    if tta_clips < 1:
        raise ValueError(f"training.tta_clips must be >=1, got {tta_clips}")

    if tta_clips == 1:
        dataset = VideoFrameDataset(
            root_dir=test_root,
            num_frames=num_frames,
            transform=eval_transform,
            clip_transform=eval_clip_transform,
            sample_list=sample_list,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        print(
            f"Starting inference: {len(dataset)} clips, batch_size={batch_size}, "
            f"{len(loader)} batches",
            flush=True,
        )
        predictions = run_inference(model, loader, device, total_videos=len(dataset))
    else:
        # Multi-clip TTA: N evenly spaced sub-bin offsets, average softmax.
        # Offsets are -0.5..+0.5 of one inter-frame step (centered around 0 for
        # symmetry), so N=4 gives offsets {-0.375, -0.125, 0.125, 0.375}.
        num_classes = int(ckpt.get("num_classes"))
        offsets = [(-0.5 + (k + 0.5) / tta_clips) for k in range(tta_clips)]
        print(
            f"TTA over {tta_clips} temporal clips (offsets={[round(o, 3) for o in offsets]}); "
            f"batch_size={batch_size}",
            flush=True,
        )
        accum: torch.Tensor | None = None
        for i, off in enumerate(offsets):
            dataset = VideoFrameDataset(
                root_dir=test_root,
                num_frames=num_frames,
                transform=eval_transform,
                clip_transform=eval_clip_transform,
                sample_list=sample_list,
                frame_offset_fraction=off,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(cfg.training.num_workers),
                pin_memory=(device.type == "cuda"),
            )
            probs = accumulate_softmax(
                model,
                loader,
                device,
                total_videos=len(dataset),
                num_classes=num_classes,
                label=f"clip {i + 1}/{tta_clips} off={off:+.3f}",
            )
            accum = probs if accum is None else (accum + probs)
        assert accum is not None
        predictions = accum.argmax(dim=1).tolist()
    print("Inference finished.", flush=True)

    if len(predictions) != len(video_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != manifest length {len(video_names)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing submission CSV: {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
