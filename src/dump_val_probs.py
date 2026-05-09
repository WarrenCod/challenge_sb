"""Dump TTA=3 val-set softmax probs for a list of checkpoints.

Outputs per checkpoint:
    submissions/<run>_val_tta3.probs.npy      (N, num_classes) float32
And once globally:
    submissions/val_tta3.labels.npy           (N,) int64
    submissions/val_tta3.names.txt            video paths, one per line

Usage (run from /Data/challenge_sb):
    uv run python src/dump_val_probs.py \
        checkpoints/exp2c_mae_transformer.pt \
        checkpoints/exp2d_mae_distill_cutmix.pt \
        ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms

VAL_DIR = Path("/Data/challenge_sb/processed_data/val")
OUT_DIR = Path("/Data/challenge_sb/submissions")
TTA_OFFSETS = [0.0, 1.0 / 3.0, 2.0 / 3.0]
BATCH_SIZE = 32
NUM_WORKERS = 8


def _load_model(ckpt_path: Path, device: torch.device):
    raw: Dict[str, Any] = torch.load(ckpt_path, map_location=device)
    cfg = OmegaConf.create(raw["config"])
    model = build_model(cfg)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()
    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(raw.get("pretrained", cfg.model.get("pretrained", False)))
    return model, num_frames, pretrained


@torch.no_grad()
def _tta_probs(model, val_samples, num_frames, eval_transform, device) -> tuple[np.ndarray, np.ndarray]:
    accum: torch.Tensor | None = None
    labels_out: np.ndarray | None = None
    for pass_idx, offset in enumerate(TTA_OFFSETS, start=1):
        ds = VideoFrameDataset(
            root_dir=VAL_DIR,
            num_frames=num_frames,
            transform=eval_transform,
            sample_list=val_samples,
            frame_offset_frac=offset,
        )
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        )
        parts: List[torch.Tensor] = []
        labels_parts: List[torch.Tensor] = []
        for batch_idx, (video_batch, labels) in enumerate(loader, start=1):
            video_batch = video_batch.to(device)
            logits = model(video_batch)
            probs = torch.softmax(logits, dim=-1).float().cpu()
            parts.append(probs)
            if pass_idx == 1:
                labels_parts.append(labels.cpu())
        probs_full = torch.cat(parts, dim=0)
        accum = probs_full if accum is None else accum + probs_full
        if pass_idx == 1:
            labels_out = torch.cat(labels_parts, dim=0).numpy()
        print(f"  TTA pass {pass_idx}/{len(TTA_OFFSETS)} done", flush=True)
    avg = (accum / float(len(TTA_OFFSETS))).numpy().astype(np.float32)
    return avg, labels_out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: dump_val_probs.py <ckpt1.pt> <ckpt2.pt> ...", file=sys.stderr)
        sys.exit(1)

    ckpt_paths = [Path(p).resolve() for p in sys.argv[1:]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_samples = collect_video_samples(VAL_DIR)
    print(f"val samples: {len(val_samples)} from {VAL_DIR}", flush=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"device: {device}", flush=True)

    labels_path = OUT_DIR / "val_tta3.labels.npy"
    names_path = OUT_DIR / "val_tta3.names.txt"

    for i, ckpt_path in enumerate(ckpt_paths, start=1):
        run_name = ckpt_path.stem
        out_path = OUT_DIR / f"{run_name}_val_tta3.probs.npy"
        print(f"\n[{i}/{len(ckpt_paths)}] {run_name}", flush=True)
        if out_path.is_file():
            print(f"  skip (exists): {out_path}", flush=True)
            continue
        model, num_frames, pretrained = _load_model(ckpt_path, device)
        eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)
        probs, labels = _tta_probs(model, val_samples, num_frames, eval_transform, device)
        np.save(out_path, probs)
        print(f"  wrote {out_path} shape={probs.shape}", flush=True)
        if not labels_path.is_file():
            np.save(labels_path, labels)
            with names_path.open("w") as f:
                for p, _lbl in val_samples:
                    f.write(str(p) + "\n")
            print(f"  wrote {labels_path} shape={labels.shape}", flush=True)
        # report top-1
        top1 = float((probs.argmax(axis=1) == labels).mean())
        print(f"  TTA=3 val top-1 = {top1:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\nDONE_DUMP_VAL_PROBS", flush=True)


if __name__ == "__main__":
    main()
