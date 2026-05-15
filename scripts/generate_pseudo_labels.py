"""
One-off: produce soft-target pseudo-labels on the test set from an ensemble
of frozen models. Output is consumed by `PseudoLabelVideoDataset` in
training (exp3b noisy-student).

Filter:
  Keep a test video iff (max softmax prob >= threshold)
                   AND (all N models agree on argmax).
  Both gates matter: prior ensemble probe (2026-05-14) showed errors at
  P≈0.87 → confidence alone admits "all confidently wrong" samples;
  agreement alone admits low-confidence agreements. Together they screen.

Usage:
    python scripts/generate_pseudo_labels.py \
        --models checkpoints/exp2m_st_perceiver.pt \
                 checkpoints/exp2n_bornagain_bidir_swa.pt \
                 checkpoints/exp3a_vivit_pairwise_multiclip.pt \
        --threshold 0.7 \
        --require_argmax_agreement \
        --output processed_data/pseudo_labels_v1.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset.video_dataset import (
    VideoFrameDataset,
    _list_frame_paths,
    collect_video_samples,
)
from train import build_model
from utils import build_transforms, set_seed


class TestVideoDataset(Dataset):
    """Indexes test/video_<id> folders (no class subfolders)."""

    def __init__(self, test_dir: Path, num_frames: int, transform) -> None:
        self.test_dir = test_dir
        self.num_frames = int(num_frames)
        self.transform = transform
        self.video_dirs: List[Path] = sorted(
            p for p in test_dir.iterdir() if p.is_dir() and p.name.startswith("video_")
        )
        if not self.video_dirs:
            raise RuntimeError(f"No video_* folders under {test_dir}")

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, idx: int):
        from dataset.video_dataset import _pick_frame_indices
        from PIL import Image

        vdir = self.video_dirs[idx]
        frame_paths = _list_frame_paths(vdir)
        indices = _pick_frame_indices(len(frame_paths), self.num_frames, offset_frac=0.0)
        frames = []
        for fi in indices:
            with Image.open(frame_paths[fi]) as im:
                frames.append(self.transform(im.convert("RGB")))
        return torch.stack(frames, dim=0), vdir.name


@torch.no_grad()
def run_model(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    probs_all = []
    names_all: List[str] = []
    for video_batch, names in loader:
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch)
        probs_all.append(torch.softmax(logits.float(), dim=-1).cpu())
        names_all.extend(names)
    return torch.cat(probs_all, dim=0), names_all


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  [{ckpt_path.name}] dropped {len(unexpected)} unexpected (aux) keys")
    model.to(device).eval()
    pretrained = bool(ckpt.get("pretrained", cfg.model.get("pretrained", False)))
    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    return model, pretrained, num_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="checkpoint paths")
    ap.add_argument("--test_dir", default=str(REPO_ROOT / "processed_data" / "test"))
    ap.add_argument("--output", required=True, help="output .pt path")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--require_argmax_agreement", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probs_by_model: List[torch.Tensor] = []
    names_ref: List[str] | None = None
    src_model_names: List[str] = []

    for ckpt_rel in args.models:
        ckpt_path = (REPO_ROOT / ckpt_rel).resolve() if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel)
        print(f"\n=== loading {ckpt_path.name}", flush=True)
        model, pretrained, num_frames = load_model(ckpt_path, device)
        transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)
        dataset = TestVideoDataset(test_dir, num_frames=num_frames, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"  inference on {len(dataset)} test videos (T={num_frames}, bs={args.batch_size})", flush=True)
        probs, names = run_model(model, loader, device)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise RuntimeError("video order differs across models — index mismatch")
        probs_by_model.append(probs)
        src_model_names.append(ckpt_path.name)
        del model
        torch.cuda.empty_cache()

    assert names_ref is not None
    stacked = torch.stack(probs_by_model, dim=0)            # (M, N, K)
    soft_mean = stacked.mean(dim=0)                          # (N, K)
    argmaxes = stacked.argmax(dim=-1)                        # (M, N)
    max_probs = soft_mean.max(dim=-1).values                 # (N,)
    ensemble_argmax = soft_mean.argmax(dim=-1)               # (N,)

    keep = max_probs >= float(args.threshold)
    if args.require_argmax_agreement:
        agree = (argmaxes == argmaxes[0:1]).all(dim=0)       # (N,)
        keep = keep & agree

    n_total = len(names_ref)
    n_keep = int(keep.sum().item())
    print(f"\n=== filtering ===")
    print(f"  total test videos       : {n_total}")
    print(f"  confidence ≥ {args.threshold:.2f}      : {int((max_probs >= args.threshold).sum())}")
    if args.require_argmax_agreement:
        agree_t = (argmaxes == argmaxes[0:1]).all(dim=0)
        print(f"  argmax agreement        : {int(agree_t.sum())}")
    print(f"  KEPT (both gates)       : {n_keep}  ({100.0 * n_keep / n_total:.1f}%)")

    kept_names = [n for n, k in zip(names_ref, keep.tolist()) if k]
    kept_soft = soft_mean[keep]
    kept_max = max_probs[keep]
    kept_argmax = ensemble_argmax[keep]

    print(f"\n=== per-class histogram (kept pseudo-labels) ===")
    hist = Counter(int(c) for c in kept_argmax.tolist())
    for cls in sorted(hist):
        print(f"  class {cls:>2}: {hist[cls]}")

    payload: Dict[str, Any] = {
        "video_names": kept_names,
        "soft_targets": kept_soft,
        "max_probs": kept_max,
        "argmax": kept_argmax,
        "threshold": float(args.threshold),
        "require_argmax_agreement": bool(args.require_argmax_agreement),
        "source_models": src_model_names,
        "n_total_test": n_total,
        "n_kept": n_keep,
    }
    torch.save(payload, output_path)
    print(f"\nsaved {n_keep} pseudo-labels → {output_path}")


if __name__ == "__main__":
    main()
