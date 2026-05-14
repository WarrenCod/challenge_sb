#!/usr/bin/env python3
"""
Softmax-averaged ensemble inference over multiple Stage-2 checkpoints.

Each checkpoint embeds its own merged Hydra config, so this script can rebuild
the correct architecture per ckpt. Per-head softmax probabilities are averaged
(robust to differing logit scales between heads — bare-Linear vs
LayerNorm+Linear), and the argmax is written to the submission CSV.

Usage::

    python src/ensemble_submissions.py \\
        --ckpts checkpoints/videomae_meanpool.pt checkpoints/videomae_transformer.pt \\
        --output submissions/videomae_ensemble.csv

The test set is discovered the same way ``create_submission.py`` does it
(``processed_data/test/video_*`` folders, sorted by name). Single-scale
center-crop only; **no geometric TTA** (direction-sensitive labels).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from create_submission import build_model_from_checkpoint, discover_all_test_videos  # noqa: E402
from dataset.video_dataset import VideoFrameDataset  # noqa: E402
from utils import build_transforms, set_seed  # noqa: E402


def _norm_from_ckpt(ckpt: Dict[str, Any]) -> bool:
    """Recover the use_imagenet_norm flag the model was trained with.

    Modular models record `model.spatial.pretrained` in the saved config;
    legacy single-class models record `pretrained` at the top of the payload.
    Defaults to False (videomae uses 0.5/0.5 normalization).
    """
    saved_cfg = ckpt.get("config")
    if saved_cfg is not None:
        try:
            cfg = OmegaConf.create(saved_cfg)
            if cfg.model.get("name") == "modular":
                return bool(cfg.model.spatial.get("pretrained", False))
            return bool(cfg.model.get("pretrained", False))
        except Exception:
            pass
    return bool(ckpt.get("pretrained", False))


@torch.no_grad()
def _ckpt_probs(
    ckpt_path: Path,
    test_root: Path,
    video_dirs: List[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    """Load one ckpt, run inference, return (N, num_classes) softmax probs."""
    print(f"[ensemble] loading {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    num_frames = int(ckpt.get("num_frames", 4))
    use_imagenet_norm = _norm_from_ckpt(ckpt)
    transform = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

    sample_list = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=transform,
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
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    for batch_idx, (video_batch, _y) in enumerate(loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch)
        probs_chunks.append(F.softmax(logits.float(), dim=-1).cpu())
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(f"  [{ckpt_path.name}] batch {batch_idx}/{n_batches}", flush=True)
    return torch.cat(probs_chunks, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True, help="One or more Stage-2 checkpoint paths.")
    p.add_argument("--test-dir", default=str(REPO / "processed_data" / "test"))
    p.add_argument("--output", default=str(REPO / "submissions" / "videomae_ensemble.csv"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] device = {device}", flush=True)

    ckpt_paths = [Path(c).resolve() for c in args.ckpts]
    for cp in ckpt_paths:
        if not cp.is_file():
            raise SystemExit(f"Checkpoint not found: {cp}")
    test_root = Path(args.test_dir).resolve()

    print(f"[ensemble] indexing test videos under {test_root}", flush=True)
    video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"[ensemble] {len(video_dirs)} test videos", flush=True)

    accum: torch.Tensor | None = None
    for cp in ckpt_paths:
        probs = _ckpt_probs(
            cp,
            test_root,
            video_dirs,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if probs.size(0) != len(video_names):
            raise RuntimeError(
                f"{cp} produced {probs.size(0)} predictions, expected {len(video_names)}"
            )
        accum = probs if accum is None else accum + probs

    assert accum is not None
    accum /= len(ckpt_paths)
    preds = accum.argmax(dim=-1).tolist()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ensemble] writing {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, preds):
            w.writerow([name, int(pred)])
    print(f"[ensemble] done. wrote {len(preds)} rows.", flush=True)


if __name__ == "__main__":
    main()
