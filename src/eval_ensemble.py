"""Average softmax across (checkpoint x temporal-offset) and either report
val top-1/top-5 or write a test submission CSV.

Each checkpoint carries its own Hydra config (saved by train.py), so backbone /
image_size / num_frames are picked up per-ckpt — heterogeneous ensembles
(e.g. VideoMAE-L 224 + VideoMAE-B 320) work without manual config.

Run from the repo root:

    python src/eval_ensemble.py \\
        --ckpt /Data/challenge_sb/best_videomae_l_k400.pt \\
        --ckpt /Data/challenge_sb/best_videomae_b16_k400.pt \\
        --offset 0.0 --offset 0.25 --offset 0.5 --offset 0.75 \\
        --mode val

    python src/eval_ensemble.py --ckpt … --mode test \\
        --submission-output processed_data/submission_ensemble.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from create_submission import discover_all_test_videos
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="Path to a checkpoint (repeatable; order doesn't matter for averaging).")
    p.add_argument("--offset", action="append", type=float, default=None,
                   help="Temporal frame-pick offset in units of step size (repeatable; default [0.0]).")
    p.add_argument("--mode", choices=["val", "test"], default="val")
    p.add_argument("--val-dir", default="processed_data/val")
    p.add_argument("--test-dir", default="processed_data/test")
    p.add_argument("--submission-output", default=None,
                   help="Required in --mode=test: CSV output path.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on the number of input clips (smoke testing).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_ckpt(path: Path, device: torch.device):
    raw = torch.load(path, map_location=device)
    if "config" not in raw or raw["config"] is None:
        raise SystemExit(f"{path}: checkpoint has no 'config'; cannot reconstruct model.")
    cfg = OmegaConf.create(raw["config"])
    model = build_model(cfg)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device).eval()
    pretrained = bool(raw.get("pretrained", cfg.model.get("pretrained", True)))
    image_size = int(cfg.get("model", {}).get("image_size", 224))
    num_frames = int(raw.get("num_frames", cfg.get("dataset", {}).get("num_frames", 4)))
    return model, image_size, num_frames, pretrained, path.stem


@torch.no_grad()
def run_one(model, loader, device, total, tag):
    n = len(loader)
    log_every = max(1, n // 10)
    seen = 0
    probs_chunks, label_chunks = [], []
    for i, (video, labels) in enumerate(loader, start=1):
        video = video.to(device, non_blocking=True)
        logits = model(video)
        probs_chunks.append(logits.softmax(dim=1).cpu())
        label_chunks.append(labels)
        seen += video.size(0)
        if i % log_every == 0 or i == n:
            print(f"  {tag} batch {i}/{n} ({seen}/{total})", flush=True)
    return torch.cat(probs_chunks, dim=0), torch.cat(label_chunks, dim=0)


def main() -> None:
    args = parse_args()
    offsets: List[float] = args.offset if args.offset else [0.0]
    if args.mode == "test" and not args.submission_output:
        print("error: --submission-output required in --mode=test", file=sys.stderr)
        sys.exit(2)

    set_seed(args.seed)
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.", flush=True)
        device_str = "cpu"
    device = torch.device(device_str)

    if args.mode == "val":
        root = Path(args.val_dir).resolve()
        sample_list = collect_video_samples(root)
        if args.max_samples is not None:
            sample_list = sample_list[: int(args.max_samples)]
        video_names = None
    else:
        root = Path(args.test_dir).resolve()
        video_names, video_dirs = discover_all_test_videos(root)
        if args.max_samples is not None:
            video_names = video_names[: int(args.max_samples)]
            video_dirs = video_dirs[: int(args.max_samples)]
        sample_list = [(p, 0) for p in video_dirs]

    total = len(sample_list)
    print(f"[{args.mode}] root={root}  samples={total}", flush=True)
    print(f"Checkpoints ({len(args.ckpt)}):", flush=True)
    for p in args.ckpt:
        print(f"  - {p}", flush=True)
    print(f"Offsets: {offsets}", flush=True)

    accumulated: torch.Tensor | None = None
    labels_ref: torch.Tensor | None = None
    pair_count = 0

    for ckpt_path in args.ckpt:
        ckpt_path = Path(ckpt_path).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        model, image_size, num_frames, pretrained, tag = load_ckpt(ckpt_path, device)
        eval_tf = build_transforms(
            is_training=False, use_imagenet_norm=pretrained, image_size=image_size,
        )
        print(f"[{tag}] image_size={image_size}, num_frames={num_frames}, "
              f"pretrained={pretrained}", flush=True)

        for off in offsets:
            ds = VideoFrameDataset(
                root_dir=root, num_frames=num_frames, transform=eval_tf,
                sample_list=sample_list, frame_offset=off,
            )
            dl = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            )
            probs, labels = run_one(model, dl, device, total, tag=f"[{tag}, off={off:+.2f}]")
            accumulated = probs if accumulated is None else accumulated + probs
            if args.mode == "val":
                if labels_ref is None:
                    labels_ref = labels
                elif not torch.equal(labels_ref, labels):
                    raise RuntimeError("Sample order drifted between runs — refusing to average.")
            pair_count += 1

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert accumulated is not None and pair_count > 0
    accumulated /= float(pair_count)

    if args.mode == "val":
        assert labels_ref is not None
        top1_pred = accumulated.argmax(dim=1)
        top1 = (top1_pred == labels_ref).float().mean().item()
        _, top5_idx = accumulated.topk(5, dim=1, largest=True, sorted=True)
        top5 = top5_idx.eq(labels_ref.view(-1, 1)).any(dim=1).float().mean().item()
        print()
        print(f"=== Ensemble: {pair_count} (ckpt x offset) softmax pairs averaged ===")
        print(f"Samples: {accumulated.size(0)}")
        print(f"Top-1 accuracy: {top1:.4f}")
        print(f"Top-5 accuracy: {top5:.4f}")
    else:
        assert video_names is not None
        preds = accumulated.argmax(dim=1).tolist()
        out_path = Path(args.submission_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_name", "predicted_class"])
            for name, pred in zip(video_names, preds):
                w.writerow([name, pred])
        print(f"\nWrote {len(preds)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
