"""
Diagnostic probe for Stage 1 (iBOT SSL) encoder.

Answers: did the encoder collapse, or did Stage 2 just fail to fine-tune on top?

What it does:
    1. Loads checkpoints/ibot_stage1.pt into ViTMAEEncoder (ViT-S/16).
    2. Extracts mean-pooled (over T=4 frames) CLS features for:
         - N_TRAIN random training videos (default 5000) as the kNN bank
         - the full real validation set (processed_data/val)
    3. Reports feature health stats:
         - mean L2 norm
         - per-dim std (collapse signal: ~0 across batch ⇒ dead dims)
         - effective rank via singular-value entropy
         - mean pairwise cosine similarity on a random subset
    4. Reports a 5-NN cosine classifier accuracy (real val) — compare to
       V-JEPA's 9.6% ceiling and to 1/33 ≈ 3.0% chance.

Run:
    python src/probe_stage1.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from dataset.video_dataset import VideoFrameDataset, collect_video_samples  # noqa: E402
from models.spatial.vit_mae import ViTMAEEncoder  # noqa: E402
from models.videomae import VideoMAEEncoder  # noqa: E402
from utils import build_transforms  # noqa: E402


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str,
    pool: str = "mean_t",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder on every clip; reduce to (B, D) via the named pool.

    pool:
        "mean_t"       — encoder returns (B, T, D); mean over T. (MAE/iBOT)
        "mean_st"      — encoder returns (B, T'*HW, D) via forward_features;
                          mean over all space-time tokens. (VideoMAE)
    """
    encoder.eval()
    feats, labels = [], []
    t0 = time.time()
    n_done = 0
    for clips, ys in loader:
        clips = clips.to(device, non_blocking=True)
        if pool == "mean_st":
            tokens = encoder.forward_features(clips)   # (B, T'*HW, D)
            clip_feat = tokens.mean(dim=1)             # (B, D)
        else:
            per_frame = encoder(clips)                 # (B, T, D)
            clip_feat = per_frame.mean(dim=1)          # (B, D)
        feats.append(clip_feat.cpu())
        labels.append(ys)
        n_done += clips.size(0)
        if n_done % 512 == 0 or n_done == len(loader.dataset):
            elapsed = time.time() - t0
            print(f"  [{label}] {n_done}/{len(loader.dataset)} in {elapsed:.1f}s")
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def feature_stats(feats: torch.Tensor, name: str) -> None:
    f = feats.float()
    norms = f.norm(dim=1)
    per_dim_std = f.std(dim=0)
    # Effective rank via singular-value entropy: exp(H(p)) where p = s^2 / Σs^2.
    s = torch.linalg.svdvals(f - f.mean(dim=0, keepdim=True))
    p = (s ** 2)
    p = p / p.sum().clamp(min=1e-12)
    entropy = -(p * (p.clamp(min=1e-12)).log()).sum().item()
    eff_rank = float(np.exp(entropy))

    # Mean pairwise cosine on a random subset (capped) — collapse ⇒ ~1.0.
    n = f.shape[0]
    k = min(512, n)
    idx = torch.randperm(n)[:k]
    sub = F.normalize(f[idx], dim=1)
    cos = sub @ sub.t()
    mask = ~torch.eye(k, dtype=torch.bool)
    cos_off = cos[mask]

    print(f"  --- {name} (N={n}, D={f.shape[1]}) ---")
    print(f"    L2 norm:        mean={norms.mean():.3f}  std={norms.std():.3f}")
    print(f"    per-dim std:    mean={per_dim_std.mean():.4f}  min={per_dim_std.min():.4f}  max={per_dim_std.max():.4f}")
    print(f"    # near-dead dims (std<1e-3):  {(per_dim_std < 1e-3).sum().item()} / {f.shape[1]}")
    print(f"    effective rank: {eff_rank:.1f} / {f.shape[1]}    (full rank ⇒ {f.shape[1]}, total collapse ⇒ 1)")
    print(f"    pairwise cos:   mean={cos_off.mean():.4f}  std={cos_off.std():.4f}    (collapse ⇒ ~1.0)")


def knn_accuracy(
    bank: torch.Tensor, bank_y: torch.Tensor,
    query: torch.Tensor, query_y: torch.Tensor,
    k: int = 5,
    num_classes: int = 33,
) -> tuple[float, float]:
    bank = F.normalize(bank.float(), dim=1)
    query = F.normalize(query.float(), dim=1)
    sims = query @ bank.t()                           # (Q, B)
    top_sim, top_idx = sims.topk(k, dim=1)            # (Q, k)
    top_labels = bank_y[top_idx]                      # (Q, k)
    # Similarity-weighted votes per class.
    weights = top_sim.clamp(min=0)                    # (Q, k)
    votes = torch.zeros(query.size(0), num_classes)
    votes.scatter_add_(1, top_labels, weights)
    top5 = votes.topk(5, dim=1).indices               # (Q, 5)
    pred = top5[:, 0]
    top1 = (pred == query_y).float().mean().item()
    top5_acc = (top5 == query_y.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, top5_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(REPO / "checkpoints" / "ibot_stage1.pt"))
    p.add_argument("--encoder", choices=["mae", "videomae"], default="mae",
                   help="Encoder family: 'mae' for MAE/iBOT/DINO ViT-S; 'videomae' for tubelet ViT.")
    p.add_argument("--train-dir", default=str(REPO / "processed_data" / "train"))
    p.add_argument("--val-dir",   default=str(REPO / "processed_data" / "val"))
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device = {device}")
    print(f"[probe] ckpt   = {args.ckpt}")

    if args.encoder == "videomae":
        encoder = VideoMAEEncoder(
            num_frames=args.num_frames,
            img_size=224,
            tubelet_time=2,
            tubelet_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            checkpoint_path=args.ckpt,
        ).to(device)
        pool = "mean_st"
        # VideoMAE pretrain uses 0.5/0.5 normalization (not ImageNet).
        transform = build_transforms(image_size=224, is_training=False, use_imagenet_norm=False)
    else:
        encoder = ViTMAEEncoder(variant="vit_s_16", checkpoint_path=args.ckpt).to(device)
        pool = "mean_t"
        transform = build_transforms(image_size=224, is_training=False, use_imagenet_norm=True)
    encoder.eval()

    train_samples = collect_video_samples(Path(args.train_dir))
    val_samples = collect_video_samples(Path(args.val_dir))
    print(f"[probe] train pool: {len(train_samples)} videos    val: {len(val_samples)} videos")

    rng = random.Random(args.seed)
    n_train = min(args.n_train, len(train_samples))
    train_subset = rng.sample(train_samples, n_train)

    train_ds = VideoFrameDataset(
        root_dir=args.train_dir,
        num_frames=args.num_frames,
        transform=transform,
        sample_list=train_subset,
    )
    val_ds = VideoFrameDataset(
        root_dir=args.val_dir,
        num_frames=args.num_frames,
        transform=transform,
        sample_list=val_samples,
    )

    common = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, **common)
    val_loader = DataLoader(val_ds, **common)

    print("[probe] extracting train features…")
    train_feats, train_labels = extract_features(encoder, train_loader, device, "train", pool=pool)
    print("[probe] extracting val features…")
    val_feats, val_labels = extract_features(encoder, val_loader, device, "val  ", pool=pool)

    print("\n[probe] FEATURE STATS")
    feature_stats(train_feats, "train (bank)")
    feature_stats(val_feats,   "val          ")

    print("\n[probe] kNN PROBE  (cosine, similarity-weighted, k={})".format(args.k))
    top1, top5 = knn_accuracy(
        train_feats, train_labels, val_feats, val_labels,
        k=args.k, num_classes=33,
    )
    print(f"    real val top-1: {top1:.4f}    top-5: {top5:.4f}")
    print(f"    reference     : chance ≈ 0.0303    V-JEPA ceiling ≈ 0.096    exp9 (Stage 2 leader) = 0.3899")


if __name__ == "__main__":
    main()
