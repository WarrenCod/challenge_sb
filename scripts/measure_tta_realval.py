"""
One-off: measure real-val top-1 vs TTA pass count for a checkpoint.

Mirrors create_submission.py's TTA loop (N temporal offsets evenly spread on [0,1)),
but runs against the labeled val_dir so we can score predictions.

    python scripts/measure_tta_realval.py \
        training.checkpoint_path=checkpoints/exp3a_vivit_pairwise_multiclip.pt \
        +tta_clips_list=[1,3,5]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def load_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"[measure_tta] dropped {len(unexpected)} unexpected (aux) keys")
    model.to(device).eval()
    return model


@torch.no_grad()
def softmax_probs(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for video_batch, labels in loader:
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch)
        all_probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


@hydra.main(version_base=None, config_path="../src/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model_from_checkpoint(ckpt, device)

    pretrained_used = bool(ckpt.get("pretrained", cfg.model.get("pretrained", False)))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)

    tta_list = list(cfg.get("tta_clips_list", [1, 5]))
    # Gather every unique offset across all requested tta settings; run each
    # pass once, cache its probs, then score every setting from cached probs.
    needed_offsets: Dict[float, torch.Tensor] = {}
    for tta in tta_list:
        for k in range(tta):
            needed_offsets.setdefault(round(k / tta, 6), None)  # type: ignore[arg-type]

    labels_ref: torch.Tensor | None = None
    for pass_idx, offset in enumerate(sorted(needed_offsets.keys()), start=1):
        dataset = VideoFrameDataset(
            root_dir=val_dir,
            num_frames=num_frames,
            transform=eval_transform,
            sample_list=val_samples,
            frame_offset_frac=offset,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"pass {pass_idx}/{len(needed_offsets)} (offset_frac={offset:.3f})", flush=True)
        probs, labels = softmax_probs(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        needed_offsets[offset] = probs

    assert labels_ref is not None
    accuracies: Dict[int, float] = {}
    for tta in tta_list:
        offsets_for_tta = [round(k / tta, 6) for k in range(tta)]
        stacked = torch.stack([needed_offsets[o] for o in offsets_for_tta], dim=0)
        avg = stacked.mean(dim=0)
        preds = avg.argmax(dim=-1)
        top1 = float((preds == labels_ref).float().mean().item())
        accuracies[tta] = top1
        print(f"[tta={tta}] real-val top-1 = {top1:.4f} on {len(labels_ref)} videos", flush=True)

    base = accuracies[tta_list[0]]
    print("\n=== summary ===")
    for tta in tta_list:
        delta = accuracies[tta] - base
        print(f"  tta={tta:>2d}  top1={accuracies[tta]:.4f}   Δ vs tta={tta_list[0]}: {delta:+.4f}")


if __name__ == "__main__":
    main()
