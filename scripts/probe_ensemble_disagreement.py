"""
One-off: ensemble-disagreement probe on real val_dir.

Loads three checkpoints (exp2m, exp2n, exp3a), runs each once on val,
reports:
  - per-model top-1
  - pairwise disagreement (argmax-A != argmax-B)
  - softmax-mean ensemble top-1 (uniform, and val-weighted)
  - oracle ceiling (any-of-3 correct)
  - error decorrelation: P(B wrong | A wrong)

Single TTA=1 pass per model. ~3 min/model on the A5000.

    python scripts/probe_ensemble_disagreement.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


CKPTS = [
    ("exp2m", "checkpoints/exp2m_st_perceiver.pt"),
    ("exp2n", "checkpoints/exp2n_bornagain_bidir_swa.pt"),
    ("exp3a", "checkpoints/exp3a_vivit_pairwise_multiclip.pt"),
]


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


def load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any], DictConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"  [{ckpt_path.name}] dropped {len(unexpected)} unexpected (aux) keys", flush=True)
    model.to(device).eval()
    return model, ckpt, cfg


@hydra.main(version_base=None, config_path="../src/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(HydraConfig.get().runtime.cwd)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)

    probs_by_model: Dict[str, torch.Tensor] = {}
    labels_ref: torch.Tensor | None = None

    for tag, rel in CKPTS:
        ckpt_path = (repo_root / rel).resolve()
        print(f"\n=== loading {tag} from {ckpt_path}", flush=True)
        model, ckpt, model_cfg = load_model(ckpt_path, device)
        pretrained_used = bool(ckpt.get("pretrained", model_cfg.model.get("pretrained", False)))
        eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)
        num_frames = int(ckpt.get("num_frames", model_cfg.dataset.num_frames))
        dataset = VideoFrameDataset(
            root_dir=val_dir,
            num_frames=num_frames,
            transform=eval_transform,
            sample_list=val_samples,
            frame_offset_frac=0.0,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        print(f"  inference on {len(val_samples)} val videos (T={num_frames}, bs={batch_size})", flush=True)
        probs, labels = softmax_probs(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        else:
            assert torch.equal(labels_ref, labels), "label order changed between models"
        probs_by_model[tag] = probs
        del model
        torch.cuda.empty_cache()

    assert labels_ref is not None
    N = len(labels_ref)
    tags = [t for t, _ in CKPTS]

    print("\n=== per-model real-val top-1 ===")
    correct: Dict[str, torch.Tensor] = {}
    preds: Dict[str, torch.Tensor] = {}
    for tag in tags:
        p = probs_by_model[tag]
        pred = p.argmax(dim=-1)
        ok = (pred == labels_ref)
        preds[tag] = pred
        correct[tag] = ok
        print(f"  {tag:>5}: top-1 = {ok.float().mean().item():.4f}  ({int(ok.sum())}/{N})")

    print("\n=== pairwise disagreement (argmax) ===")
    for a, b in combinations(tags, 2):
        disagree = (preds[a] != preds[b]).float().mean().item()
        print(f"  {a} vs {b}: {disagree:.4f}  ({int((preds[a]!=preds[b]).sum())}/{N})")

    print("\n=== pairwise error correlation ===")
    for a, b in combinations(tags, 2):
        wa = ~correct[a]
        wb = ~correct[b]
        both_wrong = (wa & wb).float().sum().item()
        pa = wa.float().sum().item()
        pb = wb.float().sum().item()
        cond_a = both_wrong / max(pa, 1)
        cond_b = both_wrong / max(pb, 1)
        # baseline if errors were independent
        indep = (pa / N) * (pb / N) * N
        ratio = both_wrong / max(indep, 1e-9)
        print(f"  {a},{b}: P({b} wrong | {a} wrong) = {cond_a:.3f}   "
              f"P({a} wrong | {b} wrong) = {cond_b:.3f}   "
              f"obs/indep = {ratio:.2f}")

    print("\n=== softmax-mean ensemble (uniform) ===")
    avg = torch.stack([probs_by_model[t] for t in tags], dim=0).mean(dim=0)
    ens_pred = avg.argmax(dim=-1)
    ens_ok = (ens_pred == labels_ref).float().mean().item()
    print(f"  3-model uniform softmax-mean: top-1 = {ens_ok:.4f}")
    for a, b in combinations(tags, 2):
        pair = (probs_by_model[a] + probs_by_model[b]) / 2.0
        pp = pair.argmax(dim=-1)
        pok = (pp == labels_ref).float().mean().item()
        print(f"  {a}+{b}        : top-1 = {pok:.4f}")

    print("\n=== oracle ceiling (any-of-N correct) ===")
    any_ok = torch.zeros(N, dtype=torch.bool)
    for tag in tags:
        any_ok |= correct[tag]
    print(f"  any of 3: top-1 = {any_ok.float().mean().item():.4f}")
    for a, b in combinations(tags, 2):
        ao = correct[a] | correct[b]
        print(f"  any of {a},{b}: top-1 = {ao.float().mean().item():.4f}")

    # Save raw probs for downstream use (drafting exp3b weights, etc.)
    out_dir = repo_root / "probes"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ensemble_probs_real_val.pt"
    torch.save(
        {"labels": labels_ref, "probs": probs_by_model, "tags": tags, "val_samples": val_samples},
        out_path,
    )
    print(f"\nsaved raw probs → {out_path}")


if __name__ == "__main__":
    main()
