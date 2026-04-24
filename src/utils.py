"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> transforms.Compose:
    """
    Standard torchvision pipeline for single RGB frames.

    use_imagenet_norm:
        True  -> mean/std from ImageNet (usual when pretrained=True)
        False -> still scale to [0,1]; you can swap norms if you prefer
    """
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    # (batch_size, max_k) indices of top predictions
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()  # (max_k, batch_size)
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        # Any hit in the top-k row slice counts
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def build_llrd_param_groups(
    model: "torch.nn.Module",
    base_lr: float,
    decay_rate: float = 0.75,
    weight_decay: float = 0.05,
    no_decay_names: Tuple[str, ...] = ("bias", "cls_token", "pos_embed"),
) -> List[dict]:
    """
    Layer-wise LR decay for a modular video model with a ViT-MAE spatial slot.

    Layers ordered shallowest (patch_embed) → deepest (final norm); depth D.
    Each layer gets lr = base_lr * decay_rate ** (D - i), so the deepest encoder
    layer is ``base_lr`` and the patch_embed gets the most decayed rate.
    Temporal + classifier heads get ``base_lr`` unscaled.

    1-D params (LayerNorm, bias, CLS / pos-embed) get weight_decay=0.
    """
    if not hasattr(model, "spatial") or not hasattr(model.spatial, "ordered_layers"):
        raise ValueError(
            "build_llrd_param_groups expects a ModularVideoModel whose spatial encoder "
            "implements ordered_layers() (e.g. ViTMAEEncoder)."
        )

    spatial_layers = list(model.spatial.ordered_layers())
    depth = len(spatial_layers)

    groups: List[dict] = []
    assigned: set = set()

    # cls_token is not inside any sub-module; attach it to the shallowest layer.
    shallow_name = spatial_layers[0][0]

    for i, (name, module) in enumerate(spatial_layers):
        scale = decay_rate ** (depth - 1 - i)  # shallowest → most decayed
        lr = base_lr * scale
        params_decay, params_nodecay = [], []
        for pname, p in module.named_parameters(recurse=True):
            if not p.requires_grad or id(p) in assigned:
                continue
            full = f"spatial.{name}.{pname}"
            assigned.add(id(p))
            if p.ndim <= 1 or any(k in pname for k in no_decay_names):
                params_nodecay.append(p)
            else:
                params_decay.append(p)
        if params_decay:
            groups.append(
                {"params": params_decay, "lr": lr, "weight_decay": weight_decay, "group": f"spatial.{name}.decay"}
            )
        if params_nodecay:
            groups.append(
                {"params": params_nodecay, "lr": lr, "weight_decay": 0.0, "group": f"spatial.{name}.no_decay"}
            )
    # cls_token + any spatial params not inside a sub-module.
    leftover_decay, leftover_nodecay = [], []
    for pname, p in model.spatial.named_parameters():
        if id(p) in assigned or not p.requires_grad:
            continue
        assigned.add(id(p))
        if p.ndim <= 1 or any(k in pname for k in no_decay_names):
            leftover_nodecay.append(p)
        else:
            leftover_decay.append(p)
    leftover_lr = base_lr * (decay_rate ** (depth - 1))  # treat like shallowest
    if leftover_decay:
        groups.append({"params": leftover_decay, "lr": leftover_lr, "weight_decay": weight_decay, "group": f"spatial.{shallow_name}.leftover"})
    if leftover_nodecay:
        groups.append({"params": leftover_nodecay, "lr": leftover_lr, "weight_decay": 0.0, "group": f"spatial.{shallow_name}.leftover_no_decay"})

    # Head (temporal + classifier): base_lr, no decay at top.
    head_decay, head_nodecay = [], []
    for pname, p in model.named_parameters():
        if id(p) in assigned or not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in pname for k in no_decay_names):
            head_nodecay.append(p)
        else:
            head_decay.append(p)
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay, "group": "head.decay"})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0, "group": "head.no_decay"})

    return groups


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Shuffle then split a list of (video_path, label) into train and validation portions.

    Mirrors a standard random hold-out so train.py and evaluate.py stay consistent.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    """Crash-safe torch.save: write to a sibling .tmp file, then atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Keys that may differ across resumes without invalidating the saved state.
_RESUME_HASH_HARMLESS = {
    ("training", "num_workers"),
    ("training", "device"),
    ("training", "checkpoint_path"),
    ("training", "snapshot_every"),
    ("training", "resume"),
}


def make_resume_hash(cfg, keys: Tuple[str, ...] = ("model", "dataset", "training", "mae")) -> str:
    """Stable short hash over the config sub-trees that affect resume compatibility."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(cfg, resolve=True)
    snapshot = {}
    for top in keys:
        if top not in raw:
            continue
        sub = raw[top]
        if isinstance(sub, dict):
            sub = {k: v for k, v in sub.items() if (top, k) not in _RESUME_HASH_HARMLESS}
        snapshot[top] = sub
    blob = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def capture_rng_state() -> dict:
    """Snapshot torch / cuda / numpy / python RNG states for resumable training."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of capture_rng_state. Tolerates missing fields and map_location moves."""
    if "torch" in state:
        t = state["torch"]
        # torch.set_rng_state requires a CPU ByteTensor; torch.load with
        # map_location=<cuda> will have moved it, so force it back.
        if not (isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.dtype == torch.uint8):
            t = t.cpu().to(torch.uint8)
        torch.set_rng_state(t)
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [s.cpu().to(torch.uint8) if s.device.type != "cpu" else s for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
