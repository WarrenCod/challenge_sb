"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
        # No RandomHorizontalFlip: SSv2 labels are direction-sensitive
        # (e.g. "Pushing X from left to right" vs "... right to left").
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
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


class ConsistentClipAug:
    """Heavy augmentation applied identically across all frames of a clip.

    Independent per-frame transforms would scramble the spatial-temporal
    correspondence that TSM (and any temporal model) relies on. This applies one
    set of random params (crop coords, jitter factors, RandAugment op) to every
    frame in the clip, then ToTensor + Normalize.

    Input: list of PIL.Image frames.
    Output: (T, C, H, W) float tensor.
    """

    def __init__(
        self,
        image_size: int = 224,
        scale: Tuple[float, float] = (0.5, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        randaug_n: int = 2,
        randaug_m: int = 9,
        erase_p: float = 0.25,
        use_imagenet_norm: bool = True,
    ) -> None:
        from torchvision.transforms import ColorJitter, RandAugment, RandomResizedCrop

        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.erase_p = erase_p
        self._RandomResizedCrop = RandomResizedCrop
        self._ColorJitter = ColorJitter
        self._randaug = RandAugment(num_ops=randaug_n, magnitude=randaug_m) if randaug_n > 0 else None
        if use_imagenet_norm:
            self._mean = [0.485, 0.456, 0.406]
            self._std = [0.229, 0.224, 0.225]
        else:
            self._mean = [0.5, 0.5, 0.5]
            self._std = [0.5, 0.5, 0.5]

    def __call__(self, frames):
        import torchvision.transforms.functional as TF

        # 1. RandomResizedCrop — pick params from the first frame, apply to all.
        i, j, h, w = self._RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        frames = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]

        # 2. ColorJitter — pick brightness/contrast/saturation/hue once.
        b_range = (max(0.0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
        c_range = (max(0.0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
        s_range = (max(0.0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
        h_range = (-self.hue, self.hue) if self.hue > 0 else None
        fn_idx, b, c, s, h_factor = self._ColorJitter.get_params(b_range, c_range, s_range, h_range)
        jittered = []
        for img in frames:
            for fid in fn_idx:
                if fid == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fid == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fid == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
                elif fid == 3 and h_factor is not None:
                    img = TF.adjust_hue(img, h_factor)
            jittered.append(img)

        # 3. RandAugment — replay the same RNG seed across frames so each frame
        # gets the same op. This is approximate (RandAugment also uses internal
        # state for sub-magnitude jitter) but visually consistent enough for TSM.
        if self._randaug is not None:
            seed = random.randint(0, 2**32 - 1)
            augmented = []
            for img in jittered:
                random.seed(seed)
                torch.manual_seed(seed)
                augmented.append(self._randaug(img))
            jittered = augmented

        # 4. ToTensor + Normalize.
        tensors = [TF.normalize(TF.to_tensor(img), mean=self._mean, std=self._std) for img in jittered]
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)

        # 5. RandomErasing — same rectangle erased on every frame.
        if self.erase_p > 0 and random.random() < self.erase_p:
            from torchvision.transforms import RandomErasing
            erase_i, erase_j, erase_h, erase_w, erase_v = RandomErasing.get_params(
                clip[0], scale=(0.02, 0.33), ratio=(0.3, 3.3), value=[0.0]
            )
            clip[:, :, erase_i : erase_i + erase_h, erase_j : erase_j + erase_w] = erase_v

        return clip


def build_strong_clip_transform(image_size: int = 224, use_imagenet_norm: bool = True) -> ConsistentClipAug:
    """Default heavy-aug clip transform for TSM-style training (no hflip, SSv2-safe)."""
    return ConsistentClipAug(image_size=image_size, use_imagenet_norm=use_imagenet_norm)


def _farneback_clip(pil_frames_gray_uint8: List[np.ndarray]) -> torch.Tensor:
    """Compute (T-1, 2, H, W) dense flow tensor between consecutive grayscale frames.

    Inputs are uint8 numpy (H, W). Output is float32, raw (dx, dy) in pixels.
    Caller is responsible for any scale normalization.
    """
    import cv2

    flows: List[torch.Tensor] = []
    for prev, nxt in zip(pil_frames_gray_uint8[:-1], pil_frames_gray_uint8[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # (H, W, 2) float32
        flow_t = torch.from_numpy(flow).permute(2, 0, 1).contiguous()  # (2, H, W)
        flows.append(flow_t)
    return torch.stack(flows, dim=0)  # (T-1, 2, H, W)


# Flow magnitudes from the smoke test sit at ~5 px mean / ~20 px p99 at 256x256.
# Dividing by 20 puts the bulk in [-1, 1] which keeps the flow conv1 + BN happy.
FLOW_NORM_DIVISOR: float = 20.0


class FlowAwareClipAug:
    """Two-stream training augmentation: returns (rgb_clip, flow_clip).

    Pipeline:
      1. RandomResizedCrop on PIL frames (consistent across the clip).
      2. Farneback flow on grayscale of the post-RRC frames (pristine flow,
         spatially aligned with the cropped RGB).
      3. ColorJitter on RGB only (flow is geometric, not photometric).
      4. ToTensor + Normalize on RGB.
      5. Same RandomErasing rectangle applied to RGB and flow so both streams
         see the same occlusion.

    RandAugment is intentionally *not* used here — its geometric ops would
    spatially mis-align RGB and flow, and reverting that on the flow tensor is
    too fragile for V1.

    Output:
      rgb_clip:  (T, 3, H, W) float, ImageNet-normalized
      flow_clip: (T-1, 2, H, W) float, divided by FLOW_NORM_DIVISOR
    """

    def __init__(
        self,
        image_size: int = 224,
        scale: Tuple[float, float] = (0.5, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        erase_p: float = 0.25,
        use_imagenet_norm: bool = True,
        flow_norm_divisor: float = FLOW_NORM_DIVISOR,
    ) -> None:
        from torchvision.transforms import ColorJitter, RandomResizedCrop

        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.erase_p = erase_p
        self.flow_norm_divisor = flow_norm_divisor
        self._RandomResizedCrop = RandomResizedCrop
        self._ColorJitter = ColorJitter
        if use_imagenet_norm:
            self._mean = [0.485, 0.456, 0.406]
            self._std = [0.229, 0.224, 0.225]
        else:
            self._mean = [0.5, 0.5, 0.5]
            self._std = [0.5, 0.5, 0.5]

    def __call__(self, frames):
        import torchvision.transforms.functional as TF

        # 1. RandomResizedCrop — same crop for all frames.
        i, j, h, w = self._RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        cropped = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]

        # 2. Farneback flow on grayscale of the cropped frames.
        gray_arrays = [np.array(f.convert("L"), dtype=np.uint8) for f in cropped]
        flow_clip = _farneback_clip(gray_arrays)  # (T-1, 2, H, W)
        flow_clip = flow_clip / self.flow_norm_divisor

        # 3. ColorJitter on RGB only.
        b_range = (max(0.0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
        c_range = (max(0.0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
        s_range = (max(0.0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
        h_range = (-self.hue, self.hue) if self.hue > 0 else None
        fn_idx, b, c, s, h_factor = self._ColorJitter.get_params(b_range, c_range, s_range, h_range)
        jittered = []
        for img in cropped:
            for fid in fn_idx:
                if fid == 0 and b is not None:
                    img = TF.adjust_brightness(img, b)
                elif fid == 1 and c is not None:
                    img = TF.adjust_contrast(img, c)
                elif fid == 2 and s is not None:
                    img = TF.adjust_saturation(img, s)
                elif fid == 3 and h_factor is not None:
                    img = TF.adjust_hue(img, h_factor)
            jittered.append(img)

        # 4. ToTensor + Normalize on RGB.
        rgb_tensors = [TF.normalize(TF.to_tensor(img), mean=self._mean, std=self._std) for img in jittered]
        rgb_clip = torch.stack(rgb_tensors, dim=0)  # (T, 3, H, W)

        # 5. Joint RandomErasing — same rectangle on RGB and flow.
        if self.erase_p > 0 and random.random() < self.erase_p:
            from torchvision.transforms import RandomErasing
            erase_i, erase_j, erase_h, erase_w, erase_v = RandomErasing.get_params(
                rgb_clip[0], scale=(0.02, 0.33), ratio=(0.3, 3.3), value=[0.0]
            )
            rgb_clip[:, :, erase_i : erase_i + erase_h, erase_j : erase_j + erase_w] = erase_v
            flow_clip[:, :, erase_i : erase_i + erase_h, erase_j : erase_j + erase_w] = 0.0

        return rgb_clip, flow_clip


class FlowAwareEvalTransform:
    """Two-stream eval/test transform: deterministic resize, no augmentation.

    Output:
      rgb_clip:  (T, 3, H, W) float, ImageNet-normalized
      flow_clip: (T-1, 2, H, W) float, divided by FLOW_NORM_DIVISOR
    """

    def __init__(
        self,
        image_size: int = 224,
        use_imagenet_norm: bool = True,
        flow_norm_divisor: float = FLOW_NORM_DIVISOR,
    ) -> None:
        self.image_size = image_size
        self.flow_norm_divisor = flow_norm_divisor
        if use_imagenet_norm:
            self._mean = [0.485, 0.456, 0.406]
            self._std = [0.229, 0.224, 0.225]
        else:
            self._mean = [0.5, 0.5, 0.5]
            self._std = [0.5, 0.5, 0.5]

    def __call__(self, frames):
        import torchvision.transforms.functional as TF

        resized = [TF.resize(f, [self.image_size, self.image_size]) for f in frames]
        gray_arrays = [np.array(f.convert("L"), dtype=np.uint8) for f in resized]
        flow_clip = _farneback_clip(gray_arrays) / self.flow_norm_divisor

        rgb_tensors = [TF.normalize(TF.to_tensor(img), mean=self._mean, std=self._std) for img in resized]
        rgb_clip = torch.stack(rgb_tensors, dim=0)
        return rgb_clip, flow_clip


def build_flow_clip_transforms(
    image_size: int = 256, is_training: bool = True, use_imagenet_norm: bool = True
) -> Any:
    """Two-stream clip transforms returning (rgb, flow) tuples."""
    if is_training:
        return FlowAwareClipAug(image_size=image_size, use_imagenet_norm=use_imagenet_norm)
    return FlowAwareEvalTransform(image_size=image_size, use_imagenet_norm=use_imagenet_norm)


class _FlowOnlyAdapter:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, frames):
        _, flow_clip = self.inner(frames)
        return flow_clip


def build_flow_only_clip_transforms(
    image_size: int = 256, is_training: bool = True, use_imagenet_norm: bool = True
) -> Any:
    """Flow-only clip transforms returning a single flow tensor (T-1, 2, H, W).

    Wraps the two-stream transforms and discards the RGB output. Farneback still
    runs on grayscale of the cropped frames; the RGB tensor is just dropped.
    """
    inner = build_flow_clip_transforms(
        image_size=image_size, is_training=is_training, use_imagenet_norm=use_imagenet_norm
    )
    return _FlowOnlyAdapter(inner)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Sample one Mixup λ per batch; return mixed inputs and (y_a, y_b, λ) for loss.

    Loss usage:  loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)
    """
    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, (y, y[perm], lam)


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """CutMix on a clip batch (B, T, C, H, W): same rectangle pasted across all frames.

    Returns the mixed clip and (y_a, y_b, lam). lam is the unmodified-area fraction.
    """
    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)

    h, w = x.shape[-2], x.shape[-1]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    if cut_h <= 0 or cut_w <= 0:
        return x, (y, y, 1.0)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    if y2 == y1 or x2 == x1:
        return x, (y, y, 1.0)

    x_mixed = x.clone()
    x_mixed[:, :, :, y1:y2, x1:x2] = x[perm][:, :, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
    return x_mixed, (y, y[perm], lam_adj)


def mixup_batch_two_stream(rgb: torch.Tensor, flow: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mixup on (rgb, flow) using the same lambda + permutation for both streams."""
    if alpha <= 0:
        return rgb, flow, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(rgb.size(0), device=rgb.device)
    rgb_mixed = lam * rgb + (1.0 - lam) * rgb[perm]
    flow_mixed = lam * flow + (1.0 - lam) * flow[perm]
    return rgb_mixed, flow_mixed, (y, y[perm], lam)


def cutmix_batch_two_stream(rgb: torch.Tensor, flow: torch.Tensor, y: torch.Tensor, alpha: float):
    """CutMix on (rgb, flow): same rectangle and same permutation pasted on both."""
    if alpha <= 0:
        return rgb, flow, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(rgb.size(0), device=rgb.device)
    h, w = rgb.shape[-2], rgb.shape[-1]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    if cut_h <= 0 or cut_w <= 0:
        return rgb, flow, (y, y, 1.0)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    if y2 == y1 or x2 == x1:
        return rgb, flow, (y, y, 1.0)
    rgb_mixed = rgb.clone()
    rgb_mixed[:, :, :, y1:y2, x1:x2] = rgb[perm][:, :, :, y1:y2, x1:x2]
    flow_mixed = flow.clone()
    flow_mixed[:, :, :, y1:y2, x1:x2] = flow[perm][:, :, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
    return rgb_mixed, flow_mixed, (y, y[perm], lam_adj)


def build_two_group_param_groups(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    backbone_prefix: str = "spatial.",
    no_decay_names: Tuple[str, ...] = (
        "bias",
        "cls_token",
        "pos_embed",
        "type_frame",
        "type_motion",
    ),
) -> List[dict]:
    """Two LR groups: backbone (spatial.*) at backbone_lr; everything else at head_lr.

    1-D params (norms, biases, type/pos/CLS embeddings) get weight_decay=0 in their
    respective groups.
    """
    backbone_decay, backbone_nodecay = [], []
    head_decay, head_nodecay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = pname.startswith(backbone_prefix)
        is_nodecay = p.ndim <= 1 or any(k in pname for k in no_decay_names)
        if is_backbone and is_nodecay:
            backbone_nodecay.append(p)
        elif is_backbone:
            backbone_decay.append(p)
        elif is_nodecay:
            head_nodecay.append(p)
        else:
            head_decay.append(p)

    groups: List[dict] = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay, "group": "backbone.decay"})
    if backbone_nodecay:
        groups.append({"params": backbone_nodecay, "lr": backbone_lr, "weight_decay": 0.0, "group": "backbone.no_decay"})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "group": "head.decay"})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0, "group": "head.no_decay"})
    return groups


class ModelEMA:
    """Exponential moving average of model parameters and buffers.

    Update at every optimizer step (or after warmup). Evaluate / save the .module
    instead of the live model — usually +0.5–2 pt on video classifiers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            if torch.is_floating_point(ema_p):
                ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                ema_p.copy_(p.detach())
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b.detach())


def init_wandb(cfg, default_run_name: Optional[str] = None) -> bool:
    """Initialize a W&B run from cfg.wandb.*; return True if active.

    Tolerates missing wandb dep / unreachable server: prints a warning and continues.
    """
    wb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wb_cfg.get("enabled", False)):
        return False
    try:
        import wandb
        from omegaconf import OmegaConf

        wandb.init(
            project=wb_cfg.get("project", "csc43m04ep"),
            entity=wb_cfg.get("entity"),
            mode=wb_cfg.get("mode", "online"),
            name=wb_cfg.get("run_name") or default_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(wb_cfg.get("tags", []) or []),
            resume="allow",
            id=wb_cfg.get("run_id"),
        )
        return True
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without logging.")
        return False


def wandb_log(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to the active W&B run; no-op if W&B isn't initialized."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def finish_wandb() -> None:
    """Close the active W&B run, if any."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


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
