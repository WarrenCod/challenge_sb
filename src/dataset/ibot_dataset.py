"""
iBOT-style multi-crop augmentation + masking, applied per frame.

Each __getitem__ returns:
  crops: list of (3, H, W) tensors. First 2 are global (224×224), rest are local
         (96×96). All are independently augmented (DINO multi-crop).
  masks: list of (N,) bool tensors (None for locals); True = patch is masked.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T


class _GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class _Solarize:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.solarize(img) if random.random() < self.p else img


class MultiCropAug:
    """DINO multi-crop pipeline. NO horizontal flip (SSv2 directionality matters)."""

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        n_local: int = 4,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.4),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        normalize = T.Compose([T.ToTensor(), T.Normalize(mean=normalize_mean, std=normalize_std)])

        color_jitter = T.Compose(
            [
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )

        # Global crop 1: blur with p=1, no solarize.
        self.global_aug_1 = T.Compose(
            [
                T.RandomResizedCrop(global_size, scale=global_scale),
                color_jitter,
                _GaussianBlur(p=1.0),
                normalize,
            ]
        )
        # Global crop 2: blur p=0.1 + solarize p=0.2 (DINO recipe).
        self.global_aug_2 = T.Compose(
            [
                T.RandomResizedCrop(global_size, scale=global_scale),
                color_jitter,
                _GaussianBlur(p=0.1),
                _Solarize(p=0.2),
                normalize,
            ]
        )
        # Local crops: lower scale, blur p=0.5.
        self.local_aug = T.Compose(
            [
                T.RandomResizedCrop(local_size, scale=local_scale),
                color_jitter,
                _GaussianBlur(p=0.5),
                normalize,
            ]
        )
        self.n_local = n_local

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        crops = [self.global_aug_1(img), self.global_aug_2(img)]
        for _ in range(self.n_local):
            crops.append(self.local_aug(img))
        return crops


def block_wise_mask(
    grid_size: int,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    max_attempts: int = 30,
) -> torch.Tensor:
    """Return a (grid_size**2,) bool mask, True = masked. Block-wise per BEiT/iBOT."""
    n_patches = grid_size * grid_size
    target = int(np.random.uniform(mask_ratio_min, mask_ratio_max) * n_patches)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    masked = 0
    attempts = 0
    while masked < target and attempts < max_attempts * target:
        attempts += 1
        block_size = np.random.randint(low=4, high=max(5, n_patches // 4))
        aspect = np.exp(np.random.uniform(np.log(0.3), np.log(3.3)))
        h = max(1, int(round(math.sqrt(block_size * aspect))))
        w = max(1, int(round(math.sqrt(block_size / aspect))))
        if h > grid_size or w > grid_size:
            continue
        top = np.random.randint(0, grid_size - h + 1)
        left = np.random.randint(0, grid_size - w + 1)
        new_mask = mask.copy()
        new_mask[top : top + h, left : left + w] = True
        delta = int(new_mask.sum() - mask.sum())
        if delta == 0:
            continue
        mask = new_mask
        masked = int(mask.sum())
    return torch.from_numpy(mask.flatten().astype(bool))


class iBOTFrameDataset(Dataset):
    """Per-frame dataset producing iBOT multi-crop tensors + per-global masks."""

    def __init__(
        self,
        frame_paths: List[Path],
        aug: MultiCropAug,
        global_grid_size: int,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
    ) -> None:
        self.frame_paths = frame_paths
        self.aug = aug
        self.grid_size = global_grid_size
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int):
        path = self.frame_paths[index]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        crops = self.aug(rgb)
        # One mask per global crop (the first 2).
        masks = [
            block_wise_mask(self.grid_size, self.mask_ratio_min, self.mask_ratio_max),
            block_wise_mask(self.grid_size, self.mask_ratio_min, self.mask_ratio_max),
        ]
        return crops, masks


def ibot_collate(batch):
    """Collate iBOTFrameDataset items into (list-of-crop-tensors, list-of-mask-tensors)."""
    n_crops = len(batch[0][0])
    crops = [torch.stack([item[0][i] for item in batch], dim=0) for i in range(n_crops)]
    masks = [torch.stack([item[1][i] for item in batch], dim=0) for i in range(2)]
    return crops, masks
