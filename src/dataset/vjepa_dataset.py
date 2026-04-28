"""
V-JEPA clip-consistent augmentation.

Same RandomResizedCrop coords, same ColorJitter factors, same blur kernel, and
same grayscale decision are applied to all T frames of a clip. The only
variation across frames within a clip is the actual temporal change in the
underlying video — which is exactly the signal V-JEPA wants to model.

No horizontal flip and no vertical flip (SSv2 labels are direction-sensitive).

Returns a (T, C, H, W) tensor when called with a list of T PIL frames; plugs
into VideoFrameDataset(clip_transform=...).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class VJEPAClipAug:
    def __init__(
        self,
        image_size: int = 224,
        scale: Tuple[float, float] = (0.6, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_p: float = 0.8,
        grayscale_p: float = 0.2,
        blur_p: float = 0.5,
        blur_radius_min: float = 0.1,
        blur_radius_max: float = 2.0,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter_p = color_jitter_p
        self.grayscale_p = grayscale_p
        self.blur_p = blur_p
        self.blur_radius_min = blur_radius_min
        self.blur_radius_max = blur_radius_max
        self.mean = normalize_mean
        self.std = normalize_std

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        # 1. RandomResizedCrop — single (i, j, h, w) for the whole clip.
        i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        frames = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]

        # 2. ColorJitter — single set of factors, applied to every frame.
        if random.random() < self.color_jitter_p:
            b_range = (max(0.0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
            c_range = (max(0.0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
            s_range = (max(0.0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
            h_range = (-self.hue, self.hue) if self.hue > 0 else None
            fn_idx, b, c, s, h_factor = T.ColorJitter.get_params(b_range, c_range, s_range, h_range)
            adjusted = []
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
                adjusted.append(img)
            frames = adjusted

        # 3. Grayscale — single decision per clip.
        if random.random() < self.grayscale_p:
            frames = [TF.rgb_to_grayscale(f, num_output_channels=3) for f in frames]

        # 4. Gaussian blur — single radius per clip.
        if random.random() < self.blur_p:
            radius = random.uniform(self.blur_radius_min, self.blur_radius_max)
            frames = [f.filter(ImageFilter.GaussianBlur(radius=radius)) for f in frames]

        # 5. ToTensor + Normalize.
        tensors = [TF.normalize(TF.to_tensor(f), mean=self.mean, std=self.std) for f in frames]
        return torch.stack(tensors, dim=0)                    # (T, C, H, W)


def build_vjepa_clip_transform(image_size: int = 224) -> VJEPAClipAug:
    return VJEPAClipAug(image_size=image_size)
