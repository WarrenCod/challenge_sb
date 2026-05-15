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


class VJEPAClipAugMinimal:
    """V-JEPA 1 minimal pretraining transform: clip-consistent RandomResizedCrop
    + ImageNet normalize. No flip (SSv2 direction-sensitive), no color jitter,
    no grayscale, no Gaussian blur. Matches the V-JEPA 1 paper's pretraining
    pipeline (modulo the disabled flip).

    Stripping color/blur is intentional: V-JEPA's masked-feature prediction
    objective already pushes the encoder toward locally-informative features,
    and heavy invariance augmentations (SimCLR/DINO style) tend to suppress
    exactly the appearance cues the predictor would otherwise learn to use.
    """

    def __init__(
        self,
        image_size: int = 224,
        scale: Tuple[float, float] = (0.3, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.mean = normalize_mean
        self.std = normalize_std

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        frames = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]
        tensors = [TF.normalize(TF.to_tensor(f), mean=self.mean, std=self.std) for f in frames]
        return torch.stack(tensors, dim=0)


def build_vjepa_clip_transform_minimal(image_size: int = 224) -> VJEPAClipAugMinimal:
    return VJEPAClipAugMinimal(image_size=image_size)


class VJEPAClipAugMild:
    """V-JEPA "mild" pretraining transform: clip-consistent RandomResizedCrop
    + clip-consistent ColorJitter + ImageNet normalize. No flip / blur /
    grayscale.

    Why this and not Minimal: V-JEPA 1's "minimal" recipe relies on horizontal
    flip for input diversity, but SSv2 labels are direction-sensitive so flip
    is forbidden. ColorJitter restores an equivalent amount of input
    perturbation without disturbing motion signal. Blur and grayscale are
    intentionally omitted — they suppress the appearance cues V-JEPA's
    masked-feature predictor wants to use.
    """

    def __init__(
        self,
        image_size: int = 224,
        scale: Tuple[float, float] = (0.3, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
        color_jitter_p: float = 0.8,
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
        self.mean = normalize_mean
        self.std = normalize_std

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=self.scale, ratio=self.ratio)
        frames = [TF.resized_crop(f, i, j, h, w, [self.image_size, self.image_size]) for f in frames]

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

        tensors = [TF.normalize(TF.to_tensor(f), mean=self.mean, std=self.std) for f in frames]
        return torch.stack(tensors, dim=0)


def build_vjepa_clip_transform_mild(image_size: int = 224) -> VJEPAClipAugMild:
    return VJEPAClipAugMild(image_size=image_size)
