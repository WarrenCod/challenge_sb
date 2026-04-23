"""Vision Transformer spatial encoder.

Torchvision ViT prepends a CLS token internally and returns only that token's
embedding after the final block, so replacing `heads` with Identity gives us
a single (B*T, hidden_dim) vector per frame — matching the (B, T, d) contract.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer

from models.base import SpatialEncoder


def _torchvision_variant(factory, weights_enum) -> Callable[[bool], nn.Module]:
    def build(pretrained: bool) -> nn.Module:
        return factory(weights=weights_enum if pretrained else None)

    return build


def _custom_variant(**vit_kwargs) -> Callable[[bool], nn.Module]:
    def build(pretrained: bool) -> nn.Module:
        if pretrained:
            raise ValueError(
                "pretrained=True is not supported for custom ViT variants "
                "(torchvision ships no weights for this size)."
            )
        return VisionTransformer(image_size=224, **vit_kwargs)

    return build


# Each entry: variant name -> builder(pretrained) -> nn.Module with `.hidden_dim` + `.heads`.
_VIT_FACTORIES = {
    "vit_ti_16": _custom_variant(
        patch_size=16, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768
    ),
    "vit_s_16": _custom_variant(
        patch_size=16, num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536
    ),
    "vit_b_16": _torchvision_variant(models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": _torchvision_variant(models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": _torchvision_variant(models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1),
}


class ViTEncoder(SpatialEncoder):
    def __init__(self, variant: str = "vit_b_16", pretrained: bool = False) -> None:
        super().__init__()
        if variant not in _VIT_FACTORIES:
            raise ValueError(
                f"Unknown vit variant: {variant}. Available: {list(_VIT_FACTORIES)}"
            )
        backbone = _VIT_FACTORIES[variant](pretrained)
        self.out_dim = backbone.hidden_dim
        backbone.heads = nn.Identity()
        self.backbone = backbone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch_size * num_frames, channels, height, width)
        feats = self.backbone(frames)
        return feats.view(batch_size, num_frames, self.out_dim)
