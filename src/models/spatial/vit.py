"""Vision Transformer spatial encoder.

Torchvision ViT prepends a CLS token internally and returns only that token's
embedding after the final block, so replacing `heads` with Identity gives us
a single (B*T, hidden_dim) vector per frame — matching the (B, T, d) contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.base import SpatialEncoder


_VIT_FACTORIES = {
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1),
}


class ViTEncoder(SpatialEncoder):
    def __init__(self, variant: str = "vit_b_16", pretrained: bool = False) -> None:
        super().__init__()
        if variant not in _VIT_FACTORIES:
            raise ValueError(
                f"Unknown vit variant: {variant}. Available: {list(_VIT_FACTORIES)}"
            )
        factory, weights_enum = _VIT_FACTORIES[variant]
        backbone = factory(weights=weights_enum if pretrained else None)
        self.out_dim = backbone.hidden_dim
        backbone.heads = nn.Identity()
        self.backbone = backbone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch_size * num_frames, channels, height, width)
        feats = self.backbone(frames)
        return feats.view(batch_size, num_frames, self.out_dim)
