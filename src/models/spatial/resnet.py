"""ResNet18 / 34 / 50 spatial encoder: shared per-frame CNN -> one vector per frame."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.base import SpatialEncoder


_RESNET_FACTORIES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class ResNetEncoder(SpatialEncoder):
    def __init__(self, variant: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        if variant not in _RESNET_FACTORIES:
            raise ValueError(
                f"Unknown resnet variant: {variant}. Available: {list(_RESNET_FACTORIES)}"
            )
        factory, weights_enum = _RESNET_FACTORIES[variant]
        backbone = factory(weights=weights_enum if pretrained else None)
        self.out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch_size * num_frames, channels, height, width)
        feats = self.backbone(frames)
        return feats.view(batch_size, num_frames, self.out_dim)
