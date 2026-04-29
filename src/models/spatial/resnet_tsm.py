"""ResNet18/50 with Temporal Shift Modules (Lin et al., 2019).

Per-frame ResNet, augmented with a zero-FLOP temporal-shift op inserted before
the first conv of every BasicBlock / Bottleneck. This lets a 2D CNN do temporal
mixing implicitly: a small fraction of channels are shifted forward (gives
frame t access to frame t+1) and another fraction are shifted backward.

Output: (B, T, C, H, W) -> (B, T, out_dim) — temporal pooling happens in the
TemporalProcessor slot of the modular model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.base import SpatialEncoder


class TemporalShift(nn.Module):
    """Shift 1/fold_div channels forward in time and 1/fold_div backward.

    Input: (B*T, C, H, W). Reshapes to (B, T, C, H, W), shifts, reshapes back.
    Boundary frames see zero-padded slots (frame T-1 has no t+1 to read from).
    """

    def __init__(self, num_segments: int, fold_div: int = 8) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        t = self.num_segments
        if bt % t != 0:
            raise RuntimeError(
                f"TemporalShift expected (B*T, C, H, W) with B*T divisible by T={t}; got {bt}."
            )
        b = bt // t
        x = x.view(b, t, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        # Channels [0, fold): each frame reads from t+1 (look-ahead).
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Channels [fold, 2*fold): each frame reads from t-1 (look-back).
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        # Remaining channels unchanged.
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return out.view(bt, c, h, w)


_RESNET_FACTORIES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
}


class ResNetTSMEncoder(SpatialEncoder):
    """ResNet backbone with a TemporalShift inserted before each block's first conv."""

    def __init__(
        self,
        variant: str = "resnet18",
        num_segments: int = 4,
        fold_div: int = 8,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if variant not in _RESNET_FACTORIES:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(_RESNET_FACTORIES)}"
            )
        factory, weights_enum = _RESNET_FACTORIES[variant]
        backbone = factory(weights=weights_enum if pretrained else None)
        self.out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
            for block in layer:
                shift = TemporalShift(num_segments=num_segments, fold_div=fold_div)
                block.conv1 = nn.Sequential(shift, block.conv1)

        self.backbone = backbone
        self.num_segments = num_segments

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        if t != self.num_segments:
            raise RuntimeError(
                f"Expected T={self.num_segments} frames per clip (TSM is wired for that), got T={t}."
            )
        frames = video.reshape(b * t, c, h, w)
        feats = self.backbone(frames)
        return feats.view(b, t, self.out_dim)
