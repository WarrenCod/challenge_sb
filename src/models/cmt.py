"""Channel-Motion Transformer (CMT).

Per-frame ResNet-18 trunk produces (B, T, 512, 7, 7). A 1x1 conv compresses to
C' "motion channels"; each channel's (T, H, W) activation map is treated as a
mini-video and encoded by a shared (2+1)D ResNet-3D-lite into a vector d. The
C' motion vectors are mixed by a small Transformer and pooled by PMA
(Set Transformer) into one video-level vector that feeds an MLP head.

Input:  (B, T, 3, H, W)
Output: (B, num_classes)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class STBlock(nn.Module):
    """(2+1)D residual block: spatial 1x3x3 + temporal 3x1x1 convs with GN/GELU."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv3d(
            c_in, c_out, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn1 = _gn(c_out)
        self.temporal_conv = nn.Conv3d(
            c_out, c_out, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.gn2 = _gn(c_out)
        self.act = nn.GELU()
        self.shortcut: nn.Module = (
            nn.Identity()
            if c_in == c_out
            else nn.Conv3d(c_in, c_out, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.spatial_conv(x)))
        h = self.gn2(self.temporal_conv(h))
        return self.act(h + self.shortcut(x))


class PerChannelMotionNet(nn.Module):
    """Shared-weight sub-net mapping (N, 1, T, H, W) -> (N, d)."""

    def __init__(self, d: int = 512, widths: Sequence[int] = (96, 192, 384)) -> None:
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv3d(1, w1, kernel_size=3, padding=1, bias=False),
            _gn(w1),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(STBlock(w1, w1), STBlock(w1, w1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.stage2 = nn.Sequential(STBlock(w1, w2), STBlock(w2, w2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.stage3 = nn.Sequential(STBlock(w2, w3), STBlock(w3, w3))
        self.project = nn.Linear(w3, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = x.mean(dim=(2, 3, 4))
        return self.project(x)


class PMA(nn.Module):
    """Pooling by Multihead Attention (Set Transformer)."""

    def __init__(self, d: int, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.seed, std=0.02)
        self.attn = nn.MultiheadAttention(d, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Linear(d * ffn_mult, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        query = self.seed.expand(batch_size, -1, -1)
        out, _ = self.attn(query, tokens, tokens, need_weights=False)
        out = self.norm1(out + query)
        out = self.norm2(out + self.ffn(out))
        return out.squeeze(1)


class ResNet18Trunk(nn.Module):
    """ResNet-18 truncated before avgpool/fc. Output (N, 512, H/32, W/32)."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.out_channels = 512

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.features(frames)


class CMT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        c_prime: int = 128,
        motion_widths: Sequence[int] = (96, 192, 384),
        d: int = 512,
        set_num_blocks: int = 4,
        set_num_heads: int = 8,
        set_ffn_mult: int = 4,
        head_hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.c_prime = c_prime

        self.backbone = ResNet18Trunk(pretrained=pretrained)
        self.bottleneck = nn.Conv2d(self.backbone.out_channels, c_prime, kernel_size=1)

        self.temporal_pos = nn.Parameter(torch.zeros(1, 1, num_frames, 1, 1))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.motion_net = PerChannelMotionNet(d=d, widths=motion_widths)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=set_num_heads,
            dim_feedforward=d * set_ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=set_num_blocks)
        self.set_pool = PMA(d, num_heads=set_num_heads, ffn_mult=set_ffn_mult)

        self.head = nn.Sequential(
            nn.Linear(d, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, T, channels, height, width = video.shape
        if T != self.num_frames:
            raise ValueError(
                f"CMT expected T={self.num_frames} frames but got {T}"
            )

        frames = video.reshape(batch_size * T, channels, height, width)
        feat = self.backbone(frames)
        _, _, h_out, w_out = feat.shape
        feat = self.bottleneck(feat)
        feat = feat.view(batch_size, T, self.c_prime, h_out, w_out)
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        feat = feat + self.temporal_pos

        per_ch = feat.reshape(batch_size * self.c_prime, 1, T, h_out, w_out)
        motion = self.motion_net(per_ch)
        motion = motion.view(batch_size, self.c_prime, -1)

        tokens = self.set_encoder(motion)
        pooled = self.set_pool(tokens)
        return self.head(pooled)
