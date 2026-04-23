"""No-op temporal mixing: average the per-frame features. Equivalent to cnn_baseline."""

from __future__ import annotations

import torch

from models.base import TemporalProcessor


class MeanPoolTemporal(TemporalProcessor):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1)
