"""Abstract component interfaces for the modular video model.

Three slots, chained sequentially inside ModularVideoModel:

    SpatialEncoder      (B, T, 3, H, W) -> (B, T, d)     # shared per-frame encoder
    TemporalProcessor   (B, T, d)       -> (B, d')       # mixes across time
    Classifier          (B, d')         -> (B, K)        # final logits

Each component exposes integer dim attributes so the composer can wire shapes
automatically (d, d') without the user re-declaring them in two places.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """Per-frame encoder. Must expose `out_dim` (= d)."""

    out_dim: int

    def forward(self, video: torch.Tensor) -> torch.Tensor:  # (B, T, 3, H, W) -> (B, T, d)
        raise NotImplementedError


class TemporalProcessor(nn.Module):
    """Collapses the time axis to a single vector. Exposes `in_dim`, `out_dim`."""

    in_dim: int
    out_dim: int

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, d')
        raise NotImplementedError


class Classifier(nn.Module):
    """Final head. Exposes `in_dim`, `num_classes`."""

    in_dim: int
    num_classes: int

    def forward(self, vec: torch.Tensor) -> torch.Tensor:  # (B, d') -> (B, K)
        raise NotImplementedError
