"""Two-layer MLP with GELU + dropout."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import Classifier


class MLPClassifier(Classifier):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.net(vec)
