"""Single linear layer: d' -> num_classes."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import Classifier


class LinearClassifier(Classifier):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.fc(vec)
