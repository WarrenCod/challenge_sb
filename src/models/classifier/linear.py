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
        # Zero-init: first forward outputs uniform logits → CE loss = log(num_classes)
        # cleanly. Avoids spurious early gradients from random classifier weights
        # that would propagate noise back into the (pretrained) encoder.
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.fc(vec)
