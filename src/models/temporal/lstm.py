"""LSTM over the frame-feature sequence; returns the last timestep's hidden state."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class LSTMTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(features)
        return output[:, -1, :]
