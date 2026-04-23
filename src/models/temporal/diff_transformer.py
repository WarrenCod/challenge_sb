"""Transformer over [frame, frame-diff] tokens.

Each time step t carries the per-frame feature x[t] concatenated with the
forward difference d[t] = x[t+1] - x[t]. The final position has no successor,
so d[T-1] is a learnable sentinel. A CLS token is prepended; a learnable
positional embedding is added; the CLS output is the video vector.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class DiffTransformerTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 64,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        token_dim = 2 * in_dim
        self.out_dim = out_dim if out_dim is not None else token_dim

        if dim_feedforward is None:
            dim_feedforward = token_dim * 4

        self.end_diff = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, token_dim))
        nn.init.trunc_normal_(self.end_diff, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.proj = (
            nn.Linear(token_dim, self.out_dim) if self.out_dim != token_dim else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = features.shape
        if num_frames + 1 > self.pos_embed.shape[1]:
            raise ValueError(
                f"T+1={num_frames + 1} exceeds max_len+1={self.pos_embed.shape[1]}"
            )

        diffs = features[:, 1:] - features[:, :-1]  # (B, T-1, d)
        end = self.end_diff.expand(batch_size, -1, -1)  # (B, 1, d)
        diffs = torch.cat([diffs, end], dim=1)  # (B, T, d)
        tokens = torch.cat([features, diffs], dim=-1)  # (B, T, 2d)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : num_frames + 1]
        tokens = self.encoder(tokens)
        return self.proj(self.norm(tokens[:, 0]))
