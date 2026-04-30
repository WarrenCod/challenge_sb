"""Transformer encoder over the frame sequence.

Prepends a learnable CLS token, adds a learnable positional embedding, then
runs `num_layers` of pre-norm TransformerEncoderLayer. The CLS token's final
embedding becomes the video vector (optionally projected to `out_dim`).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class TransformerTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 64,
        out_dim: Optional[int] = None,
        in_proj_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        # Optional bottleneck: project per-frame features down before the encoder.
        # Lets us run an attention head at d=512 on top of a d=2048 backbone (R50)
        # without ballooning head params from O(d²·layers) at d=2048.
        if in_proj_dim is not None and in_proj_dim != in_dim:
            self.in_proj: nn.Module = nn.Linear(in_dim, in_proj_dim)
            token_dim = in_proj_dim
        else:
            self.in_proj = nn.Identity()
            token_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else token_dim

        if dim_feedforward is None:
            dim_feedforward = token_dim * 4

        # Zero-init cls_token + pos_embed: at step 0 the temporal transformer
        # behaves order-blind (no positional signal) and the prepended CLS is a
        # neutral query. The temporal head degenerates to a learned mean-pool
        # initially and gradually grows order-awareness as gradients populate
        # pos_embed. Avoids the divergence we saw at lr=1e-3 with trunc_normal.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, token_dim))

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
        self.proj = nn.Linear(token_dim, self.out_dim) if self.out_dim != token_dim else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = features.shape
        if num_frames + 1 > self.pos_embed.shape[1]:
            raise ValueError(
                f"T+1={num_frames + 1} exceeds max_len+1={self.pos_embed.shape[1]}"
            )
        features = self.in_proj(features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, features], dim=1)
        tokens = tokens + self.pos_embed[:, : num_frames + 1]
        tokens = self.encoder(tokens)
        return self.proj(self.norm(tokens[:, 0]))
