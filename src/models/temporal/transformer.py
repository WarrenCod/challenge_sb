"""Transformer encoder over the frame sequence.

Prepends a learnable CLS token, adds a learnable positional embedding, then
runs ``num_layers`` of pre-norm transformer blocks with optional stochastic
depth (DropPath) on attention and MLP residual branches. The CLS token's final
embedding becomes the video vector (optionally projected to ``out_dim``).

Implementing the block by hand (rather than nn.TransformerEncoderLayer) lets us
inject a per-layer DropPath rate on each residual — the standard ViT/MAE recipe
for regularising deep transformers, valuable since SSv2-style training is
extremely overfitting-prone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob) / keep_prob
    return x * mask


class _PreNormBlock(nn.Module):
    """Pre-norm transformer block with DropPath on each residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + _drop_path(attn_out, self.drop_path, self.training)
        h = self.norm2(x)
        x = x + _drop_path(self.mlp(h), self.drop_path, self.training)
        return x


class TransformerTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        max_len: int = 64,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim

        if dim_feedforward is None:
            dim_feedforward = in_dim * 4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, in_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Linear DropPath schedule from 0 (block 0) to drop_path (last block).
        rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                _PreNormBlock(
                    dim=in_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    drop_path=rates[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, self.out_dim) if self.out_dim != in_dim else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = features.shape
        if num_frames + 1 > self.pos_embed.shape[1]:
            raise ValueError(
                f"T+1={num_frames + 1} exceeds max_len+1={self.pos_embed.shape[1]}"
            )
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, features], dim=1)
        tokens = tokens + self.pos_embed[:, : num_frames + 1]
        for block in self.blocks:
            tokens = block(tokens)
        return self.proj(self.norm(tokens[:, 0]))
