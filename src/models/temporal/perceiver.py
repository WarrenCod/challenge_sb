"""Perceiver-style temporal head.

Consumes the full per-frame token sequence from a ViT backbone — *not* just the
CLS token. Inputs shape: (B, T, N+1, D), where N+1 = 1 CLS + N patch tokens.

A small set of learnable queries (default 16) cross-attend to the flattened
(T*(N+1)) keys/values. The queries then run through a self-attention tower of
``num_layers`` standard transformer encoder layers. Mean-pool over the queries
gives a fixed-size video vector for the classifier.

Why this rather than a relational head over per-frame CLS tokens:
- exposes per-position spatial-temporal patterns that the CLS token alone
  averages out;
- compresses the resulting (T*(N+1) ≈ 788 for T=4, N=196) tokens to a
  small learnable code (16 queries) so downstream cost is modest;
- temporal positional embedding is learnable and zero-initialized so that
  at training step 0 frames are interchangeable and the head behaves like a
  standard "permute-equivariant pool over (frames, patches)".
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


def _sinusoidal_temporal_pos(max_frames: int, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding over ``max_frames`` positions, ``dim`` features.

    Standard Vaswani et al. parameterization: even indices use ``sin`` and odd
    indices use ``cos`` of position scaled by a geometric series of wavelengths
    from 1 to 10000. Returns a (max_frames, dim) float tensor.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pos embed requires even dim; got {dim}")
    pos = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1)             # (T, 1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_frames, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class _CrossAttentionBlock(nn.Module):
    """One pre-norm cross-attention block: Q attends to KV, then FFN on Q."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Q, D); kv: (B, K, D)
        # Run attention in fp32: fp16 softmax over T*(N+1)=788 KV tokens
        # routinely overflows on this dataset and turned the run NaN at ep 10.
        h = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            h32 = h.float()
            kv32 = kv_n.float()
            a, _ = self.attn(h32, kv32, kv32, need_weights=False)
        a = a.to(q.dtype)
        q = q + a
        q = q + self.mlp(self.norm_mlp(q))
        return q


class PerceiverHead(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_queries: int = 16,
        num_heads: int = 6,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 4,
        out_dim: Optional[int] = None,
        temporal_pos_init: str = "zero",
        num_cross_attn: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.max_frames = int(max_frames)
        self.num_queries = int(num_queries)
        self.temporal_pos_init = str(temporal_pos_init)

        # Learnable queries; std=0.02 init like ViT CLS.
        self.queries = nn.Parameter(torch.zeros(1, self.num_queries, in_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Temporal positional embedding broadcast across patch axis (T entries × D).
        #   "zero"                         → learnable, zero-init; step-0 is
        #                                    permutation-symmetric across frames.
        #                                    Used by exp2k..exp2n; default for
        #                                    backwards-compatible teacher loading.
        #   "sinusoidal_fixed_with_scale"  → fixed sinusoidal pattern + a single
        #                                    learnable scalar. Order-aware at
        #                                    step 0 (exp3a). Stores the pattern
        #                                    as a buffer to keep parameter
        #                                    counts unchanged.
        if self.temporal_pos_init == "zero":
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_frames, 1, in_dim))
        elif self.temporal_pos_init == "sinusoidal_fixed_with_scale":
            pe = _sinusoidal_temporal_pos(self.max_frames, in_dim)           # (T, D)
            self.register_buffer("temporal_pos_buf", pe.view(1, self.max_frames, 1, in_dim))
            self.temporal_pos_scale = nn.Parameter(torch.ones(1))
        else:
            raise ValueError(
                f"unknown temporal_pos_init {self.temporal_pos_init!r}; "
                f"expected 'zero' or 'sinusoidal_fixed_with_scale'"
            )

        # Cross-attn stack. `self.cross` is always the first block (preserves
        # state_dict key compatibility with exp2*/exp3a checkpoints). When
        # num_cross_attn>1, additional blocks go in `self.cross_extra` and are
        # zero-init on attention out_proj AND MLP final Linear, so the step-0
        # forward is bit-identical to num_cross_attn=1.
        if int(num_cross_attn) < 1:
            raise ValueError(f"num_cross_attn must be ≥ 1; got {num_cross_attn}")
        self.num_cross_attn = int(num_cross_attn)
        self.cross = _CrossAttentionBlock(
            dim=in_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        if self.num_cross_attn > 1:
            self.cross_extra = nn.ModuleList([
                _CrossAttentionBlock(
                    dim=in_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(self.num_cross_attn - 1)
            ])
            for blk in self.cross_extra:
                nn.init.zeros_(blk.attn.out_proj.weight)
                nn.init.zeros_(blk.attn.out_proj.bias)
                nn.init.zeros_(blk.mlp[-2].weight)
                nn.init.zeros_(blk.mlp[-2].bias)
        else:
            self.cross_extra = nn.ModuleList()

        # Self-attention tower on the queries.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=int(in_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = (
            nn.Linear(in_dim, self.out_dim)
            if self.out_dim != in_dim
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, N+1, D) — full token sequence per frame.
        if features.dim() != 4:
            raise ValueError(
                f"PerceiverHead expects (B, T, N+1, D); got shape {tuple(features.shape)}"
            )
        B, T, N1, D = features.shape
        if T > self.max_frames:
            raise ValueError(
                f"T={T} exceeds max_frames={self.max_frames}"
            )
        # add temporal pos embed (T entries, broadcast over the N+1 axis)
        if self.temporal_pos_init == "sinusoidal_fixed_with_scale":
            pos = self.temporal_pos_scale * self.temporal_pos_buf[:, :T, :, :]
        else:
            pos = self.temporal_pos[:, :T, :, :]
        kv = features + pos
        kv = kv.reshape(B, T * N1, D)            # (B, T*(N+1), D)

        q = self.queries.expand(B, -1, -1)       # (B, Q, D)
        q = self.cross(q, kv)                    # (B, Q, D)
        for blk in self.cross_extra:
            q = blk(q, kv)                       # (B, Q, D)
        q = self.encoder(q)                      # (B, Q, D)
        q = self.norm(q)
        pooled = q.mean(dim=1)                   # (B, D)
        return self.proj(pooled)
