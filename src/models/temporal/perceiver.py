"""Perceiver-style temporal head.

Consumes the full space-time token grid from a ViT backbone — not the
already-pooled per-tubelet vector. Input shape: ``(B, T', N, D)`` where:
  * ``T'`` = number of temporal slots exposed by the backbone (post-tubelet)
  * ``N``  = number of patch tokens per slot (e.g. ``H'·W' = 196`` for ViT-S/16
            at 224 px with patch 16 — VideoMAE has no CLS token, but a backbone
            that does just sets ``N = 1 + H'·W'``)
  * ``D``  = embed_dim

A small set of learnable queries (default 16) cross-attend to the flattened
``(B, T'·N, D)`` keys/values. The queries then run through a self-attention
tower of ``num_layers`` standard transformer encoder layers. Mean-pool over
the queries gives a fixed-size video vector for the classifier.

Why this rather than a relational head over per-frame CLS / mean-pool tokens:
- exposes per-position spatial-temporal patterns that the mean-pool averages
  out — on VideoMAE/ViT-S/16 + 4 frames + tubelet_time=2, that's 196×
  more discriminative signal than the (B, T'=2, D) the current heads see;
- compresses ``T'·N`` tokens to 16 learnable codes so downstream cost is modest;
- temporal positional embedding is learnable and zero-initialised so at training
  step 0 frames are interchangeable and the head behaves as a permute-equivariant
  pool over ``(frames, patches)`` — gain has to be earned by gradients.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


def _sinusoidal_temporal_pos(max_frames: int, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding over ``max_frames`` positions, ``dim`` features.

    Standard Vaswani et al. parameterisation: even indices = sin, odd = cos, of
    position scaled by a geometric series of wavelengths 1..10000. Returns a
    ``(max_frames, dim)`` float tensor.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pos embed requires even dim; got {dim}")
    pos = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1)
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
        # Run attention in fp32: fp16 softmax over ~392 KV tokens routinely
        # overflows on this dataset and turned the Jabiru run NaN at ep ~10.
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

        # Temporal positional embedding broadcast across patch axis (T' entries × D).
        #   "zero": learnable, zero-init. Step-0 permutation-symmetric across frames.
        #   "sinusoidal_fixed_with_scale": fixed sinusoidal pattern stored as a
        #       buffer, gated by a single learnable scalar (zero-init at first to
        #       leave step-0 forward unchanged, then learned). Order-aware from
        #       step 0 once the scale lifts off zero.
        if self.temporal_pos_init == "zero":
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_frames, 1, in_dim))
            self.register_buffer("_temporal_pos_buf", torch.zeros(0), persistent=False)
            self.temporal_pos_scale = None
        elif self.temporal_pos_init == "sinusoidal_fixed_with_scale":
            self.register_buffer(
                "_temporal_pos_buf",
                _sinusoidal_temporal_pos(self.max_frames, in_dim).view(1, self.max_frames, 1, in_dim),
                persistent=False,
            )
            self.temporal_pos_scale = nn.Parameter(torch.zeros(1))
            self.temporal_pos = None
        else:
            raise ValueError(
                f"temporal_pos_init must be 'zero' or 'sinusoidal_fixed_with_scale'; "
                f"got {self.temporal_pos_init!r}"
            )

        self.cross = _CrossAttentionBlock(
            dim=in_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )

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
        # features: (B, T', N, D) — full token grid per temporal slot.
        if features.dim() != 4:
            raise ValueError(
                f"PerceiverHead expects (B, T', N, D); got shape {tuple(features.shape)}"
            )
        B, T, N, D = features.shape
        if T > self.max_frames:
            raise ValueError(f"T'={T} exceeds max_frames={self.max_frames}")
        # add temporal pos embed (T' entries, broadcast over the N axis)
        if self.temporal_pos is not None:
            kv = features + self.temporal_pos[:, :T, :, :]
        else:
            kv = features + self.temporal_pos_scale * self._temporal_pos_buf[:, :T, :, :]
        kv = kv.reshape(B, T * N, D)              # (B, T'*N, D)

        q = self.queries.expand(B, -1, -1)        # (B, Q, D)
        q = self.cross(q, kv)                     # (B, Q, D)
        q = self.encoder(q)                       # (B, Q, D)
        q = self.norm(q)
        pooled = q.mean(dim=1)                    # (B, D)
        return self.proj(pooled)
