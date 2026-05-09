"""Divided space-time transformer over CNN feature tokens.

Consumes the pre-pool feature map of a per-frame backbone — shape
(B, T, H, W, d_in) — instead of the global-pooled (B, T, d_in) vector. This
exposes the temporal head to 2D spatial structure at every frame, which matters
on motion-heavy SSv2-style classes (T=4 here is too small for a frame-level
transformer to extract meaningful order signal on its own).

Per layer (TimeSformer-style divided attention, Bertasius et al., 2021):
  1. Temporal MHSA along T at each spatial location (H*W independent T-tokens).
  2. Spatial  MHSA within each frame's H*W grid (T independent S-tokens).
  3. Position-wise MLP.
Cost ≈ 2·O(T·S²·d) instead of O((T·S)²·d) for full joint attention.

No CLS token: the readout is a mean over all T·H·W output tokens (after the
final LayerNorm). Simpler and empirically equivalent on this problem size.

Position embeddings are factorized: one (H*W, d) spatial table broadcast over T,
plus one (T, d) temporal table broadcast over H*W. Zero-init: the temporal head
starts as a learned mean-pool and grows order-awareness during training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class _DropPath(nn.Module):
    """Stochastic depth: drop the residual branch with prob `drop_prob`."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample mask broadcast across all non-batch dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class _DividedSpaceTimeBlock(nn.Module):
    def __init__(
        self,
        d: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(d)
        self.attn_t = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(d)
        self.attn_s = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d),
            nn.Dropout(dropout),
        )
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: torch.Tensor, t: int, s: int) -> torch.Tensor:
        # x: (B, T*S, d).
        b, ts, d = x.shape
        assert ts == t * s

        # 1. Temporal attention along T at each spatial location.
        h = self.norm_t(x).view(b, t, s, d).transpose(1, 2).reshape(b * s, t, d)
        h_out, _ = self.attn_t(h, h, h, need_weights=False)
        h_out = h_out.reshape(b, s, t, d).transpose(1, 2).reshape(b, t * s, d)
        x = x + self.drop_path(h_out)

        # 2. Spatial attention within each frame's H*W grid.
        h = self.norm_s(x).reshape(b * t, s, d)
        h_out, _ = self.attn_s(h, h, h, need_weights=False)
        h_out = h_out.reshape(b, t * s, d)
        x = x + self.drop_path(h_out)

        # 3. Position-wise FFN.
        x = x + self.drop_path(self.ffn(self.norm_ff(x)))
        return x


class SpaceTimeTransformer(TemporalProcessor):
    """Divided space-time transformer over (B, T, H, W, d_in) tokens."""

    # Tells ModularVideoModel to call spatial.forward_tokens instead of forward.
    wants_tokens: bool = True

    def __init__(
        self,
        in_dim: int,
        in_proj_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.15,
        drop_path: float = 0.1,
        max_t: int = 16,
        max_s: int = 256,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        token_dim = in_proj_dim
        self.out_dim = out_dim if out_dim is not None else token_dim
        if dim_feedforward is None:
            dim_feedforward = token_dim * 4

        self.in_proj = nn.Linear(in_dim, token_dim) if in_proj_dim != in_dim else nn.Identity()

        # Factorized pos-embed: spatial (max_s, d) + temporal (max_t, d), zero-init.
        # At step 0 the head is order-blind (degenerates to mean-pool); gradients
        # populate the embeddings during training.
        self.pos_spatial = nn.Parameter(torch.zeros(1, 1, max_s, token_dim))
        self.pos_temporal = nn.Parameter(torch.zeros(1, max_t, 1, token_dim))
        self.max_s = max_s
        self.max_t = max_t

        # Linear stochastic-depth schedule 0 → drop_path across layers.
        dp_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                _DividedSpaceTimeBlock(
                    d=token_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    drop_path=dp_rates[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(token_dim)
        self.proj = (
            nn.Linear(token_dim, self.out_dim) if self.out_dim != token_dim else nn.Identity()
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, H, W, d_in)
        if tokens.ndim != 5:
            raise ValueError(
                f"SpaceTimeTransformer expects (B, T, H, W, d), got {tuple(tokens.shape)}"
            )
        b, t, h, w, _ = tokens.shape
        s = h * w
        if t > self.max_t:
            raise ValueError(f"T={t} exceeds max_t={self.max_t}")
        if s > self.max_s:
            raise ValueError(f"H*W={s} exceeds max_s={self.max_s}")

        x = self.in_proj(tokens)              # (B, T, H, W, d)
        x = x.view(b, t, s, -1)
        x = x + self.pos_spatial[:, :, :s, :] + self.pos_temporal[:, :t, :, :]
        x = x.view(b, t * s, -1)

        for block in self.blocks:
            x = block(x, t, s)

        x = self.norm(x).mean(dim=1)
        return self.proj(x)
