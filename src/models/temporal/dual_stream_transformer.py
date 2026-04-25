"""Dual-stream transformer over frame and motion tokens.

For T frames it builds:

    [CLS, F0, M01, F1, M12, F2, M23, F3, ...]

where Ft = projected per-frame feature and Mt,t+1 = MLP([x_t, x_{t+1}, x_{t+1}-x_t])
shared across pairs. Frames and motion tokens get separate learned type
embeddings; a learned absolute positional embedding is added per slot.

Designed for the ResNet50-TSM + this-temporal + MLP-classifier recipe (T=4),
but works for any T >= 2 up to ``max_frames``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class MotionMLP(nn.Module):
    """Shared pairwise fusion: [x_t, x_{t+1}, x_{t+1}-x_t] -> motion vector."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_a, x_b, x_b - x_a], dim=-1)
        return self.norm(self.net(z))


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
    return x * mask


class _Block(nn.Module):
    """Pre-norm transformer block with stochastic-depth on each residual."""

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
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + _drop_path(a, self.drop_path, self.training)
        h = self.norm2(x)
        x = x + _drop_path(self.ffn(h), self.drop_path, self.training)
        return x


class DualStreamTransformerTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        out_dim: Optional[int] = None,
        max_frames: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.out_dim = out_dim if out_dim is not None else d_model

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Backbone projection (in_dim -> d_model). Always norm so frame/motion
        # streams enter the encoder at comparable scale.
        if in_dim != d_model:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.input_proj = nn.LayerNorm(d_model)

        self.motion_mlp = MotionMLP(dim=d_model, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_frame = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_motion = nn.Parameter(torch.zeros(1, 1, d_model))

        # Sequence layout: [CLS, F0, M01, F1, M12, F2, ..., F_{T-1}]  -> 1 + (2T - 1)
        self.max_seq_len = 1 + 2 * max_frames - 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_frame, std=0.02)
        nn.init.trunc_normal_(self.type_motion, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Linear stochastic-depth schedule: 0 -> drop_path across layers.
        if num_layers > 1:
            dp_rates = [drop_path * i / (num_layers - 1) for i in range(num_layers)]
        else:
            dp_rates = [drop_path]
        self.layers = nn.ModuleList(
            [
                _Block(d_model, num_heads, dim_feedforward, dropout, dp_rates[i])
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = (
            nn.Linear(d_model, self.out_dim) if self.out_dim != d_model else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, in_dim)
        b, t, _ = features.shape
        if t < 2:
            raise ValueError(f"DualStream needs T>=2, got T={t}")

        x = self.input_proj(features)  # (B, T, d_model)

        x_a = x[:, :-1].reshape(b * (t - 1), self.d_model)
        x_b = x[:, 1:].reshape(b * (t - 1), self.d_model)
        motion = self.motion_mlp(x_a, x_b).view(b, t - 1, self.d_model)

        x = x + self.type_frame
        motion = motion + self.type_motion

        # Interleave: even slots = frames, odd slots = motion.
        seq_len = 2 * t - 1
        seq = torch.empty(b, seq_len, self.d_model, device=x.device, dtype=x.dtype)
        seq[:, 0::2] = x
        seq[:, 1::2] = motion

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, seq], dim=1)  # (B, 1 + 2T - 1, d_model)

        if tokens.size(1) > self.pos_embed.size(1):
            raise ValueError(
                f"Sequence length {tokens.size(1)} exceeds pos_embed capacity "
                f"{self.pos_embed.size(1)} (raise max_frames in the config)."
            )
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        for layer in self.layers:
            tokens = layer(tokens)

        return self.proj(self.norm(tokens[:, 0]))
