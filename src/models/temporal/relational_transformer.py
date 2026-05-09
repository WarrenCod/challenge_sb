"""Transformer over [CLS, frame tokens, pairwise-difference tokens].

Distinct from `diff_transformer.py`, which concatenates each frame with its
forward difference along the channel dim into a 2d-wide token.

This module keeps token width = d and instead expands the token *count*:
for T input frames it emits 1 + T + (T-1) tokens — one CLS, T frame tokens,
(T-1) consecutive forward-difference tokens. Each token type carries its own
learned type-embedding so the encoder can tell frames from deltas; positional
embeddings are separate per slot.

For T = 4 this gives 8 tokens, vs. 5 for the plain transformer. The temporal
head no longer has to rediscover differencing inside attention — it gets it
as an explicit input.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.base import TemporalProcessor


class RelationalTransformerTemporal(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 6,
        num_layers: int = 3,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 8,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        token_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else token_dim
        self.max_len = int(max_len)

        if dim_feedforward is None:
            dim_feedforward = token_dim * 4

        # Type embeddings distinguish CLS / frame / delta tokens. Non-zero init
        # so the encoder can tell frame tokens from delta tokens at step 0.
        self.type_cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.type_frame = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.type_delta = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.type_cls, std=0.02)
        nn.init.trunc_normal_(self.type_frame, std=0.02)
        nn.init.trunc_normal_(self.type_delta, std=0.02)

        # CLS + per-slot positional embeddings. Zero-init for cold-start
        # stability (matches the rationale in transformer.py).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_frame = nn.Parameter(torch.zeros(1, self.max_len, token_dim))
        # max_len-1 delta slots (one fewer than frames).
        self.pos_delta = nn.Parameter(torch.zeros(1, max(self.max_len - 1, 1), token_dim))

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
            nn.Linear(token_dim, self.out_dim)
            if self.out_dim != token_dim
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = features.shape
        if num_frames > self.max_len:
            raise ValueError(
                f"T={num_frames} exceeds max_len={self.max_len}"
            )
        if num_frames < 2:
            raise ValueError(
                f"RelationalTransformerTemporal needs T >= 2; got T={num_frames}"
            )

        # Frame tokens: x_t + type_frame + pos_frame[t]
        frame_tokens = features + self.type_frame + self.pos_frame[:, :num_frames]

        # Delta tokens: (x_{t+1} - x_t) + type_delta + pos_delta[t]
        diffs = features[:, 1:] - features[:, :-1]  # (B, T-1, d)
        delta_tokens = diffs + self.type_delta + self.pos_delta[:, : num_frames - 1]

        # CLS token: cls + type_cls + pos_cls
        cls = (self.cls_token + self.type_cls + self.pos_cls).expand(batch_size, -1, -1)

        tokens = torch.cat([cls, frame_tokens, delta_tokens], dim=1)
        tokens = self.encoder(tokens)
        return self.proj(self.norm(tokens[:, 0]))
