"""Joint space-time transformer head (TimeSformer / ViViT-1 style).

Consumes the V-JEPA / MAE / iBOT encoder output in ``pool="tokens"`` mode:
``(B, T, 1+N, D)`` — CLS at index 0, N patch tokens after. Drops the input
CLS (V-JEPA never trains it; MAE/iBOT trained it but it's redundant when
the head can attend over patches), prepends a fresh learned ``[CLS_st]``,
adds sincos-2D space pos-embed + a learned time pos-embed, and runs
``num_layers`` of pre-norm timm Blocks over the joint sequence.

Output: the ``[CLS_st]`` embedding, ``(B, D)``. ``out_dim = D``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from models.base import TemporalProcessor
from models._vit_utils import _sincos_2d_posembed


class SpaceTimeTransformer(TemporalProcessor):
    def __init__(
        self,
        in_dim: int,
        num_patches: int,
        max_frames: int = 8,
        num_heads: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.num_patches = num_patches
        self.max_frames = max_frames

        # 2D sincos space pos-embed, fixed buffer; broadcast over T at fwd.
        # Shape: (1, 1, N, D).
        grid = int(num_patches ** 0.5)
        if grid * grid != num_patches:
            raise ValueError(f"num_patches must be a square; got {num_patches}")
        self.register_buffer(
            "pos_embed_space",
            _sincos_2d_posembed(in_dim, grid, cls_token=False).unsqueeze(1),  # (1, 1, N, D)
            persistent=False,
        )

        # Learned time pos-embed, zero-init: (1, max_frames, 1, D).
        self.pos_embed_time = nn.Parameter(torch.zeros(1, max_frames, 1, in_dim))

        # Joint-sequence CLS, zero-init.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))

        mlp_ratio = dim_feedforward / in_dim
        dpr = [x.item() for x in torch.linspace(0.0, drop_path, num_layers)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    in_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    proj_drop=dropout,
                    attn_drop=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, 1+N, D) — CLS at idx 0, N patch tokens after.
        if features.dim() != 4:
            raise ValueError(
                f"SpaceTimeTransformer expects 4-D (B,T,1+N,D); got {tuple(features.shape)}. "
                "Set spatial.pool=tokens on the encoder."
            )
        B, T, S, D = features.shape
        if S != 1 + self.num_patches:
            raise ValueError(
                f"Expected S=1+num_patches={1 + self.num_patches}; got S={S}."
            )
        if T > self.max_frames:
            raise ValueError(f"T={T} exceeds max_frames={self.max_frames}")

        # Drop encoder CLS — not trained by V-JEPA; redundant for MAE/iBOT here.
        x = features[:, :, 1:, :]                              # (B, T, N, D)
        x = x + self.pos_embed_space + self.pos_embed_time[:, :T]  # (B, T, N, D)
        x = x.reshape(B, T * self.num_patches, D)              # (B, T*N, D)

        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                         # (B, 1+T*N, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                         # (B, D)
