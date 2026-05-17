"""
Stage-2 spatial encoder: a timm ViT that can load a Stage-1 MAE encoder checkpoint.

Matches the architecture of ``src/models/mae_vit.py`` (same patch-embed + CLS +
sincos pos-embed + blocks + final norm) so that ``encoder_state_dict`` drops in
without key renaming. Decoder / mask token are absent — they were discarded at
the Stage-1 boundary.

Output per frame: the CLS-token embedding. (B, T, 3, H, W) -> (B, T, d).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.base import SpatialEncoder
from models._vit_utils import PatchEmbed, _sincos_2d_posembed, _sincos_3d_posembed
from models.vjepa import TubeletPatchEmbed
from timm.models.vision_transformer import Block


_VIT_VARIANTS = {
    "vit_s_16": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16),
    "vit_ti_16": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16),
    "vit_b_16": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16),
}


class ViTMAEEncoder(SpatialEncoder):
    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        checkpoint_path: Optional[str] = None,
        drop_path: float = 0.0,
        pool: str = "cls",
        freeze: bool = False,
        tubelet_size: int = 1,
        num_frames: int = 4,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown vit variant {variant}. Options: {list(_VIT_VARIANTS)}")
        if pool not in ("cls", "tokens", "mean_patches", "attn_pool"):
            raise ValueError(
                f"pool must be 'cls', 'tokens', 'mean_patches', or 'attn_pool', got {pool!r}"
            )
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        if tubelet_size > 1 and num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames {num_frames} must be divisible by tubelet_size {tubelet_size}")
        v = _VIT_VARIANTS[variant]

        self.pool = pool
        self.frozen = freeze
        self.out_dim = v["embed_dim"]
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.t_tokens = num_frames // tubelet_size if tubelet_size > 1 else 1

        if tubelet_size == 1:
            self.patch_embed = PatchEmbed(
                img_size=image_size,
                patch_size=v["patch_size"],
                in_chans=3,
                embed_dim=v["embed_dim"],
            )
            self.num_patches = self.patch_embed.num_patches
            self.grid_size = self.patch_embed.grid_size
            pos = _sincos_2d_posembed(v["embed_dim"], self.grid_size, cls_token=True)
        else:
            self.patch_embed = TubeletPatchEmbed(
                img_size=image_size,
                patch_size=v["patch_size"],
                tubelet_size=tubelet_size,
                in_chans=3,
                embed_dim=v["embed_dim"],
            )
            self.num_patches = self.patch_embed.num_spatial_patches
            self.grid_size = self.patch_embed.grid_size
            pos = _sincos_3d_posembed(v["embed_dim"], self.t_tokens, self.grid_size, cls_token=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, v["embed_dim"]))
        self.register_buffer("pos_embed", pos, persistent=False)
        self.blocks = nn.ModuleList(
            [
                Block(v["embed_dim"], v["num_heads"], v["mlp_ratio"], qkv_bias=True, drop_path=drop_path)
                for _ in range(v["depth"])
            ]
        )
        self.norm = nn.LayerNorm(v["embed_dim"])

        if checkpoint_path:
            self._load_mae_checkpoint(checkpoint_path)

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

        # attn-pool head is constructed AFTER the freeze loop so its params stay
        # trainable while the encoder remains frozen. dropout=0 keeps train/eval
        # behavior identical, matching the rest of the frozen-probe setup.
        if pool == "attn_pool":
            self.attn_pool_query = nn.Parameter(torch.zeros(1, 1, v["embed_dim"]))
            nn.init.trunc_normal_(self.attn_pool_query, std=0.02)
            self.attn_pool_attn = nn.MultiheadAttention(
                embed_dim=v["embed_dim"],
                num_heads=v["num_heads"],
                dropout=0.0,
                batch_first=True,
            )
            self.attn_pool_norm = nn.LayerNorm(v["embed_dim"])

    def _attn_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (M, K, D) -> (M, D). One learned query, cross-attends K tokens.
        M = tokens.shape[0]
        q = self.attn_pool_query.expand(M, -1, -1)
        y, _ = self.attn_pool_attn(q, tokens, tokens, need_weights=False)
        y = self.attn_pool_norm(y + q)
        return y.squeeze(1)

    def _load_mae_checkpoint(self, path: str) -> None:
        ck = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
        state = ck.get("encoder_state_dict", ck)  # support raw dict too
        missing, unexpected = self.load_state_dict(state, strict=False)
        # Expected: no missing / unexpected (layouts match).
        if missing:
            print(f"[vit_mae] missing keys when loading MAE ckpt: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[vit_mae] unexpected keys when loading MAE ckpt: {len(unexpected)} (e.g. {unexpected[:3]})")
        print(f"[vit_mae] loaded encoder from {path}")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        N = self.num_patches

        if self.tubelet_size == 1:
            # Per-frame ViT: fold T into batch.
            x = self.patch_embed(video.reshape(B * T, C, H, W))
            x = x + self.pos_embed[:, 1:, :]
            cls = self.cls_token + self.pos_embed[:, :1, :]
            cls = cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)                                  # (B*T, 1+N, D)
            T_out = T
        else:
            # Joint space-time ViT over the whole clip.
            T_tok = self.t_tokens
            if T != self.num_frames:
                raise ValueError(f"got T={T}, expected num_frames={self.num_frames}")
            x = video.permute(0, 2, 1, 3, 4).contiguous()      # (B, C, T, H, W)
            x = self.patch_embed(x)                            # (B, T_tok*N, D)
            x = x + self.pos_embed[:, 1:, :]
            cls = self.cls_token + self.pos_embed[:, :1, :]
            cls = cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)                     # (B, 1+T_tok*N, D)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)                                   # (B, 1+T_tok*N, D)
            # Reshape patch tokens to (B*T_tok, 1+N, D)-equivalent layout so
            # downstream temporal modules see one token-time per slice.
            patches = x[:, 1:, :].reshape(B, T_tok, N, self.out_dim)
            cls_b = x[:, :1, :]                                # (B, 1, D)
            T_out = T_tok

        if self.tubelet_size == 1:
            if self.pool == "cls":
                return x[:, 0, :].view(B, T, self.out_dim)
            if self.pool == "mean_patches":
                patches_mean = x[:, 1:, :].mean(dim=1)         # (B*T, D)
                return patches_mean.view(B, T, self.out_dim)
            if self.pool == "attn_pool":
                pooled = self._attn_pool(x)                    # (B*T, D)
                return pooled.view(B, T, self.out_dim)
            # pool == "tokens"
            return x.view(B, T, 1 + N, self.out_dim)

        # tubelet > 1
        if self.pool == "cls":
            # Single CLS shared across the clip: replicate across T_tok so
            # downstream temporal heads see a (B, T_tok, D) tensor.
            return cls_b.expand(-1, T_out, -1).contiguous()
        if self.pool == "mean_patches":
            return patches.mean(dim=2)                          # (B, T_tok, D)
        if self.pool == "attn_pool":
            cls_rep = cls_b.unsqueeze(1).expand(-1, T_out, -1, -1)
            tokens_per_slice = torch.cat([cls_rep, patches], dim=2)  # (B, T_tok, 1+N, D)
            tokens_flat = tokens_per_slice.reshape(B * T_out, 1 + N, self.out_dim)
            pooled = self._attn_pool(tokens_flat)                # (B*T_tok, D)
            return pooled.view(B, T_out, self.out_dim)
        # pool == "tokens": (B, T_tok, 1+N, D), CLS replicated per slice.
        cls_rep = cls_b.unsqueeze(1).expand(-1, T_out, -1, -1)
        return torch.cat([cls_rep, patches], dim=2)

    # Iterator over depth-ordered blocks for layer-wise LR decay.
    def ordered_layers(self):
        """Yield (name, submodule) from shallowest to deepest.
        'embed' groups patch_embed + cls_token + pos_embed; each block is its own layer;
        final 'norm' sits at the top. Used by build_llrd_param_groups in utils.py.
        """
        yield ("embed", nn.ModuleList([self.patch_embed]))  # cls_token handled as param
        for i, blk in enumerate(self.blocks):
            yield (f"block_{i}", blk)
        yield ("norm", self.norm)
