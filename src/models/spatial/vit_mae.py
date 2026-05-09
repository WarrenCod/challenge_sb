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
from models.mae_vit import PatchEmbed, _sincos_2d_posembed
from timm.models.vision_transformer import Block


_VIT_VARIANTS = {
    "vit_s_16": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, patch_size=16),
    "vit_ti_16": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, patch_size=16),
    "vit_b_16": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, patch_size=16),
}


class SpaceTimeBlock(nn.Module):
    """Wraps a timm spatial Block with a temporal-attention pre-pass.

    Divided space-time attention: at each patch position, attend across the T
    frames; then dispatch to the original (MAE-init) spatial Block. The
    temporal MHA's output projection is zero-initialised so at step 0 the
    wrapped block's forward is bit-identical to the underlying spatial Block
    — exp2m's t=0 forward equals exp2k's, and any drift from the 41.6%
    baseline must be earned by gradient updates.
    """

    def __init__(
        self,
        spatial_block: nn.Module,
        embed_dim: int,
        num_heads: int,
        num_frames: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.attn_t = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        nn.init.zeros_(self.attn_t.out_proj.weight)
        nn.init.zeros_(self.attn_t.out_proj.bias)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.num_frames, 1, embed_dim))
        self.block = spatial_block

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if T != self.num_frames:
            raise RuntimeError(
                f"SpaceTimeBlock built for num_frames={self.num_frames}, got T={T}"
            )
        BT, N1, D = x.shape
        B = BT // T
        xt = x.view(B, T, N1, D) + self.temporal_pos
        xt2 = xt.permute(0, 2, 1, 3).reshape(B * N1, T, D)
        h = self.norm_t(xt2)
        # fp32 softmax to match the rest of the project's stability fixes.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            h_fp32 = h.float()
            attn_out, _ = self.attn_t(h_fp32, h_fp32, h_fp32, need_weights=False)
        xt2 = xt2 + attn_out.to(xt2.dtype)
        xt = xt2.view(B, N1, T, D).permute(0, 2, 1, 3).reshape(BT, N1, D)
        return self.block(xt)


class ViTMAEEncoder(SpatialEncoder):
    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        checkpoint_path: Optional[str] = None,
        drop_path: float = 0.0,
        return_all_tokens: bool = False,
        space_time_layers: int = 0,
        space_time_num_frames: int = 4,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown vit variant {variant}. Options: {list(_VIT_VARIANTS)}")
        v = _VIT_VARIANTS[variant]

        self.return_all_tokens = bool(return_all_tokens)
        self.space_time_layers = int(space_time_layers)
        self.space_time_num_frames = int(space_time_num_frames)
        self.out_dim = v["embed_dim"]
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=v["patch_size"],
            in_chans=3,
            embed_dim=v["embed_dim"],
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, v["embed_dim"]))
        self.register_buffer(
            "pos_embed",
            _sincos_2d_posembed(v["embed_dim"], self.patch_embed.grid_size, cls_token=True),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(v["embed_dim"], v["num_heads"], v["mlp_ratio"], qkv_bias=True, drop_path=drop_path)
                for _ in range(v["depth"])
            ]
        )
        self.norm = nn.LayerNorm(v["embed_dim"])

        if checkpoint_path:
            self._load_mae_checkpoint(checkpoint_path)

        if self.space_time_layers > 0:
            depth = v["depth"]
            K = self.space_time_layers
            if K > depth:
                raise ValueError(f"space_time_layers={K} > depth={depth}")
            for i in range(depth - K, depth):
                self.blocks[i] = SpaceTimeBlock(
                    spatial_block=self.blocks[i],
                    embed_dim=v["embed_dim"],
                    num_heads=v["num_heads"],
                    num_frames=self.space_time_num_frames,
                )
            print(
                f"[vit_mae] wrapped last {K} block(s) "
                f"({depth-K}..{depth-1}) as SpaceTimeBlock(T={self.space_time_num_frames})"
            )

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
        imgs = video.reshape(B * T, C, H, W)

        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            if isinstance(blk, SpaceTimeBlock):
                x = blk(x, T)
            else:
                x = blk(x)
        x = self.norm(x)
        if self.return_all_tokens:
            # x: (B*T, N+1, D) -> (B, T, N+1, D)
            return x.view(B, T, x.shape[1], self.out_dim)
        cls_out = x[:, 0, :]                 # CLS token
        return cls_out.view(B, T, self.out_dim)

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
