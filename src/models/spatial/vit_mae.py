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


class ViTMAEEncoder(SpatialEncoder):
    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        checkpoint_path: Optional[str] = None,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown vit variant {variant}. Options: {list(_VIT_VARIANTS)}")
        v = _VIT_VARIANTS[variant]

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
            x = blk(x)
        x = self.norm(x)
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
