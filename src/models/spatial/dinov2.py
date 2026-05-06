"""DINOv2 ViT spatial encoder (timm-backed, pretrained on LVD-142M).

Per-frame encoder for Track-2 fine-tuning. Wraps a `timm` DINOv2 ViT — by default
the ViT-S/14 reg4 variant, since the DINOv2 paper reports register tokens give a
small but consistent quality lift on downstream transfer.

Pooling options match `vit.py`:
  - 'cls':     CLS token only        -> out_dim = embed_dim       (384 for S/14)
  - 'avg':     mean of patch tokens  -> out_dim = embed_dim
  - 'cls_avg': concat CLS + avg      -> out_dim = 2 * embed_dim   (768 for S/14)

Patch grid (14x14 patches) is independent of the temporal head — we pool to a
single vector per frame, so the temporal stage never sees the patch count.

`ordered_layers()` is exposed so `build_llrd_param_groups` (utils.py) works
unmodified — layer-wise LR decay is the standard recipe for fine-tuning a
pretrained ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import timm

from models.base import SpatialEncoder


_DEFAULT_VARIANT = "vit_small_patch14_reg4_dinov2.lvd142m"


class DINOv2Encoder(SpatialEncoder):
    """Per-frame DINOv2 ViT.

    Args:
        variant: any timm DINOv2 model id, e.g. 'vit_small_patch14_reg4_dinov2.lvd142m'
                 or 'vit_small_patch14_dinov2.lvd142m'.
        pretrained: load DINOv2 weights from HuggingFace Hub (default True; first
                    use downloads ~85 MB for ViT-S).
        pool: 'cls' | 'avg' | 'cls_avg'.
        drop_path: stochastic depth rate, linearly scaled across blocks by timm.
        freeze_backbone: if True, freeze all backbone params (linear-probe mode).
                         Default False — full fine-tune with LLRD is the recipe.
        image_size: input HxW. Must be a multiple of 14 (224 = 16*14).
    """

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        pretrained: bool = True,
        pool: str = "cls_avg",
        drop_path: float = 0.0,
        freeze_backbone: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if pool not in {"cls", "avg", "cls_avg"}:
            raise ValueError(f"pool must be one of cls/avg/cls_avg, got {pool}")
        if image_size % 14 != 0:
            raise ValueError(
                f"image_size must be a multiple of 14 for DINOv2 patch-14 ViT, got {image_size}"
            )

        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,                # drop classifier head
            img_size=image_size,
            drop_path_rate=drop_path,
        )
        self.embed_dim = self.backbone.embed_dim
        self.pool = pool
        self.out_dim = 2 * self.embed_dim if pool == "cls_avg" else self.embed_dim
        self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)

        # forward_features returns post-norm tokens: (N, num_prefix + num_patches, embed_dim).
        tokens = self.backbone.forward_features(frames)
        cls = tokens[:, 0]
        patches = tokens[:, self.num_prefix_tokens:]

        if self.pool == "cls":
            feats = cls
        elif self.pool == "avg":
            feats = patches.mean(dim=1)
        else:  # cls_avg
            feats = torch.cat([cls, patches.mean(dim=1)], dim=-1)

        return feats.view(B, T, self.out_dim)

    def ordered_layers(self):
        """Shallowest -> deepest, for layer-wise LR decay.

        cls_token / pos_embed / reg_token are bare nn.Parameters at the backbone
        level; build_llrd_param_groups picks them up via its leftover-params pass
        and assigns them the shallowest layer's LR.
        """
        yield ("embed", self.backbone.patch_embed)
        for i, blk in enumerate(self.backbone.blocks):
            yield (f"block_{i}", blk)
        yield ("norm", self.backbone.norm)
