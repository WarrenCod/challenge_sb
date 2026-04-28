"""Vision Transformer spatial encoder.

Per-frame encoder. Internally, torchvision's VisionTransformer prepends a CLS
token and runs N transformer blocks, then keeps only the CLS embedding. We swap
``heads`` with Identity so the backbone returns either:
  - the CLS token alone   (pool='cls', the original behaviour, out_dim=hidden),
  - mean-pooled patch tokens (pool='avg', out_dim=hidden), or
  - concat of CLS + mean of patches (pool='cls_avg', out_dim=2*hidden).

Track-1 SSv2-style data benefits from patch-level features (fine-grained
motion is in the patches, not the CLS), so 'cls_avg' is the recommended pick
when training a richer modular pipeline from scratch.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer

from models.base import SpatialEncoder


def _torchvision_variant(factory, weights_enum) -> Callable[[bool], nn.Module]:
    def build(pretrained: bool) -> nn.Module:
        return factory(weights=weights_enum if pretrained else None)

    return build


def _custom_variant(**vit_kwargs) -> Callable[[bool], nn.Module]:
    def build(pretrained: bool) -> nn.Module:
        if pretrained:
            raise ValueError(
                "pretrained=True is not supported for custom ViT variants "
                "(torchvision ships no weights for this size)."
            )
        return VisionTransformer(image_size=224, **vit_kwargs)

    return build


# Each entry: variant name -> builder(pretrained) -> nn.Module with `.hidden_dim` + `.heads`.
_VIT_FACTORIES = {
    "vit_ti_16": _custom_variant(
        patch_size=16, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768
    ),
    "vit_s_16": _custom_variant(
        patch_size=16, num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536
    ),
    "vit_b_16": _torchvision_variant(models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": _torchvision_variant(models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": _torchvision_variant(models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1),
}


class ViTEncoder(SpatialEncoder):
    """Per-frame ViT.

    Args:
        variant: key in _VIT_FACTORIES.
        pretrained: load ImageNet weights (only valid for torchvision variants).
        pool: 'cls' (default, original behaviour), 'avg' (mean of patch tokens),
              or 'cls_avg' (concat of CLS and mean patch — out_dim doubles).
        drop_path: stochastic depth rate applied uniformly across encoder layers
                   on the residual branch. 0 disables (default keeps original behaviour).
    """

    def __init__(
        self,
        variant: str = "vit_b_16",
        pretrained: bool = False,
        pool: str = "cls",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in _VIT_FACTORIES:
            raise ValueError(
                f"Unknown vit variant: {variant}. Available: {list(_VIT_FACTORIES)}"
            )
        if pool not in {"cls", "avg", "cls_avg"}:
            raise ValueError(f"pool must be one of cls/avg/cls_avg, got {pool}")
        backbone = _VIT_FACTORIES[variant](pretrained)
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.hidden_dim = backbone.hidden_dim
        self.pool = pool
        self.out_dim = 2 * self.hidden_dim if pool == "cls_avg" else self.hidden_dim

        if drop_path > 0:
            _apply_drop_path(backbone, drop_path)

    def _forward_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """Replicates torchvision VisionTransformer.forward but returns all tokens.

        Returns a (N, 1+P, hidden) tensor: CLS at index 0, P patch tokens after.
        """
        bb = self.backbone
        x = bb._process_input(frames)              # (N, P, hidden)
        cls = bb.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)             # (N, 1+P, hidden)
        x = bb.encoder(x)                          # encoder applies pos_embed + dropout + blocks + ln
        return x

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frames = video.reshape(batch_size * num_frames, channels, height, width)

        if self.pool == "cls":
            feats = self.backbone(frames)                  # (N, hidden)
        else:
            tokens = self._forward_tokens(frames)          # (N, 1+P, hidden)
            cls = tokens[:, 0]
            patch_mean = tokens[:, 1:].mean(dim=1)
            if self.pool == "avg":
                feats = patch_mean
            else:  # cls_avg
                feats = torch.cat([cls, patch_mean], dim=-1)

        return feats.view(batch_size, num_frames, self.out_dim)


def _apply_drop_path(backbone: VisionTransformer, drop_path: float) -> None:
    """Wrap each encoder block's MLP+attention residuals with DropPath.

    Linear schedule from 0 (block 0) to drop_path (last block), the standard ViT recipe.
    """
    blocks = backbone.encoder.layers
    n = len(blocks)
    for i, block in enumerate(blocks):
        rate = drop_path * i / max(n - 1, 1)
        block.mlp = _ResWithDropPath(block.mlp, rate)
        block.self_attention = _ResWithDropPath(block.self_attention, rate)


class _ResWithDropPath(nn.Module):
    """Wraps a residual sub-module to apply stochastic depth on its output.

    torchvision's EncoderBlock already does ``x = x + sub(ln(x))``; we drop the
    ``sub(...)`` term randomly per-sample at training time.
    """

    def __init__(self, sub: nn.Module, drop_prob: float) -> None:
        super().__init__()
        self.sub = sub
        self.drop_prob = drop_prob

    def forward(self, *args, **kwargs):
        out = self.sub(*args, **kwargs)
        if not self.training or self.drop_prob == 0.0:
            return out
        # Self-attention returns (out, attn_weights); we only drop the tensor.
        if isinstance(out, tuple):
            tensor, *rest = out
            keep = _bernoulli_keep(tensor, self.drop_prob)
            return (tensor * keep, *rest)
        keep = _bernoulli_keep(out, self.drop_prob)
        return out * keep


def _bernoulli_keep(tensor: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Per-sample Bernoulli mask, scaled so expected output magnitude is preserved."""
    keep_prob = 1.0 - drop_prob
    shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    mask = tensor.new_empty(shape).bernoulli_(keep_prob)
    return mask / keep_prob
