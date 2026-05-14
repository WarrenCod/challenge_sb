"""
Stage-2 spatial encoder: a tubelet ViT loaded from a VideoMAE Stage-1 checkpoint.

Wraps ``models.videomae.VideoMAEEncoder`` so it conforms to the modular
``SpatialEncoder`` interface (input ``(B, T, 3, H, W)`` → output ``(B, T, d)``).
Exposes ``ordered_layers()`` so the existing layer-wise LR decay machinery works.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from models.base import SpatialEncoder
from models.spatial.videomae_st_block import VideoMAESpaceTimeBlock
from models.videomae import VideoMAEEncoder


_VIT_VARIANTS = {
    "vit_s_16": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, tubelet_size=16),
    "vit_b_16": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, tubelet_size=16),
}


def _resolve_drop_path(
    drop_path: float, depth: int, schedule: str
) -> Union[float, List[float]]:
    """Constant or linear-per-block drop_path schedule."""
    if schedule == "constant":
        return float(drop_path)
    if schedule == "linear":
        denom = max(1, depth - 1)
        return [float(drop_path) * i / denom for i in range(depth)]
    raise ValueError(f"drop_path_schedule must be 'constant' or 'linear'; got {schedule!r}")


class VideoMAESpatial(SpatialEncoder):
    def __init__(
        self,
        variant: str = "vit_s_16",
        image_size: int = 224,
        num_frames: int = 4,
        tubelet_time: int = 2,
        checkpoint_path: Optional[str] = None,
        drop_path: float = 0.0,
        drop_path_schedule: str = "constant",
        return_all_tokens: bool = False,
        space_time_layers: int = 0,
        space_time_num_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        if variant not in _VIT_VARIANTS:
            raise ValueError(f"Unknown videomae variant {variant}. Options: {list(_VIT_VARIANTS)}")
        v = _VIT_VARIANTS[variant]

        dp_rates = _resolve_drop_path(drop_path, v["depth"], drop_path_schedule)
        if isinstance(dp_rates, list):
            print(
                f"[videomae] linear drop_path schedule: "
                f"{dp_rates[0]:.3f} → {dp_rates[-1]:.3f} across {v['depth']} blocks"
            )

        self.encoder = VideoMAEEncoder(
            num_frames=num_frames,
            img_size=image_size,
            tubelet_time=tubelet_time,
            tubelet_size=v["tubelet_size"],
            embed_dim=v["embed_dim"],
            depth=v["depth"],
            num_heads=v["num_heads"],
            mlp_ratio=v["mlp_ratio"],
            drop_path=dp_rates,
            checkpoint_path=checkpoint_path,
        )
        self.out_dim = v["embed_dim"]
        self.return_all_tokens = bool(return_all_tokens)
        self.space_time_layers = int(space_time_layers)

        if self.space_time_layers > 0:
            K = self.space_time_layers
            if K > v["depth"]:
                raise ValueError(f"space_time_layers={K} > depth={v['depth']}")
            t_grid_default = self.encoder.t_grid  # T' from the loaded encoder
            t_grid = int(space_time_num_frames) if space_time_num_frames is not None else t_grid_default
            if t_grid != t_grid_default:
                print(
                    f"[videomae] WARNING: space_time_num_frames={t_grid} != encoder t_grid={t_grid_default}"
                )
            hw_count = self.encoder.hw_count
            for i in range(v["depth"] - K, v["depth"]):
                self.encoder.blocks[i] = VideoMAESpaceTimeBlock(
                    spatial_block=self.encoder.blocks[i],
                    embed_dim=v["embed_dim"],
                    num_heads=v["num_heads"],
                    t_grid=t_grid,
                    hw_count=hw_count,
                )
            print(
                f"[videomae] wrapped last {K} block(s) "
                f"({v['depth'] - K}..{v['depth'] - 1}) as VideoMAESpaceTimeBlock(T'={t_grid})"
            )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.return_all_tokens:
            # (B, T, 3, H, W) -> (B, T', H'·W', D). The Perceiver head keeps the
            # (frames, patches) factorisation so it can add a per-frame temporal
            # pos-embed before flattening to (B, T'·H'·W', D) for cross-attn.
            tokens = self.encoder.forward_features(video)            # (B, T'·H'·W', D)
            B, _, D = tokens.shape
            return tokens.view(B, self.encoder.t_grid, self.encoder.hw_count, D)
        # (B, T, 3, H, W) -> (B, T'=T/tubelet_time, D). The temporal head sees
        # T' tokens (2 for the default 4-frames+tubelet_time=2 setup), not T.
        return self.encoder(video)

    def ordered_layers(self):
        # Delegate to the inner encoder so build_llrd_param_groups walks the
        # tubelet_embed → blocks → norm chain (skipping our own .encoder. prefix
        # would break the path otherwise, but build_llrd_param_groups iterates
        # `model.spatial.ordered_layers()` and uses the returned submodules
        # directly via `module.named_parameters()` — so prefixes don't matter).
        return self.encoder.ordered_layers()
