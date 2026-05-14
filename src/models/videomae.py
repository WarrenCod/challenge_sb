"""VideoMAE classifier (joint space-time ViT, K400-pretrained).

Per-tubelet space-time encoder loaded from HuggingFace (MCG-NJU/videomae-base by
default — ViT-B/16, self-supervised pretraining on Kinetics-400). Tubelet patch
embed is (2, 16, 16): 4 frames @ 224 -> 2 temporal * 14 * 14 = 392 tokens. The
encoder applies joint space-time attention across all tokens.

The wrapper is shaped to look like a ModularVideoModel from the harness's POV
(``model.spatial.ordered_layers()`` for LLRD, head sits outside ``.spatial``):

    spatial (VideoMAEModel + ordered_layers)   (B,T,C,H,W) -> (B,N,d)
            ^ LLRD applies here
    head    (LayerNorm + dropout + Linear)     (B,d)       -> (B,K)
            ^ base_lr (no decay)

Pooling is mean over all space-time tokens — matches the original VideoMAE
classification recipe.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from transformers import VideoMAEConfig, VideoMAEModel


class VideoMAEEncoderWrap(nn.Module):
    """HuggingFace VideoMAEModel + ordered_layers for LLRD compatibility."""

    def __init__(
        self,
        hf_id: str = "MCG-NJU/videomae-base",
        pretrained: bool = True,
        num_frames: int = 4,
        image_size: int = 224,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        if pretrained:
            cfg = VideoMAEConfig.from_pretrained(
                hf_id,
                num_frames=num_frames,
                image_size=image_size,
                drop_path_rate=drop_path,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.encoder = VideoMAEModel.from_pretrained(hf_id, config=cfg)
        else:
            cfg = VideoMAEConfig(
                num_frames=num_frames,
                image_size=image_size,
                drop_path_rate=drop_path,
            )
            self.encoder = VideoMAEModel(cfg)
        self.embed_dim = self.encoder.config.hidden_size

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # HF VideoMAE expects (B, T, C, H, W) — same layout as our dataset.
        return self.encoder(video).last_hidden_state  # (B, N, d)

    def ordered_layers(self):
        """Shallowest -> deepest, for build_llrd_param_groups."""
        yield ("embed", self.encoder.embeddings)
        for i, blk in enumerate(self.encoder.encoder.layer):
            yield (f"block_{i}", blk)
        # K400-supervised checkpoints set use_mean_pooling=True; in that case
        # HF skips encoder.layernorm (the final norm lives in the classifier
        # head, which we replace with our own head_norm in VideoMAEClassifier).
        if self.encoder.layernorm is not None:
            yield ("norm", self.encoder.layernorm)


class VideoMAEClassifier(nn.Module):
    """Mean-pooled space-time tokens -> linear head."""

    def __init__(
        self,
        num_classes: int,
        hf_id: str = "MCG-NJU/videomae-base",
        pretrained: bool = True,
        num_frames: int = 4,
        image_size: int = 224,
        drop_path: float = 0.1,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial = VideoMAEEncoderWrap(
            hf_id=hf_id,
            pretrained=pretrained,
            num_frames=num_frames,
            image_size=image_size,
            drop_path=drop_path,
        )
        d = self.spatial.embed_dim
        self.head_norm = nn.LayerNorm(d)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(d, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        tokens = self.spatial(video)            # (B, N, d)
        pooled = tokens.mean(dim=1)             # (B, d)
        return self.head(self.head_drop(self.head_norm(pooled)))
