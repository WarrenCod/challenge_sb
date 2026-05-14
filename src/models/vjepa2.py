"""V-JEPA 2 classifier (joint space-time ViT, SSL-pretrained on web video).

Per-tubelet space-time encoder loaded from HuggingFace
(``facebook/vjepa2-vitl-fpc64-256`` by default — ViT-L/16, SSL pretraining only,
no supervised K400 head). Tubelet patch embed is (2, 16, 16): 4 frames @ 256
-> 2 temporal * 16 * 16 = 512 tokens.  Attention is RoPE-based, so the
positional encoding is parameter-free and stays in-distribution across crop
sizes — no pos-embed interpolation to debug.

Shaped to look like a ModularVideoModel from the harness's POV
(``model.spatial.ordered_layers()`` for LLRD, head sits outside ``.spatial``),
identical to VideoMAEClassifier so train.py / utils.py reuse applies as-is:

    spatial (VJEPA2Model + ordered_layers)     (B,T,C,H,W) -> (B,N,d)
            ^ LLRD applies here
    head    (LayerNorm + dropout + Linear)     (B,d)       -> (B,K)
            ^ base_lr (no decay)

Pooling is mean over all space-time tokens — parity with VideoMAEClassifier;
V-JEPA 2's attentive pooler is reserved for a v2 head if mean-pool plateaus.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from transformers import VJEPA2Config, VJEPA2Model


class VJepa2EncoderWrap(nn.Module):
    """HuggingFace VJEPA2Model + ordered_layers for LLRD compatibility."""

    def __init__(
        self,
        hf_id: str = "facebook/vjepa2-vitl-fpc64-256",
        pretrained: bool = True,
        num_frames: int = 4,
        image_size: int = 256,
        drop_path: float = 0.2,
    ) -> None:
        super().__init__()
        if pretrained:
            cfg = VJEPA2Config.from_pretrained(
                hf_id,
                frames_per_clip=num_frames,
                crop_size=image_size,
                drop_path_rate=drop_path,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.encoder = VJEPA2Model.from_pretrained(hf_id, config=cfg)
        else:
            cfg = VJEPA2Config(
                frames_per_clip=num_frames,
                crop_size=image_size,
                drop_path_rate=drop_path,
            )
            self.encoder = VJEPA2Model(cfg)
        self.embed_dim = self.encoder.config.hidden_size

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # HF V-JEPA 2 expects pixel_values_videos in (B, T, C, H, W) — same as our dataset.
        # get_vision_features = forward(skip_predictor=True): skips the JEPA predictor
        # (~30% VRAM tax) and returns encoder.last_hidden_state directly.
        return self.encoder.get_vision_features(video)  # (B, N, d)

    def ordered_layers(self):
        """Shallowest -> deepest, for build_llrd_param_groups."""
        yield ("embed", self.encoder.encoder.embeddings)
        for i, blk in enumerate(self.encoder.encoder.layer):
            yield (f"block_{i}", blk)
        yield ("norm", self.encoder.encoder.layernorm)


class VJepa2Classifier(nn.Module):
    """Mean-pooled space-time tokens -> linear head."""

    def __init__(
        self,
        num_classes: int,
        hf_id: str = "facebook/vjepa2-vitl-fpc64-256",
        pretrained: bool = True,
        num_frames: int = 4,
        image_size: int = 256,
        drop_path: float = 0.2,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial = VJepa2EncoderWrap(
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
