"""Composer + registries + factory for the plug-and-play video model.

Usage from a future build_model branch:

    from models.modular import build_modular_model
    model = build_modular_model(cfg.model, num_classes=cfg.num_classes)

Expected cfg shape:

    model:
      name: modular
      spatial:    { name: resnet, variant: resnet18, pretrained: true }
      temporal:   { name: transformer, num_heads: 8, num_layers: 2 }
      classifier: { name: linear }        # or: { name: mlp, hidden_dim: 512, dropout: 0.5 }

`in_dim` / `num_classes` are injected by the factory — do not put them in the config.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from models.base import Classifier, SpatialEncoder, TemporalProcessor
from models.classifier.linear import LinearClassifier
from models.classifier.mlp import MLPClassifier
from models.spatial.dinov2 import DINOv2Encoder
from models.spatial.resnet import ResNetEncoder
from models.spatial.resnet_tsm import ResNetTSMEncoder
from models.spatial.vit import ViTEncoder
from models.spatial.vit_mae import ViTMAEEncoder
from models.temporal.diff_transformer import DiffTransformerTemporal
from models.temporal.lstm import LSTMTemporal
from models.temporal.mean_pool import MeanPoolTemporal
from models.temporal.transformer import TransformerTemporal

try:
    from omegaconf import DictConfig, OmegaConf

    _HAS_OMEGACONF = True
except ImportError:
    _HAS_OMEGACONF = False


SPATIAL_REGISTRY: Dict[str, type] = {
    "resnet": ResNetEncoder,
    "resnet_tsm": ResNetTSMEncoder,
    "vit": ViTEncoder,
    "vit_mae": ViTMAEEncoder,
    "dinov2": DINOv2Encoder,
}
TEMPORAL_REGISTRY: Dict[str, type] = {
    "mean_pool": MeanPoolTemporal,
    "lstm": LSTMTemporal,
    "transformer": TransformerTemporal,
    "diff_transformer": DiffTransformerTemporal,
}
CLASSIFIER_REGISTRY: Dict[str, type] = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
}


class ModularVideoModel(nn.Module):
    """Chains the three slots. Checks shape compatibility at construction time."""

    def __init__(
        self,
        spatial: SpatialEncoder,
        temporal: TemporalProcessor,
        classifier: Classifier,
    ) -> None:
        super().__init__()
        if temporal.in_dim != spatial.out_dim:
            raise ValueError(
                f"Dim mismatch: spatial.out_dim={spatial.out_dim} vs "
                f"temporal.in_dim={temporal.in_dim}"
            )
        if classifier.in_dim != temporal.out_dim:
            raise ValueError(
                f"Dim mismatch: temporal.out_dim={temporal.out_dim} vs "
                f"classifier.in_dim={classifier.in_dim}"
            )
        self.spatial = spatial
        self.temporal = temporal
        self.classifier = classifier

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        frame_features = self.spatial(video)       # (B, T, d)
        video_vector = self.temporal(frame_features)  # (B, d')
        return self.classifier(video_vector)       # (B, num_classes)


def _to_plain(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if _HAS_OMEGACONF and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _pop_name(cfg: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    plain = _to_plain(cfg)
    if "name" not in plain:
        raise ValueError("Sub-component config must have a 'name' field.")
    name = plain.pop("name")
    return name, plain


def build_spatial(cfg: Mapping[str, Any]) -> SpatialEncoder:
    name, kwargs = _pop_name(cfg)
    if name not in SPATIAL_REGISTRY:
        raise ValueError(
            f"Unknown spatial encoder '{name}'. Available: {sorted(SPATIAL_REGISTRY)}"
        )
    return SPATIAL_REGISTRY[name](**kwargs)


def build_temporal(cfg: Mapping[str, Any], in_dim: int) -> TemporalProcessor:
    name, kwargs = _pop_name(cfg)
    if name not in TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal processor '{name}'. Available: {sorted(TEMPORAL_REGISTRY)}"
        )
    kwargs["in_dim"] = in_dim
    return TEMPORAL_REGISTRY[name](**kwargs)


def build_classifier(
    cfg: Mapping[str, Any], in_dim: int, num_classes: int
) -> Classifier:
    name, kwargs = _pop_name(cfg)
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {sorted(CLASSIFIER_REGISTRY)}"
        )
    kwargs["in_dim"] = in_dim
    kwargs["num_classes"] = num_classes
    return CLASSIFIER_REGISTRY[name](**kwargs)


def build_modular_model(
    cfg: Mapping[str, Any], num_classes: int
) -> ModularVideoModel:
    spatial = build_spatial(cfg["spatial"])
    temporal = build_temporal(cfg["temporal"], in_dim=spatial.out_dim)
    classifier = build_classifier(
        cfg["classifier"], in_dim=temporal.out_dim, num_classes=num_classes
    )
    return ModularVideoModel(spatial=spatial, temporal=temporal, classifier=classifier)
