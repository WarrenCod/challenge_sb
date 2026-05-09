"""Two-stream RGB + optical-flow model.

Composes two ``ModularVideoModel`` instances (one for RGB, one for flow) and
fuses their logits with a single learnable scalar gate. Per-sample modality
dropout zeros one stream's contribution at random during training to prevent
the model from leaning entirely on the easier modality.

Forward signature is ``(rgb, flow) -> logits``. ``forward_with_aux`` mirrors
``ModularVideoModel.forward_with_aux`` but returns per-stream auxiliary logits
so each stream can be deeply supervised independently.

Shape contract:
    rgb:  (B, T,    3, H, W)
    flow: (B, T-1, 2, H, W)   from FlowAwareClipAug / FlowAwareEvalTransform
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn

from models.modular import ModularVideoModel, build_modular_model


class TwoStreamModel(nn.Module):
    def __init__(
        self,
        rgb_model: ModularVideoModel,
        flow_model: ModularVideoModel,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        # sigmoid(gate) = RGB weight, (1 - sigmoid(gate)) = flow weight.
        # Init 0.0 -> 0.5/0.5 fusion.
        self.gate = nn.Parameter(torch.zeros(1))
        self.modality_dropout = float(modality_dropout)

    def _stream_masks(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample keep masks. Shape (B, 1). 0 means dropped this sample."""
        if not self.training or self.modality_dropout <= 0.0:
            ones = torch.ones(batch_size, 1, device=device)
            return ones, ones
        p = self.modality_dropout
        r = torch.rand(batch_size, 1, device=device)
        keep_rgb = (r >= p).float()           # drop RGB when r < p
        keep_flow = (r < 1.0 - p).float()     # drop flow when r >= 1 - p
        return keep_rgb, keep_flow

    def _fuse(
        self,
        logits_rgb: torch.Tensor,
        logits_flow: torch.Tensor,
        keep_rgb: torch.Tensor,
        keep_flow: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate)
        return alpha * keep_rgb * logits_rgb + (1.0 - alpha) * keep_flow * logits_flow

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        logits_rgb = self.rgb_model(rgb)
        logits_flow = self.flow_model(flow)
        keep_rgb, keep_flow = self._stream_masks(rgb.size(0), rgb.device)
        return self._fuse(logits_rgb, logits_flow, keep_rgb, keep_flow)

    def forward_with_aux(
        self, rgb: torch.Tensor, flow: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rgb_main, rgb_aux = self.rgb_model.forward_with_aux(rgb)
        flow_main, flow_aux = self.flow_model.forward_with_aux(flow)
        keep_rgb, keep_flow = self._stream_masks(rgb.size(0), rgb.device)
        fused = self._fuse(rgb_main, flow_main, keep_rgb, keep_flow)
        return fused, (rgb_aux, flow_aux)

    @property
    def aux_head(self) -> nn.Module | None:
        # Truthy iff at least one stream has aux supervision attached.
        # train.py uses ``getattr(model, 'aux_head', None)`` to gate the aux loss.
        rgb_has = getattr(self.rgb_model, "aux_head", None) is not None
        flow_has = getattr(self.flow_model, "aux_head", None) is not None
        if rgb_has or flow_has:
            return self  # marker; the real heads live on the substreams
        return None


def build_two_stream_model(cfg: Mapping[str, Any], num_classes: int) -> TwoStreamModel:
    """Build a TwoStreamModel from a model config of the form::

        model:
          name: two_stream
          num_classes: ${num_classes}
          modality_dropout: 0.1
          rgb:
            spatial: { ... }      # ModularVideoModel sub-config (RGB)
            temporal: { ... }
            classifier: { ... }
          flow:
            spatial: { ... }      # spatial.in_channels: 2, num_segments: T-1
            temporal: { ... }
            classifier: { ... }
    """
    if "rgb" not in cfg or "flow" not in cfg:
        raise ValueError("two_stream config requires 'rgb' and 'flow' sub-trees.")
    rgb_model = build_modular_model(cfg["rgb"], num_classes=num_classes)
    flow_model = build_modular_model(cfg["flow"], num_classes=num_classes)
    modality_dropout = float(cfg.get("modality_dropout", 0.1))
    return TwoStreamModel(
        rgb_model=rgb_model,
        flow_model=flow_model,
        modality_dropout=modality_dropout,
    )
