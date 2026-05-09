"""Auxiliary head: from frames 1..T-1 spatial CLS tokens, predict frame T's CLS.

Cosine loss against ``stopgrad(target_cls)``. The stop-gradient prevents
trivial collapse where the backbone learns to make CLS tokens arbitrary
constants — the gradient flows only through the prediction MLP, not the
target.

Why this auxiliary loss: the challenge IS "what happens next?". Forcing the
model to predict the future frame embedding from the past gives the temporal
representation a direct predictive inductive bias, separate from the
discriminative CE/KD signal.

Total training loss with this aux:
    L = (1 - alpha)·CE_smoothed + alpha·KD + lambda_pred · (1 - cos(p, sg(z_T)))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictNextCLSAuxHead(nn.Module):
    """Predict the last-frame CLS from earlier frames' CLS via a 2-layer MLP.

    Args:
        embed_dim: backbone CLS dim (D).
        num_input_frames: number of past frames used as input. Must equal T-1
            for the train clips (T=4 → 3).
        hidden_dim: inner MLP width.
    """

    def __init__(self, embed_dim: int, num_input_frames: int = 3, hidden_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_input_frames = int(num_input_frames)
        self.fc1 = nn.Linear(self.num_input_frames * self.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.embed_dim)
        # Zero-init the output layer so initial prediction is the zero vector,
        # producing a small constant cosine loss that doesn't dominate at step 0.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, frame_cls: torch.Tensor) -> torch.Tensor:
        """Compute the cosine prediction loss.

        Args:
            frame_cls: (B, T, D) per-frame CLS tokens.
        Returns:
            scalar loss = mean(1 - cos(predicted, stopgrad(target))).
        """
        if frame_cls.dim() != 3:
            raise ValueError(
                f"PredictNextCLSAuxHead expects (B, T, D); got {tuple(frame_cls.shape)}"
            )
        B, T, D = frame_cls.shape
        if T != self.num_input_frames + 1:
            raise ValueError(
                f"PredictNextCLSAuxHead got T={T} but num_input_frames={self.num_input_frames}; "
                f"need T == num_input_frames + 1"
            )
        past = frame_cls[:, : T - 1, :].reshape(B, (T - 1) * D)  # (B, (T-1)*D)
        target = frame_cls[:, T - 1, :].detach()                  # (B, D), stopgrad
        # Compute MLP + cosine in fp32. fp16 cosine on backbone-derived
        # activations was a NaN source on exp2k @ ep10.
        with torch.amp.autocast(device_type=past.device.type, enabled=False):
            past32 = past.float()
            target32 = target.float()
            pred32 = self.fc2(self.act(self.fc1(past32)))         # (B, D)
            pred_n = F.normalize(pred32, dim=-1, eps=1e-6)
            target_n = F.normalize(target32, dim=-1, eps=1e-6)
            cos = (pred_n * target_n).sum(dim=-1)                 # (B,)
            loss = (1.0 - cos).mean()
        return loss
