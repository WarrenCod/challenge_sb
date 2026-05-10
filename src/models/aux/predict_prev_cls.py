"""Auxiliary head: from frames 1..T-1 spatial CLS tokens, predict frame 0's CLS.

Mirror of ``predict_next_cls`` with the time axis reversed: the network
predicts what the *initial* frame embedding should have been, given the
later frames. Together with predict-next, this gives the temporal
representation a symmetric "what came before / what comes next" inductive
bias — directly relevant to the SSv2-style task.

Cosine loss against ``stopgrad(target_cls)``. Same fp32 cast and zero-init
output projection as ``PredictNextCLSAuxHead`` to keep step-0 forward
identical to the no-aux baseline.

Total training loss with both auxes:
    L = (1 - alpha)·CE_smoothed
        + alpha·KD
        + lambda_next · (1 - cos(p_next, sg(z_T)))
        + lambda_prev · (1 - cos(p_prev, sg(z_0)))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictPrevCLSAuxHead(nn.Module):
    """Predict the first-frame CLS from later frames' CLS via a 2-layer MLP.

    Args:
        embed_dim: backbone CLS dim (D).
        num_input_frames: number of later frames used as input. Must equal
            T-1 for the train clips (T=4 → 3).
        hidden_dim: inner MLP width.
    """

    def __init__(self, embed_dim: int, num_input_frames: int = 3, hidden_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_input_frames = int(num_input_frames)
        self.fc1 = nn.Linear(self.num_input_frames * self.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.embed_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, frame_cls: torch.Tensor) -> torch.Tensor:
        """Compute the cosine prediction loss against frame-0 CLS.

        Args:
            frame_cls: (B, T, D) per-frame CLS tokens.
        Returns:
            scalar loss = mean(1 - cos(predicted, stopgrad(target))).
        """
        if frame_cls.dim() != 3:
            raise ValueError(
                f"PredictPrevCLSAuxHead expects (B, T, D); got {tuple(frame_cls.shape)}"
            )
        B, T, D = frame_cls.shape
        if T != self.num_input_frames + 1:
            raise ValueError(
                f"PredictPrevCLSAuxHead got T={T} but num_input_frames={self.num_input_frames}; "
                f"need T == num_input_frames + 1"
            )
        later = frame_cls[:, 1:T, :].reshape(B, (T - 1) * D)  # (B, (T-1)*D)
        target = frame_cls[:, 0, :].detach()                  # (B, D), stopgrad
        with torch.amp.autocast(device_type=later.device.type, enabled=False):
            later32 = later.float()
            target32 = target.float()
            pred32 = self.fc2(self.act(self.fc1(later32)))    # (B, D)
            pred_n = F.normalize(pred32, dim=-1, eps=1e-6)
            target_n = F.normalize(target32, dim=-1, eps=1e-6)
            cos = (pred_n * target_n).sum(dim=-1)             # (B,)
            loss = (1.0 - cos).mean()
        return loss
