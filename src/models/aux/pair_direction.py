"""Auxiliary head: classify the video from each ordered pair of frame summary tokens.

For ``T`` summary tokens (per-tubelet means in the VideoMAE case) there are
``T·(T-1)/2`` ordered (i<j) pairs. For each pair, we build a 4-feature input —
``[cls_i, cls_j, cls_j - cls_i, cls_i + cls_j]`` — push it through a 2-layer
MLP, and produce ``num_classes`` logits. Logits are averaged across pairs and
the resulting ``(B, K)`` tensor is supervised with the mixup-aware label CE.

Why this aux: SSv2 / "What Happens Next?" is fundamentally directional — what
*changed* between frames is the discriminative signal. The ``cls_j - cls_i``
feature carries the sign of the temporal change, giving the backbone an
explicit motion-supervision gradient separate from the main classifier's
permutation-friendlier aggregation through the temporal head.

VideoMAE caveat: with tubelet_time=2 and num_frames=4, the encoder exposes
only T'=2 "summary" tokens (per-tubelet spatial means), so only 1 ordered
pair exists. The aux gradient is much weaker than on Jabiru's per-frame
4-token setup (6 pairs). Expect a modest contribution (+0..0.5 pp).

Step-0 invariant: ``fc2`` weights and bias are zero-initialised, so the
head emits zero logits and the label-smoothed CE contributes a bounded
constant loss that does not dominate the early gradient.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class PairDirectionAuxHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_frames: int = 4,
        num_classes: int = 33,
        hidden_dim: int = 768,
    ) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2 for pairwise direction; got {num_frames}")
        self.embed_dim = int(embed_dim)
        self.num_frames = int(num_frames)
        self.num_classes = int(num_classes)

        self.fc1 = nn.Linear(4 * self.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        pairs: List[Tuple[int, int]] = [
            (i, j) for i in range(self.num_frames) for j in range(i + 1, self.num_frames)
        ]
        if len(pairs) == 0:
            raise ValueError("PairDirectionAuxHead needs at least one (i, j) pair.")
        self.register_buffer("pair_i", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor([p[1] for p in pairs], dtype=torch.long))

    def logits(self, frame_cls: torch.Tensor) -> torch.Tensor:
        """Compute averaged per-pair logits.

        Args:
            frame_cls: ``(B, T, D)`` — T summary tokens per video.
        Returns:
            ``(B, num_classes)``.
        """
        if frame_cls.dim() != 3:
            raise ValueError(
                f"PairDirectionAuxHead expects (B, T, D); got {tuple(frame_cls.shape)}"
            )
        B, T, D = frame_cls.shape
        if T != self.num_frames:
            raise ValueError(
                f"PairDirectionAuxHead got T={T} but num_frames={self.num_frames}"
            )
        cls_i = frame_cls.index_select(1, self.pair_i)
        cls_j = frame_cls.index_select(1, self.pair_j)
        feat = torch.cat([cls_i, cls_j, cls_j - cls_i, cls_i + cls_j], dim=-1)  # (B, P, 4D)
        h = self.act(self.fc1(feat))
        per_pair_logits = self.fc2(h)             # (B, P, K)
        return per_pair_logits.mean(dim=1)        # (B, K)

    def forward(
        self,
        frame_cls: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Mixup-aware CE on averaged pair logits."""
        logits = self.logits(frame_cls)
        return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
