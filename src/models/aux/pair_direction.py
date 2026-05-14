"""Auxiliary head: classify the video from each ordered pair of frame CLS tokens.

For T=4 frames there are 6 ordered pairs (i, j) with i < j. For each pair, we
build a 4-feature input — [cls_i, cls_j, cls_j - cls_i, cls_i + cls_j] — push
it through a 2-layer MLP, and produce 33-class logits. Logits are averaged
across the 6 pairs and the resulting (B, K) tensor is supervised with the
mixup-aware label CE.

Why this aux: SSv2 / "What Happens Next?" is fundamentally directional — what
*changed* between frames is the discriminative signal. Forcing the backbone to
encode action-discriminative information into pairwise frame deltas (the
``cls_j - cls_i`` feature carries the sign of the temporal change) gives an
explicit motion-supervision gradient, separate from the main classifier's
permutation-friendlier aggregation through the temporal head.

Step 0 invariant: ``fc2`` is zero-initialized, so the head emits zero logits
and the label-smoothed CE contributes a small constant loss that does not
dominate the early gradient.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class PairDirectionAuxHead(nn.Module):
    """Per-pair frame-direction classifier.

    Args:
        embed_dim: backbone CLS dim (D).
        num_frames: T (must equal dataset.num_frames). Hard-coded ordered
            (i < j) pairs are derived from this.
        num_classes: output classes (33 for the challenge).
        hidden_dim: MLP inner width.
    """

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
        # Zero-init the output so the head emits zero logits at step 0; the
        # label-smoothed CE contribution is a small bounded constant.
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
        """Compute (B, K) averaged per-pair logits.

        Args:
            frame_cls: (B, T, D).
        Returns:
            (B, num_classes).
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
        # (B, P, D) by indexing along the time axis.
        cls_i = frame_cls.index_select(1, self.pair_i)
        cls_j = frame_cls.index_select(1, self.pair_j)
        feat = torch.cat([cls_i, cls_j, cls_j - cls_i, cls_i + cls_j], dim=-1)  # (B, P, 4D)
        h = self.act(self.fc1(feat))
        per_pair_logits = self.fc2(h)            # (B, P, K)
        return per_pair_logits.mean(dim=1)       # (B, K)

    def forward(
        self,
        frame_cls: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Mixup-aware CE on averaged pair logits.

        Args:
            frame_cls: (B, T, D) per-frame CLS tokens.
            y_a, y_b: (B,) integer labels (the mixup pair).
            lam: mixup interpolation; ``lam * CE(_, y_a) + (1 - lam) * CE(_, y_b)``.
            loss_fn: the same CE (with label smoothing) used by the main head.
        Returns:
            scalar loss.
        """
        logits = self.logits(frame_cls)
        return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
