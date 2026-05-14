"""Two-clip wrapper for multi-clip consistency training.

For each video, ``__getitem__`` returns two independently augmented clips
(both drawn from the underlying ``VideoFrameDataset.__getitem__``, which
re-rolls the clip transform's RRC / ColorJitter params per call) plus the
shared label. The default torch ``collate_fn`` handles 3-tuples natively.

Used by ``src/train.py`` when ``cfg.training.multi_clip.enabled=true``. The
training loop computes a symmetric KL between the two clips' predictions
as a consistency regularizer (FixMatch / BYOL style, applied to the same
video — not across different videos).

Note vs Jabiru's version: nandou's ``VideoFrameDataset`` samples frames
deterministically (``linspace``) regardless of call — so the two clips
share frame indices and differ only in spatial augmentation. With
num_frames=4 on a 4-frame dataset this is the only available freedom; the
consistency signal is therefore "augmentation-invariant", not also
"temporal-offset-invariant". Still useful, but weaker than Jabiru's
random-offset multi-clip.
"""

from __future__ import annotations

from typing import Tuple

import torch

from dataset.video_dataset import VideoFrameDataset


class MultiClipVideoDataset(VideoFrameDataset):
    """Wraps ``VideoFrameDataset`` to yield ``(clip_a, clip_b, label)``."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_a, label = super().__getitem__(index)
        clip_b, _ = super().__getitem__(index)
        return clip_a, clip_b, label
