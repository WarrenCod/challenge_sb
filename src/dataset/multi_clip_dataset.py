"""Two-clip wrapper for multi-clip consistency training.

For each video, ``__getitem__`` returns two independently sampled clips
(different temporal offsets *and* different spatial augmentations, both drawn
from the underlying ``VideoFrameDataset`` pipeline) and the shared label.

The default torch ``collate_fn`` handles 3-tuples natively, stacking each
position into its own tensor. No custom collate is needed.

Used by ``src/train.py`` when ``cfg.training.multi_clip.enabled=true``. The
training loop computes a symmetric KL between the two clips' predictions as a
consistency regularizer.
"""

from __future__ import annotations

from typing import Tuple

import torch

from dataset.video_dataset import VideoFrameDataset


class MultiClipVideoDataset(VideoFrameDataset):
    """Like ``VideoFrameDataset`` but yields ``(clip_a, clip_b, label)``.

    Both clips come from independent ``super().__getitem__`` calls — each
    redraws ``offset_frac`` (requires ``random_offset_frac=True``) and rerolls
    the clip transform (RandomResizedCrop / ColorJitter / RandAugment /
    RandomErasing). The two clips are therefore two augmented views of the
    same underlying video, in the spirit of FixMatch / BYOL consistency setups.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.random_offset_frac:
            raise ValueError(
                "MultiClipVideoDataset requires random_offset_frac=True; "
                "otherwise both clips would sample the same temporal offset."
            )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_a, label = super().__getitem__(index)
        clip_b, _ = super().__getitem__(index)
        return clip_a, clip_b, label
