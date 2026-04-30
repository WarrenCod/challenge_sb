"""
Unlabeled video-clip dataset for V-JEPA SSL pretraining.

Samples T frames per video and returns a ``(T, C, H, W)`` tensor — no labels.
Still restricts to the "canonical" class folders (those present in val/) so the
train-time clip distribution matches what the supervised downstream task sees.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.video_dataset import _list_frame_paths, collect_video_samples


def _sample_frame_indices(
    num_available: int, num_frames: int, rng: random.Random
) -> List[int]:
    """Random stride + random start. Falls back to clamped indices at the tail."""
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_available == 1:
        return [0] * num_frames

    max_stride = max(1, (num_available - 1) // max(1, num_frames - 1))
    stride = rng.randint(1, max_stride)
    span = stride * (num_frames - 1)
    start_max = max(0, num_available - 1 - span)
    start = rng.randint(0, start_max)
    return [min(start + stride * i, num_available - 1) for i in range(num_frames)]


class UnlabeledVideoClipDataset(Dataset):
    """Yields ``(T, C, H, W)`` tensors; no labels."""

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable[[Image.Image], torch.Tensor],
        canonical_classes: Optional[set] = None,
        seed: int = 0,
    ) -> None:
        samples = collect_video_samples(Path(root_dir))
        if canonical_classes is not None:
            samples = [s for s in samples if s[0].parent.name in canonical_classes]
        if not samples:
            raise RuntimeError(
                f"No samples after canonical-class filter under {root_dir}"
            )
        self.video_dirs: List[Path] = [s[0] for s in samples]
        self.num_frames = num_frames
        self.transform = transform
        self._seed = seed

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int) -> torch.Tensor:
        video_dir = self.video_dirs[index]
        frame_paths = _list_frame_paths(video_dir)
        # Per-call RNG so DataLoader workers produce different temporal crops.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(
            (self._seed * 0x9E3779B1)
            ^ (index * 0xBF58476D)
            ^ (worker_id * 0x94D049BB)
            ^ int(torch.randint(0, 2**31 - 1, (1,)).item())
        )
        idxs = _sample_frame_indices(len(frame_paths), self.num_frames, rng)

        frames: List[torch.Tensor] = []
        for i in idxs:
            with Image.open(frame_paths[i]) as image:
                rgb = image.convert("RGB")
            frames.append(self.transform(rgb))
        return torch.stack(frames, dim=0)  # (T, C, H, W)
