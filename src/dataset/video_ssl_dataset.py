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


def collect_clip_dirs_deep(roots) -> List[Path]:
    """Find every video directory under any number of roots (mixed depths).

    Handles both ``train/CLS/video_<id>/frame_*.jpg`` (class/video/frame) and
    ``test/video_<id>/frame_*.jpg`` (video/frame) layouts uniformly: a directory
    is treated as a video folder iff it directly contains image files. Required
    for SSL clip pretraining over the train+val+test pool.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out: List[Path] = []
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            has_frame = any(
                (d / name).suffix.lower() in exts
                for name in (p.name for p in d.iterdir() if p.is_file())
            )
            if has_frame:
                out.append(d)
    if not out:
        raise RuntimeError(f"No video directories under any of: {roots}")
    out.sort()
    return out


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
    """Yields ``(T, C, H, W)`` tensors; no labels.

    Two construction modes:
      * ``root_dir`` (+ optional ``canonical_classes``) — walks a class/video tree.
      * ``video_dirs`` — pre-built list of video folders. Use this when pooling
        across mixed-depth roots like train (class/video) + test (video).
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        num_frames: int = 4,
        transform: Callable[[Image.Image], torch.Tensor] = None,
        canonical_classes: Optional[set] = None,
        seed: int = 0,
        video_dirs: Optional[List[Path]] = None,
        clip_transform: Optional[Callable] = None,
    ) -> None:
        if video_dirs is not None:
            self.video_dirs = [Path(d) for d in video_dirs]
        else:
            if root_dir is None:
                raise ValueError("Pass either root_dir or video_dirs.")
            samples = collect_video_samples(Path(root_dir))
            if canonical_classes is not None:
                samples = [s for s in samples if s[0].parent.name in canonical_classes]
            if not samples:
                raise RuntimeError(
                    f"No samples after canonical-class filter under {root_dir}"
                )
            self.video_dirs = [s[0] for s in samples]
        self.num_frames = num_frames
        self.transform = transform
        self.clip_transform = clip_transform
        if transform is None and clip_transform is None:
            raise ValueError("Pass either transform (per-frame) or clip_transform (per-clip).")
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

        pil_frames: List[Image.Image] = []
        for i in idxs:
            with Image.open(frame_paths[i]) as image:
                pil_frames.append(image.convert("RGB"))
        if self.clip_transform is not None:
            return self.clip_transform(pil_frames)  # (T, C, H, W)
        frames = [self.transform(img) for img in pil_frames]
        return torch.stack(frames, dim=0)  # (T, C, H, W)
