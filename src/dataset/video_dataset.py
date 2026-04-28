"""
VideoFrameDataset: loads a fixed number of RGB frames per video folder.

Expected layout under root_dir::

    root_dir/
      000_SomeClassName/
        video_12345/
          frame_000.jpg
          frame_001.jpg
          ...
      001_AnotherClass/
        ...

Class index is parsed from the leading number in the class folder name (000, 001, ...).
Each __getitem__ returns:
    video_tensor: float tensor of shape (T, C, H, W)
    label: int64 scalar class index
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_frame_paths(video_dir: Path) -> List[Path]:
    """All image files in a video folder, sorted by name."""
    paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)


def _parse_class_index(class_dir_name: str) -> Optional[int]:
    """
    Expect folder names like '017_Class_name'. Returns 17, or None if no prefix.
    """
    match = re.match(r"^(\d+)_", class_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_video_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    """
    Walk root_dir: each class folder contains video subfolders with frames.

    Returns list of (video_folder_path, class_index).
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: List[Tuple[Path, int]] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    # If folders lack numeric prefix, assign indices by sorted order (0..C-1).
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue
            samples.append((video_dir, class_index))

    if len(samples) == 0:
        raise RuntimeError(f"No video folders with frames under {root_dir}")

    return samples


def _pick_frame_indices(
    num_available: int, num_frames: int, offset: float = 0.0
) -> List[int]:
    """
    Evenly spaced indices in [0, num_available - 1], inclusive, shifted by
    ``offset * step`` and clamped to the valid range.

    ``offset=0.0`` reproduces the original linspace (default). Used at inference
    for temporal TTA: calling with offsets like [-0.5, 0.0, +0.5] yields three
    differently-jittered samplings of the same clip whose logits can be averaged.
    If fewer frames than requested, indices may repeat (last frame duplicated).
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if num_available == 1:
        return [0] * num_frames

    positions = torch.linspace(0, num_available - 1, steps=num_frames)
    if offset != 0.0 and num_frames > 1:
        step = (num_available - 1) / (num_frames - 1)
        positions = positions + offset * step
    last = num_available - 1
    indices = [max(0, min(last, int(round(float(x))))) for x in positions]
    return indices


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sample_list: Optional[List[Tuple[Path, int]]] = None,
        clip_transform: Optional[Callable[[List[Image.Image]], torch.Tensor]] = None,
        frame_offset: float = 0.0,
    ) -> None:
        """
        Args:
            root_dir: Split root (contains class folders).
            num_frames: T in the returned tensor (T, C, H, W).
            transform: Per-frame PIL -> tensor pipeline. Used when clip_transform is None.
            sample_list: Optional pre-built list of (video_dir, label) for train/val splits.
            clip_transform: If set, takes the list of PIL frames and returns a (T, C, H, W)
                tensor directly. Use this for clip-aware augmentations (RandomResizedCrop,
                ColorJitter, RandAugment) applied identically across frames — required
                whenever you want temporal coherence under strong aug.
        """
        if transform is None and clip_transform is None:
            raise ValueError("VideoFrameDataset needs either transform or clip_transform.")

        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.clip_transform = clip_transform
        self.frame_offset = frame_offset

        if sample_list is None:
            self.samples = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        indices = _pick_frame_indices(
            len(frame_paths), self.num_frames, offset=self.frame_offset
        )

        pil_frames: List[Image.Image] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                pil_frames.append(image.convert("RGB"))

        if self.clip_transform is not None:
            video_tensor = self.clip_transform(pil_frames)  # (T, C, H, W)
        else:
            tensors = [self.transform(img) for img in pil_frames]
            video_tensor = torch.stack(tensors, dim=0)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return video_tensor, label_tensor
