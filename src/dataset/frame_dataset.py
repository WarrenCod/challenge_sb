"""
FrameDataset: flat per-frame iterator, used by Stage-1 MAE pretraining.

Walks root_dir/<class_folder>/<video_folder>/<frame>.jpg and yields one
transformed frame at a time. Labels are discarded (MAE doesn't use them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def collect_frame_paths(root_dir: Path) -> List[Path]:
    """Recursively collect every frame image under root_dir (depth 3: class/video/frame)."""
    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []
    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for p in sorted(video_dir.iterdir()):
                if p.suffix.lower() in exts:
                    paths.append(p)
    if not paths:
        raise RuntimeError(f"No frames under {root_dir}")
    return paths


def collect_frame_paths_deep(root_dirs) -> List[Path]:
    """Concatenate frame paths under any number of roots, depth-agnostic.

    Walks each root recursively (rglob) so it handles `train/CLS/video/frame.jpg`
    (3 levels) and `test/video/frame.jpg` (2 levels) uniformly. Required for
    SSL pretraining over the train+val+test pool.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out: List[Path] = []
    for root in root_dirs:
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    if not out:
        raise RuntimeError(f"No frames under any of: {root_dirs}")
    out.sort()
    return out


class FrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        max_samples: Optional[int] = None,
        frame_list: Optional[List[Path]] = None,
    ) -> None:
        if frame_list is None:
            self.frames = collect_frame_paths(Path(root_dir))
        else:
            self.frames = list(frame_list)
        if max_samples is not None:
            self.frames = self.frames[: int(max_samples)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.frames[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb)
