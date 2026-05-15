"""
PseudoLabelVideoDataset: loads frames from test/video_<id> folders paired
with pre-computed soft-target probability vectors (from a frozen ensemble).

Used by noisy-student training: the test set has no real labels, so we
distill from the ensemble's softmax-mean output. The pseudo-label file is
produced once by `scripts/generate_pseudo_labels.py`.

Pseudo-label file shape (torch-saved dict):
    {
        "video_names":  list[str]  — folder names under test_dir (e.g. "video_19412")
        "soft_targets": tensor (N, num_classes) — softmax probs from the ensemble
        "max_probs":    tensor (N,)            — max prob per video (sanity)
        "argmax":       tensor (N,) long       — argmax of soft_targets (sanity)
        "threshold":    float                  — confidence threshold used
        "require_argmax_agreement": bool
        "source_models": list[str]
    }

Each __getitem__ returns:
    video_tensor: (T, C, H, W)
    soft_target:  (num_classes,) float32, sums to ~1
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.video_dataset import _list_frame_paths, _pick_frame_indices


class PseudoLabelVideoDataset(Dataset):
    def __init__(
        self,
        test_dir: str | Path,
        pseudo_file: str | Path,
        num_frames: int,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        clip_transform: Optional[Callable[[List[Image.Image]], torch.Tensor]] = None,
        random_offset_frac: bool = True,
    ) -> None:
        if transform is None and clip_transform is None:
            raise ValueError("PseudoLabelVideoDataset needs either transform or clip_transform.")
        self.test_dir = Path(test_dir).resolve()
        self.num_frames = int(num_frames)
        self.transform = transform
        self.clip_transform = clip_transform
        self.random_offset_frac = bool(random_offset_frac)

        blob = torch.load(Path(pseudo_file).resolve(), map_location="cpu", weights_only=False)
        names: List[str] = list(blob["video_names"])
        soft: torch.Tensor = blob["soft_targets"].float()
        if soft.dim() != 2 or soft.size(0) != len(names):
            raise ValueError(
                f"Pseudo-label file malformed: names={len(names)} vs soft_targets={tuple(soft.shape)}"
            )
        self.video_dirs: List[Path] = []
        kept_soft: List[torch.Tensor] = []
        missing = 0
        for name, target in zip(names, soft):
            vdir = self.test_dir / name
            if vdir.is_dir() and _list_frame_paths(vdir):
                self.video_dirs.append(vdir)
                kept_soft.append(target)
            else:
                missing += 1
        if missing:
            print(f"[pseudo] {missing} pseudo entries had no frames under {self.test_dir}; dropped")
        if not self.video_dirs:
            raise RuntimeError(f"PseudoLabelVideoDataset is empty under {self.test_dir}")
        self.soft_targets = torch.stack(kept_soft, dim=0)
        self.num_classes = int(self.soft_targets.size(1))

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir = self.video_dirs[index]
        frame_paths = _list_frame_paths(video_dir)
        offset_frac = float(torch.rand(()).item()) if self.random_offset_frac else 0.0
        indices = _pick_frame_indices(
            len(frame_paths), self.num_frames, offset_frac=offset_frac
        )

        pil_frames: List[Image.Image] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                pil_frames.append(image.convert("RGB"))

        if self.clip_transform is not None:
            video_tensor = self.clip_transform(pil_frames)
        else:
            tensors = [self.transform(img) for img in pil_frames]
            video_tensor = torch.stack(tensors, dim=0)

        return video_tensor, self.soft_targets[index].clone()
