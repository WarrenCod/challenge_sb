"""Smoke-test Farneback flow extraction on a handful of SSv2 clips.

Times per-pair flow at 256x256, prints summary, saves visualizations of one clip
under /tmp/flow_smoke/ for visual inspection.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA_ROOT = Path("/Data/challenge_sb/processed_data/train")
OUT_DIR = Path("/tmp/flow_smoke")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = 256


def load_frames(video_dir: Path) -> list[np.ndarray]:
    paths = sorted(video_dir.glob("frame_*.jpg"))
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        frames.append(np.array(img))
    return frames


def farneback(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    nxt = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev, nxt, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV color-wheel encoding (just for human-readable visualization)."""
    h, w, _ = flow.shape
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 16, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main() -> None:
    classes = sorted(DATA_ROOT.iterdir())[:5]
    sample_clips: list[Path] = []
    for cls in classes:
        videos = sorted(cls.glob("video_*"))
        if videos:
            sample_clips.append(videos[0])

    print(f"Sampling {len(sample_clips)} clips, target_size={TARGET_SIZE}")
    per_pair_times_ms: list[float] = []
    flow_stats: list[tuple[float, float, float]] = []  # (mean_mag, p99_mag, max_mag)

    for i, clip_dir in enumerate(sample_clips):
        frames = load_frames(clip_dir)
        if len(frames) != 4:
            print(f"  SKIP {clip_dir.name}: {len(frames)} frames")
            continue
        clip_pair_times = []
        clip_flows = []
        for t in range(len(frames) - 1):
            t0 = time.perf_counter()
            flow = farneback(frames[t], frames[t + 1])
            dt = (time.perf_counter() - t0) * 1000
            clip_pair_times.append(dt)
            clip_flows.append(flow)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_stats.append((float(mag.mean()), float(np.percentile(mag, 99)), float(mag.max())))
        per_pair_times_ms.extend(clip_pair_times)
        print(f"  [{i}] {clip_dir.parent.name}/{clip_dir.name}: "
              f"per-pair ms = {[f'{x:.1f}' for x in clip_pair_times]}, "
              f"clip total = {sum(clip_pair_times):.1f}ms")

        # Save visualizations for the first clip only.
        if i == 0:
            for t, flow in enumerate(clip_flows):
                Image.fromarray(frames[t]).save(OUT_DIR / f"clip0_frame_{t}.jpg")
                Image.fromarray(flow_to_rgb(flow)).save(OUT_DIR / f"clip0_flow_{t}_{t+1}.jpg")
            Image.fromarray(frames[-1]).save(OUT_DIR / f"clip0_frame_{len(frames)-1}.jpg")

    if per_pair_times_ms:
        arr = np.array(per_pair_times_ms)
        mags = np.array(flow_stats)
        print(f"\nPer-pair Farneback latency (ms):")
        print(f"  n={len(arr)}  mean={arr.mean():.2f}  median={np.median(arr):.2f}  "
              f"p95={np.percentile(arr, 95):.2f}  max={arr.max():.2f}")
        print(f"Per-clip cost (3 pairs): mean={3*arr.mean():.1f}ms")
        print(f"Flow magnitude (px): mean_mag avg={mags[:,0].mean():.2f}  "
              f"p99 avg={mags[:,1].mean():.2f}  max avg={mags[:,2].mean():.2f}")
        print(f"Visualizations: {OUT_DIR}/clip0_*")


if __name__ == "__main__":
    main()
