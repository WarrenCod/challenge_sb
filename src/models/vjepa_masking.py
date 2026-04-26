"""
Mask samplers on a tubelet grid ``(T, H, W)`` for V-JEPA-style pretraining.

Every call returns a pair of disjoint index tensors ``(context_ids, target_ids)``
into the flattened grid (row-major: ``t*H*W + h*W + w``). ``context_ids`` is what
the context encoder sees; ``target_ids`` is what the predictor has to regress.

Two strategies share this interface:

* :func:`spatial_block_mask` — several contiguous H×W rectangles masked across
  *all* time steps (V-JEPA "short-range" mask). Forces spatial semantics.
* :func:`temporal_future_mask` — mask every tubelet from a random time-cutoff to
  the end. Encodes "predict what happens next", aligned with SSv2 semantics.

A dispatcher :func:`sample_mask` picks one strategy per call according to a
probability mix.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch


def _flat_idx(t: int, h: int, w: int, H: int, W: int) -> int:
    return t * H * W + h * W + w


def spatial_block_mask(
    t_grid: int,
    h_grid: int,
    w_grid: int,
    target_ratio: float = 0.7,
    n_blocks: int = 3,
    aspect_min: float = 0.5,
    aspect_max: float = 2.0,
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask N rectangles in (H, W) applied to every time step.

    The target number of masked tubelets is ``target_ratio * T*H*W``; block sizes
    are sampled until the mask coverage first exceeds that ratio.
    """
    rng = rng or random.Random()
    total = t_grid * h_grid * w_grid
    target_count = int(target_ratio * total)

    mask_hw = torch.zeros(h_grid, w_grid, dtype=torch.bool)
    # Try to place at least n_blocks blocks; enlarge if coverage is short.
    attempts = 0
    while mask_hw.sum().item() * t_grid < target_count and attempts < 32:
        attempts += 1
        area_frac = target_ratio / max(1, n_blocks)
        area = max(1, int(area_frac * h_grid * w_grid))
        aspect = rng.uniform(aspect_min, aspect_max)
        bh = max(1, min(h_grid, int(round((area * aspect) ** 0.5))))
        bw = max(1, min(w_grid, int(round(area / max(1, bh)))))
        top = rng.randint(0, h_grid - bh)
        left = rng.randint(0, w_grid - bw)
        mask_hw[top : top + bh, left : left + bw] = True

    # Lift (H,W) mask to (T,H,W).
    mask = mask_hw.unsqueeze(0).expand(t_grid, -1, -1).reshape(-1)
    target_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
    context_ids = torch.nonzero(~mask, as_tuple=False).squeeze(1)
    return context_ids, target_ids


def temporal_future_mask(
    t_grid: int,
    h_grid: int,
    w_grid: int,
    keep_first: int,
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask all tubelets with time-index ``>= keep_first``.

    A small uniform jitter of ±1 is added so the model doesn't anchor to a
    fixed split.
    """
    rng = rng or random.Random()
    jitter = rng.randint(-1, 1)
    split = max(1, min(t_grid - 1, keep_first + jitter))

    ids = torch.arange(t_grid * h_grid * w_grid)
    t_of = ids // (h_grid * w_grid)
    mask = t_of >= split
    target_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
    context_ids = torch.nonzero(~mask, as_tuple=False).squeeze(1)
    return context_ids, target_ids


def sample_mask(
    t_grid: int,
    h_grid: int,
    w_grid: int,
    spatial_ratio: float = 0.7,
    spatial_n_blocks: int = 3,
    temporal_keep_first: int = 5,
    p_spatial: float = 0.5,
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Pick a strategy, return ``(context_ids, target_ids, strategy_name)``."""
    rng = rng or random.Random()
    if rng.random() < p_spatial:
        ctx, tgt = spatial_block_mask(
            t_grid, h_grid, w_grid,
            target_ratio=spatial_ratio,
            n_blocks=spatial_n_blocks,
            rng=rng,
        )
        return ctx, tgt, "spatial"
    ctx, tgt = temporal_future_mask(
        t_grid, h_grid, w_grid, keep_first=temporal_keep_first, rng=rng
    )
    return ctx, tgt, "temporal"


def batched_masks(
    batch_size: int,
    t_grid: int,
    h_grid: int,
    w_grid: int,
    spatial_ratio: float,
    spatial_n_blocks: int,
    temporal_keep_first: int,
    p_spatial: float,
    rng: random.Random,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Produce per-sample (context_ids, target_ids). Shapes vary across samples
    because spatial blocks land differently — batching is done via padded gather
    in the model, not by stacking ids here."""
    ctxs, tgts, strats = [], [], []
    for _ in range(batch_size):
        c, t, s = sample_mask(
            t_grid, h_grid, w_grid,
            spatial_ratio=spatial_ratio,
            spatial_n_blocks=spatial_n_blocks,
            temporal_keep_first=temporal_keep_first,
            p_spatial=p_spatial,
            rng=rng,
        )
        ctxs.append(c)
        tgts.append(t)
        strats.append(s)
    return ctxs, tgts, strats
