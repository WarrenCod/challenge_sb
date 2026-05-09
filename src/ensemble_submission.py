#!/usr/bin/env python3
"""
Average per-video softmax probabilities from N submissions and write a single
ensemble CSV.

Each input is a CSV produced by ``create_submission.py`` *with*
``+submission.dump_probs=true``, which writes:

    submissions/<run>_submission.csv
    submissions/<run>_submission.probs.npy   # (N_videos, num_classes) float32
    submissions/<run>_submission.names.txt   # one video_name per line

Usage::

    python src/ensemble_submission.py \\
        --inputs submissions/exp2c_mae_transformer_tta3_submission.csv \\
                 submissions/exp2d_mae_distill_cutmix_tta3_submission.csv \\
                 submissions/exp2e_mae_bornagain_ensembleteacher_tta3_submission.csv \\
        --weights 1.0 1.0 1.0 \\
        --output submissions/ensemble_2c2d2e_tta3_submission.csv

If ``--weights`` is omitted, uniform averaging is used. Weights are normalised
internally so any positive scale works (e.g. raw val accuracies).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np


def _paths_for_csv(csv_path: Path) -> tuple[Path, Path]:
    probs = csv_path.with_suffix(".probs.npy")
    names = csv_path.with_suffix(".names.txt")
    return probs, names


def _read_names(p: Path) -> list[str]:
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more *_submission.csv paths (probs.npy/names.txt must exist next to each).")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-input weights (any positive scale). Default: uniform.")
    ap.add_argument("--output", required=True, help="Destination CSV path for the ensemble.")
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.inputs]
    weights = args.weights if args.weights is not None else [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise SystemExit(f"weights ({len(weights)}) != inputs ({len(inputs)})")
    w_arr = np.asarray(weights, dtype=np.float64)
    if (w_arr <= 0).any():
        raise SystemExit("weights must be strictly positive")
    w_arr = w_arr / w_arr.sum()

    probs_list: list[np.ndarray] = []
    names_ref: list[str] | None = None
    for csv_path in inputs:
        probs_path, names_path = _paths_for_csv(csv_path)
        if not probs_path.is_file() or not names_path.is_file():
            raise SystemExit(f"missing probs/names for {csv_path}: {probs_path}, {names_path}")
        probs = np.load(probs_path)
        names = _read_names(names_path)
        if probs.shape[0] != len(names):
            raise SystemExit(f"{csv_path}: probs rows {probs.shape[0]} != names {len(names)}")
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise SystemExit(f"video name order in {names_path} doesn't match {inputs[0].with_suffix('.names.txt')}")
        probs_list.append(probs.astype(np.float64))
        print(f"loaded {csv_path.name}: shape={probs.shape}, weight={w_arr[len(probs_list)-1]:.4f}")

    assert names_ref is not None
    stack = np.stack(probs_list, axis=0)  # (M, N, K)
    avg = (stack * w_arr[:, None, None]).sum(axis=0)  # (N, K)
    preds = avg.argmax(axis=-1).tolist()

    # Disagreement vs each individual model (using each model's own argmax).
    print("\n--- disagreement: ensemble vs each input ---")
    individual_preds = [p.argmax(axis=-1) for p in probs_list]
    ensemble_arr = np.asarray(preds)
    for csv_path, ind in zip(inputs, individual_preds):
        n_diff = int((ind != ensemble_arr).sum())
        n = len(ensemble_arr)
        print(f"  {csv_path.stem}: {n_diff}/{n} = {n_diff/n:.3f}")

    print("\n--- pairwise disagreement among inputs ---")
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            n_diff = int((individual_preds[i] != individual_preds[j]).sum())
            n = len(individual_preds[i])
            print(f"  {inputs[i].stem} vs {inputs[j].stem}: {n_diff}/{n} = {n_diff/n:.3f}")

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(names_ref, preds):
            w.writerow([name, int(pred)])
    print(f"\nWrote {len(preds)} rows to {out}")


if __name__ == "__main__":
    main()
