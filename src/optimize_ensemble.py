"""Find optimal ensemble weights from per-model val probs, then apply to test probs.

Inputs (on disk):
    submissions/<run>_val_tta3.probs.npy        per-model val softmax probs (N_val, K)
    submissions/val_tta3.labels.npy             val labels (N_val,)
    submissions/<run>_tta3_submission.probs.npy per-model test softmax probs (N_test, K)
    submissions/<run>_tta3_submission.names.txt video_name list (N_test,)

Procedure:
    1. Stack val probs -> P_val (M, N_val, K). Compute val acc per model.
    2. Optimize mixing weights w (M,) on the simplex to minimize val NLL of
       sum_m w_m * P_val[m]. Also evaluate uniform and acc-weighted baselines.
    3. Stack test probs -> P_test (M, N_test, K), apply optimal w, argmax,
       write submission CSV.

Run from /Data/challenge_sb:
    uv run python src/optimize_ensemble.py \
        --runs exp2c_mae_transformer exp2d_mae_distill_cutmix ... \
        --output submissions/ensemble_optimal_tta3_submission.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch

SUB_DIR = Path("/Data/challenge_sb/submissions")
EPS = 1e-8


def _val_probs_path(run: str) -> Path:
    return SUB_DIR / f"{run}_val_tta3.probs.npy"


def _test_probs_path(run: str) -> Path:
    return SUB_DIR / f"{run}_tta3_submission.probs.npy"


def _test_names_path(run: str) -> Path:
    return SUB_DIR / f"{run}_tta3_submission.names.txt"


def _check_files(runs: List[str]) -> None:
    missing = []
    for r in runs:
        for path in (_val_probs_path(r), _test_probs_path(r), _test_names_path(r)):
            if not path.is_file():
                missing.append(path)
    if missing:
        print("Missing inputs:")
        for p in missing:
            print(f"  {p}")
        raise SystemExit(1)
    if not (SUB_DIR / "val_tta3.labels.npy").is_file():
        raise SystemExit(f"Missing {SUB_DIR/'val_tta3.labels.npy'}")


def _val_top1(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def _val_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)
    return float(-np.log(p).mean())


def optimize_weights_torch(P_val: torch.Tensor, labels: torch.Tensor,
                           steps: int = 2000, lr: float = 0.05,
                           init: torch.Tensor | None = None) -> torch.Tensor:
    """Minimize NLL of (sum_m softmax(logits)_m * P_val[m]) wrt logits.
    Returns simplex weights (M,).
    """
    M = P_val.shape[0]
    if init is None:
        logits = torch.zeros(M, requires_grad=True)
    else:
        logits = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    best_nll = float("inf")
    best_w = None
    for step in range(steps):
        opt.zero_grad()
        w = torch.softmax(logits, dim=0)  # (M,)
        mix = (w[:, None, None] * P_val).sum(dim=0)  # (N, K)
        mix = mix.clamp(min=EPS)
        nll = -mix[torch.arange(labels.shape[0]), labels].log().mean()
        nll.backward()
        opt.step()
        nll_val = float(nll.item())
        if nll_val < best_nll:
            best_nll = nll_val
            best_w = w.detach().clone()
    return best_w  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run names (without _val_tta3.probs.npy / _tta3_submission.probs.npy suffixes).")
    ap.add_argument("--output", required=True, help="Output ensemble CSV path.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    runs = list(args.runs)
    _check_files(runs)

    labels = np.load(SUB_DIR / "val_tta3.labels.npy")
    print(f"val: {labels.shape[0]} samples")

    val_list = [np.load(_val_probs_path(r)) for r in runs]
    P_val = np.stack(val_list, axis=0)  # (M, N, K)
    print(f"val probs stacked: {P_val.shape}")

    print("\n--- per-model val (TTA=3) ---")
    per_model_acc = []
    for r, p in zip(runs, val_list):
        acc = _val_top1(p, labels)
        nll = _val_nll(p, labels)
        per_model_acc.append(acc)
        print(f"  {r:50s}  top1={acc:.4f}  nll={nll:.4f}")

    print("\n--- baselines ---")
    uniform = np.full(len(runs), 1.0 / len(runs), dtype=np.float64)
    p_unif = (uniform[:, None, None] * P_val).sum(axis=0)
    print(f"  uniform              top1={_val_top1(p_unif, labels):.4f}  nll={_val_nll(p_unif, labels):.4f}")

    acc_arr = np.array(per_model_acc, dtype=np.float64)
    accw = acc_arr / acc_arr.sum()
    p_accw = (accw[:, None, None] * P_val).sum(axis=0)
    print(f"  acc-weighted         top1={_val_top1(p_accw, labels):.4f}  nll={_val_nll(p_accw, labels):.4f}")

    print("\n--- optimizing weights via Adam on NLL ---")
    P_val_t = torch.from_numpy(P_val).float()
    labels_t = torch.from_numpy(labels).long()
    w_opt = optimize_weights_torch(P_val_t, labels_t, steps=args.steps, lr=args.lr).numpy()
    p_opt = (w_opt[:, None, None] * P_val).sum(axis=0)
    print(f"  optimal              top1={_val_top1(p_opt, labels):.4f}  nll={_val_nll(p_opt, labels):.4f}")
    print("  optimal weights:")
    for r, wv in zip(runs, w_opt):
        print(f"    {r:50s}  {wv:.4f}")

    # apply to test
    print("\n--- applying optimal weights to test ---")
    names0 = (_test_names_path(runs[0])).read_text().splitlines()
    names0 = [n.strip() for n in names0 if n.strip()]
    test_list = []
    for r in runs:
        p = np.load(_test_probs_path(r))
        names = [n.strip() for n in _test_names_path(r).read_text().splitlines() if n.strip()]
        if names != names0:
            raise SystemExit(f"name order mismatch between {runs[0]} and {r}")
        test_list.append(p)
    P_test = np.stack(test_list, axis=0)  # (M, N_test, K)
    p_mix = (w_opt[:, None, None] * P_test).sum(axis=0)
    preds = p_mix.argmax(axis=1)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["video_name", "predicted_class"])
        for name, pred in zip(names0, preds):
            w_csv.writerow([name, int(pred)])
    print(f"wrote {out}  rows={len(preds)}")


if __name__ == "__main__":
    main()
