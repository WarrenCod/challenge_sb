# exp2c_mae_transformer — changelog

Single-axis follow-up on `exp2b_mae_transformer`. Same encoder
(MAE-pretrained ViT-S, Track 1), same 2-layer temporal transformer head,
same `num_frames=4` (hard cap), same regularization stack (strong clip aug,
mixup 0.2, drop_path 0.2, temporal dropout 0.2, label smoothing 0.1, WD 0.05).
Only the schedule and weight averaging change.

## Why

Final state of `exp2b` (50/50 epochs):

- `val/best = 0.5459` at **epoch 45** — best checkpoint was overwritten **36
  times** across 50 epochs.
- `train/loss` still drifting down at the end (`2.218 → 2.186` over ep 45–50)
  with `lr ≈ 0`.
- Slope by decade: ep 20→30 +5.1 pp, ep 30→40 +3.7 pp, **ep 40→50 +0.8 pp**.
  Curve flattened because the cosine ran out of LR, not because the model
  saturated.
- `train_acc 0.30 < val_acc 0.54` under aug → recipe is correctly regularized;
  the live model is not under-fit, it's under-trained.

Conclusion: the regularization recipe from exp2b is the right one — the run
just stopped too early, and there is a free ~1–2 pp on the table from EMA
weight averaging. exp2c is exp2b with **only** those two knobs moved, so the
delta vs exp2b is a clean measurement of "schedule + EMA" together.

## What changed vs `exp2b_mae_transformer`

| field                       | exp2b   | **exp2c (this)** | why |
|-----------------------------|---------|------------------|-----|
| `training.epochs`           | 50      | **100**          | exp2b's cosine bottomed out with the model still improving. Doubling gives the schedule room to actually converge. |
| `training.warmup_epochs`    | 5       | **8**            | Roughly preserves the exp2b warmup ratio (5/50 ≈ 8/100) so the early dynamics match. |
| `training.ema`              | unset   | **`true`**       | Enables `ModelEMA` (already wired in `train.py` / `utils.py`). The best checkpoint is saved from EMA weights when on, which is what gets evaluated. |
| `training.ema_decay`        | —       | **`0.9999`**     | Standard for ~2k steps/epoch × 100 epochs of video classifier training. |

Unchanged (intentionally — single-axis run vs exp2b on these two knobs):
`num_frames=4`, `batch_size=16`, `lr=3e-4`, `llrd=0.75`, `weight_decay=0.05`,
`label_smoothing=0.1`, `mixup_alpha=0.2`, `strong_clip_aug=true`,
`drop_path=0.2` (spatial), `dropout=0.2` (temporal), `grad_clip=1.0`,
`amp=true`. Same MAE Stage-1 init, same temporal head shape (`num_heads=6`,
`num_layers=2`, `dim_feedforward=1536`, `max_len=8`).

## What was *not* changed and why

- **Architecture / encoder size** — closing the schedule + averaging gap is
  the cheap lever. The architectural step is exp3 (iBOT Stage-1 → ViT-S),
  blocked on the remote pretraining checkpoint.
- **`num_frames`** — hard cap at 4 (data constraint).
- **TTA at submission** — preprocessed test videos contain exactly 4 frames
  each, so multi-clip TTA isn't possible without re-extracting test frames.
  Spatial TTA on already-square frames was estimated ~+0.3–0.7 pp; not worth
  the code change before exp2c lands.
- **Heavier head, EMA-warmup, stronger aug, lower WD** — all single-axis
  knobs that should be tested *after* exp2c, not bundled into it.

## Files

- New: `src/configs/experiment/exp2c_mae_transformer.yaml`
- Reuses (no edits): `src/configs/model/vit_mae_transformer.yaml`,
  `src/utils.py::ModelEMA`, `src/train.py` (already reads `training.ema`
  and saves EMA weights as the best checkpoint when on).
- Checkpoint will land at `checkpoints/exp2c_mae_transformer.pt`.

## How to run

Smoke test first:

```bash
cd /Data/challenge_sb
uv run python src/train.py experiment=exp2c_mae_transformer \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  training.resume=false wandb.enabled=false
```

(Smoke test writes its own `_last.pt`; delete both `checkpoints/exp2c_mae_transformer*.pt`
before the full run, or `training.resume=false` on the full launch and let
the first real epoch overwrite them.)

Full run inside tmux:

```bash
tmux new -s exp2c "uv run python src/train.py experiment=exp2c_mae_transformer 2>&1 | tee logs/exp2c_mae_transformer.log"
```

Evaluate / submit after training:

```bash
uv run python src/evaluate.py training.checkpoint_path=checkpoints/exp2c_mae_transformer.pt
uv run python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2c_mae_transformer.pt \
  dataset.submission_output=submissions/exp2c_mae_transformer_submission.csv
```

## Expected behaviour

- Train accuracy under aug rises slowly through the longer schedule, similar
  shape to exp2b but extending another ~50 epochs of useful steps.
- Best checkpoint (EMA weights) lands above exp2b's `0.5459`. Conservative
  band **0.555–0.575**; tighter bound depends on how much EMA picks up vs how
  much was already captured by val/best in exp2b.
- If `val/best ≤ 0.55`, schedule + EMA were not the bottleneck and the next
  move is the architectural step (exp3 — iBOT Stage-1 encoder).
- If `val/best ≥ 0.57`, this confirms exp2b was schedule-limited; exp2c
  becomes the new baseline for any further single-axis ablation.

## Status

Launched **2026-04-27** under `tmux new -s exp2c …`. Results section to be
appended once training finishes.
