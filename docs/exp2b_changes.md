# exp2b — regularization pass on `exp2_mae_transformer`

## Why

`exp2_mae_transformer` (Stage 2, Track 1, MAE-pretrained ViT-S + 2-layer
temporal transformer head) finished at:

| metric        | value  |
|---------------|--------|
| train/acc     | 0.994  |
| train/loss    | 0.673  |
| **val/best**  | 0.5212 (≈ epoch 12) |
| val/acc (ep 50) | 0.518 |
| val/loss (ep 50) | 2.87 (rising for the last 35 epochs) |

The 47-pp train↔val gap and the steadily climbing val/loss are the signature
of an unregularized recipe. The encoder, schedule, and temporal head are
fine — what was missing was real augmentation and stochastic regularization.

`num_frames=4` is fixed (data constraint), so this pass leaves the
architecture and clip length alone and only changes the regularization knobs.

## Diff vs `exp2_mae_transformer.yaml`

| knob                          | exp2     | exp2b   | rationale |
|-------------------------------|----------|---------|-----------|
| `training.strong_clip_aug`    | `false` (default) | **`true`** | Switches the train-time pipeline from `build_transforms` (Resize+ToTensor+Normalize, identical to eval) to `ConsistentClipAug`: RandomResizedCrop + ColorJitter + RandAugment + RandomErasing, **applied with the same params across all frames of a clip**. SSv2-safe — no horizontal/vertical flips. This is the single biggest lever for the train/val gap. |
| `training.mixup_alpha`        | unset (0.0) | **`0.2`** | Hooks into `mixup_batch` already wired through `train_one_epoch`. Compounds with the existing `label_smoothing=0.1`. |
| `model.spatial.drop_path`     | `0.1`    | **`0.2`** | ViT stochastic depth; standard fine-tuning level once aug is on. |
| `model.temporal.dropout`      | `0.1`    | **`0.2`** | Same idea on the 2-layer transformer head. |

Unchanged (intentionally):
`num_frames=4`, `batch_size=16`, `lr=3e-4`, `llrd=0.75`, `weight_decay=0.05`,
`label_smoothing=0.1`, `epochs=50`, `warmup_epochs=5`, `grad_clip=1.0`,
`amp=true`. Same MAE Stage-1 checkpoint, same transformer head shape
(num_heads=6, num_layers=2, dim_feedforward=1536, max_len=8).

## What was *not* changed and why

- **`num_frames`** stays at 4 — data constraint (only 4 frames available per
  clip in the current preprocessing). Going to 8 would otherwise be one of
  the biggest single wins for SSv2-style direction-sensitive classes.
- **EMA / model averaging** — not in the training loop today; would require
  a code change. The added regularization should make the late-stage
  val/loss climb mostly disappear, which was the main thing EMA would have
  smoothed over.
- **Test-time augmentation (multi-clip eval)** — submission-time change,
  out of scope here.
- **Architecture** (encoder size, head depth, classifier) — left alone;
  closing the train/val gap is the dominant lever, not capacity.

## Files

- New: `src/configs/experiment/exp2b_mae_transformer.yaml`
- Reuses (no edits): `src/configs/model/vit_mae_transformer.yaml`,
  `src/utils.py::build_strong_clip_transform`, `src/utils.py::mixup_batch`,
  `src/train.py` (already reads `training.strong_clip_aug` and
  `training.mixup_alpha`).
- Checkpoint will land at `checkpoints/exp2b_mae_transformer.pt`.

## How to run

Smoke test first (matches repo convention):

```bash
cd /Data/challenge_sb
uv run python src/train.py experiment=exp2b_mae_transformer \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4
```

Full run inside tmux (so SSH drops don't kill it):

```bash
tmux new -s exp2b "uv run python src/train.py experiment=exp2b_mae_transformer 2>&1 | tee logs/exp2b_mae_transformer.log"
```

Evaluate / submit after training:

```bash
uv run python src/evaluate.py training.checkpoint_path=checkpoints/exp2b_mae_transformer.pt
uv run python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2b_mae_transformer.pt \
  dataset.submission_output=processed_data/submission_exp2b_mae_transformer.csv
```

## Expected behaviour

- Train accuracy will rise more slowly and likely cap below 0.99.
- Val loss should stop climbing partway through (the climb in exp2 was the
  un-regularized model becoming confidently wrong).
- Val/best should land somewhere in the **0.54–0.57** range; tighter
  bound depends on how much the model was previously losing to spatial
  memorization vs. true temporal modeling. If it lands at ≤ 0.525, the
  bottleneck is not regularization and the next move is num_frames or
  encoder/pretraining (see `exp3_ibot_transformer`).

## Submission for the previous (`exp2`) checkpoint

Generated alongside this change:

```
processed_data/submission_exp2_mae_transformer.csv
```

(from `checkpoints/exp2_mae_transformer.pt`, val_acc 0.5212.)
