# exp2b_mae_transformer — results

Single-axis regularization pass on top of `exp2_mae_transformer`. Same encoder
(MAE-pretrained ViT-S, Track 1), same 2-layer temporal transformer head, same
`num_frames=4` (hard cap), same optimizer / schedule. Only the regularization
knobs change.

Full plan + diff is in [`docs/exp2b_changes.md`](../docs/exp2b_changes.md);
this file is the post-run recap.

## Changes vs `exp2_mae_transformer`

| knob                         | exp2     | **exp2b** |
|------------------------------|----------|-----------|
| `training.strong_clip_aug`   | `false`  | **`true`** — `ConsistentClipAug`: RandomResizedCrop + ColorJitter + RandAugment + RandomErasing, applied with the same params across all frames. SSv2-safe (no flips). |
| `training.mixup_alpha`       | unset    | **`0.2`** |
| `model.spatial.drop_path`    | `0.1`    | **`0.2`** |
| `model.temporal.dropout`     | `0.1`    | **`0.2`** |

Unchanged: `num_frames=4`, `batch_size=16`, `lr=3e-4`, `llrd=0.75`,
`weight_decay=0.05`, `label_smoothing=0.1`, `epochs=50`, `warmup_epochs=5`,
`grad_clip=1.0`, `amp=true`. Same MAE Stage-1 init.

## Results (50 epochs)

| run    | train/acc | train/loss | val/best | val/loss (final) | gap (train − val) |
|--------|-----------|------------|----------|------------------|-------------------|
| exp2   | **0.994** | 0.673      | 0.5212   | 2.87 ↑           | **+47 pp**        |
| exp2b  | 0.298     | 2.186      | **0.5459** (ep 45) | 1.95 (flat) | **−25 pp** (val > train under aug) |

`val/best = 0.5459` at epoch 45. Final epoch 50: `train 0.2976 / val 0.5442`,
`val_loss 1.948`.

Best checkpoint: `checkpoints/exp2b_mae_transformer.pt`.

## Read

- **+2.5 pp val/best** vs exp2 (0.5459 vs 0.5212), in the lower half of the
  predicted 0.54–0.57 band.
- The exp2 pathology is gone. `val/loss` plateaus at ~1.95 from ~ep 40 instead
  of climbing for 35 epochs — the model is no longer becoming confidently
  wrong as it overfits.
- Train accuracy stays low because it is reported under heavy augmentation
  (RandomResizedCrop + ColorJitter + RandAugment + RandomErasing + mixup +
  label smoothing). A clean-pass train acc would be much higher; the fact
  that val > train under aug is the expected signature of a now-correctly
  regularized recipe, not under-fitting.
- The +2.5 pp gain is real but smaller than the gap-closing might suggest —
  consistent with the exp2b doc's "if it lands at ≤ 0.525, regularization
  isn't the bottleneck" criterion: we landed above 0.525, so regularization
  *was* a bottleneck, but not the only one.

## What this implies for next moves

The remaining gap is no longer about overfitting. With `num_frames` hard-capped
at 4, the next levers are encoder/pretraining (iBOT Stage-1 → exp3 once the
checkpoint lands) and possibly EMA / multi-clip eval at submission time.
Capacity / head depth changes are unlikely to move the needle while temporal
context is fixed at 4.

## Submission

Generated from this checkpoint:

```
submissions/exp2b_mae_transformer_submission.csv
```
