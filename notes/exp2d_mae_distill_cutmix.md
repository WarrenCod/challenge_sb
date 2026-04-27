# exp2d_mae_distill_cutmix — changelog

Two-axis follow-up on `exp2c_mae_transformer`. Same encoder (MAE-pretrained
ViT-S, Track 1), same 2-layer temporal transformer head, same `num_frames=4`,
same regularization stack (strong clip aug, mixup 0.2, drop_path 0.2, temporal
dropout 0.2, label smoothing 0.1, WD 0.05), same schedule (100 epochs, warmup
8), same EMA (decay 0.9999). Two additions: **CutMix** and **self-distillation
from exp2c**.

## Why

exp2c hit `val/best = 0.5868` at epoch 80 (+4.09 pp over exp2b) by fixing
schedule + averaging. The recipe is now correctly trained — `train_acc 0.39 <
val_acc 0.59` under aug, val_loss has a clear bottom around epoch 60, EMA is
absorbing oscillation. Further Stage-2 wins have to come from new signal,
not more steps.

Two cheap, well-established sources of new signal stack cleanly on top of
exp2c without changing the architecture or pretraining:

1. **CutMix** — every published ViT finetune recipe (DeiT III, MAE, BEiT)
   alternates Mixup *and* CutMix per-batch, with `cutmix_alpha ≈ 1.0`.
   exp2c only ran `mixup_alpha=0.2`. Both functions and the 50/50
   alternation logic already live in `train_one_epoch`, so this is a
   one-line config change. Expected: **+0.5 to +1.0 pp**.

2. **Self-distillation** from the exp2c EMA snapshot — KL divergence on
   temperature-scaled logits, applied on the same already-mixed input the
   student sees. Self-distillation tends to add signal even when the
   teacher is the same architecture (label-smoothing-on-steroids, plus
   inter-class similarity structure). Expected: **+1 to +2 pp**.

The two are orthogonal: CutMix changes the input distribution, distillation
changes the loss target. They should compose. Bundling them into one
exp2d (rather than running each alone) is the explicit single-shot tradeoff
— the per-axis ablation is sacrificed for one clean comparison vs exp2c.

## What changed vs `exp2c_mae_transformer`

| field                       | exp2c   | **exp2d (this)** | why |
|-----------------------------|---------|------------------|-----|
| `training.cutmix_alpha`     | unset   | **`1.0`**        | Standard ViT finetune recipe. Alternates 50/50 with `mixup_alpha=0.2` (already wired in `train_one_epoch`, lines 140–148). |
| `training.distill`          | absent  | **block**        | New optional config block read in `main()`. Builds a frozen teacher from the embedded Hydra config of `checkpoints/exp2c_mae_transformer.pt`. |
| `training.distill.alpha`    | —       | **`0.5`**        | Splits the loss 50/50 between hard CE on (mixed) labels and soft KL from teacher. Conservative starting point. |
| `training.distill.temperature` | —    | **`4.0`**        | DeiT default. Softens both distributions; KL term is scaled by `T²` so total magnitude stays comparable to CE. |

Unchanged (intentionally — single-axis on these two additions):
`num_frames=4`, `batch_size=16`, `lr=3e-4`, `llrd=0.75`, `weight_decay=0.05`,
`label_smoothing=0.1`, `mixup_alpha=0.2` (kept low to keep CutMix the headline
change), `strong_clip_aug=true`, `drop_path=0.2` (spatial), `dropout=0.2`
(temporal), `grad_clip=1.0`, `amp=true`, `epochs=100`, `warmup_epochs=8`,
`ema=true`, `ema_decay=0.9999`. Same MAE Stage-1 init, same temporal head
shape.

## Implementation

Three localized edits to `src/train.py`:

1. **`load_teacher(checkpoint_path, device)`** — rebuilds a frozen teacher
   from the checkpoint's embedded `cfg` via `build_model()`, loads the EMA
   state dict (which is what exp2c's `model_state_dict` already is — see
   `train.py` line ~514: best is saved from EMA when on), calls `eval()`
   and `requires_grad_(False)`. Same loader pattern as
   `create_submission.py::build_model_from_checkpoint`.

2. **`train_one_epoch` signature** extended with `teacher`,
   `distill_alpha`, `distill_temperature`. Inside `_forward_loss`, after
   computing student `logits` and the (mixup/cutmix-aware) hard CE loss,
   if a teacher is provided:

   ```python
   with torch.no_grad():
       teacher_logits = teacher(video)        # same already-mixed input
   tk = distill_temperature
   student_logp = F.log_softmax(logits / tk, dim=-1)
   teacher_p    = F.softmax(teacher_logits / tk, dim=-1)
   kl = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (tk * tk)
   loss = (1.0 - distill_alpha) * loss + distill_alpha * kl
   ```

   The teacher forward runs inside the same `autocast` block as the
   student (cheap) and is wrapped in `torch.no_grad()` so it contributes
   no gradients.

3. **`main()`** reads `cfg.training.distill`; if present, calls
   `load_teacher` and threads `teacher`, `distill_alpha`,
   `distill_temperature` into `train_one_epoch`. Absent → no-op (existing
   experiments are unaffected).

CutMix needed **no code changes** — `cutmix_batch` and the 50/50 alternation
were already present in `train_one_epoch`, and `cutmix_alpha` was already
read from config (line 462). It was effectively dormant until now.

## What was *not* changed and why

- **`mixup_alpha` left at 0.2** — bumping it is a separate single-axis
  experiment (exp2e candidate). Standard recipes use 0.8, but doubling
  CutMix and Mixup intensity in the same run would muddle the delta.
- **Distillation alpha / temperature** — `α=0.5, T=4.0` are conservative
  defaults; a sweep is exp2e/exp2f territory.
- **Hard-distillation token (DeiT-style)** — would require touching the
  model. The soft-KL formulation works fine for self-distillation and
  keeps the student architecture identical to exp2c.
- **Architecture** — exp3 (iBOT Stage-1) and joint space-time attention
  are deliberately deferred; this run isolates "Stage-2 training tricks
  on top of exp2c" so the delta is interpretable.

## Files

- New: `src/configs/experiment/exp2d_mae_distill_cutmix.yaml`
- Edited: `src/train.py` (added `load_teacher`, three params on
  `train_one_epoch`, KL term in `_forward_loss`, teacher build in `main`,
  added `import torch.nn.functional as F`).
- Reuses: `src/configs/model/vit_mae_transformer.yaml`, `src/utils.py`
  (`mixup_batch`, `cutmix_batch`, `ModelEMA`).
- Teacher: `checkpoints/exp2c_mae_transformer.pt` (frozen, EMA snapshot).
- Output: `checkpoints/exp2d_mae_distill_cutmix.pt`.

## How to run

Smoke test first:

```bash
cd /Data/challenge_sb
python src/train.py experiment=exp2d_mae_distill_cutmix \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  training.num_workers=2 training.resume=false wandb.enabled=false
```

Look for these log lines confirming both pieces wired correctly:

```
[train] loading teacher from /Data/challenge_sb/checkpoints/exp2c_mae_transformer.pt
[train] distillation enabled (alpha=0.5, T=4.0)
```

Full run inside tmux:

```bash
tmux new -s exp2d "python src/train.py experiment=exp2d_mae_distill_cutmix 2>&1 | tee logs/exp2d_mae_distill_cutmix.log"
```

Evaluate / submit after training:

```bash
python src/evaluate.py training.checkpoint_path=checkpoints/exp2d_mae_distill_cutmix.pt
python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2d_mae_distill_cutmix.pt \
  dataset.submission_output=submissions/exp2d_mae_distill_cutmix_submission.csv
```

## Expected behaviour

- Throughput: same student forward as exp2c plus a frozen teacher forward
  on every batch — wall-clock per epoch should be ~1.4–1.7× exp2c. On the
  A5000, observed ~11 it/s during the warmup window (vs ~14 it/s in exp2c
  smoke), implying ~3.5–4h for 100 epochs.
- Train_loss is the (1−α)·CE + α·T²·KL combination, so the absolute scale
  is *not* directly comparable to exp2c's curve. Trust val/best.
- val/best lands above exp2c's `0.5868`. Conservative band **0.595–0.610**
  (CutMix ~+0.5–1.0 pp, distillation ~+1–2 pp, with some interaction).
- If `val/best ≤ 0.59`, distillation/CutMix are not the bottleneck on top
  of exp2c, and the next move shifts to architecture (exp3 — iBOT, or a
  joint space-time temporal head) rather than more training tricks.
- If `val/best ≥ 0.61`, exp2d becomes the new Stage-2 baseline and we
  ablate the two axes in exp2e (CutMix only) and exp2f (distillation
  only) to attribute the gain.

## Status

Launched **2026-04-27** under `tmux new -s exp2d "python src/train.py
experiment=exp2d_mae_distill_cutmix 2>&1 | tee logs/exp2d_mae_distill_cutmix.log"`.
Smoke test passed: teacher loaded, distillation log line printed, mixup/cutmix
alternation works, 1-epoch run finished cleanly. Results section to be
appended once training finishes.
