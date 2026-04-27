# exp7_tsm_r50_mlp_aux_long — changelog

Capacity + optimization + schedule changes on top of `exp5_tsm_r50_transformer`.
**Reg recipe held constant** (the post-mortem of exp6 in
[`exp6_tsm_r50_lighter_reg_results.md`](exp6_tsm_r50_lighter_reg_results.md)
showed exp5's reg package is load-bearing — touching it broke things).

## Why

Final state of `exp5` (epoch 150):

- best EMA `val/acc = 0.5797` (peaked at ep 100, drifted slightly after)
- `train/acc = 0.504`, `train/loss = 1.24`
- `val/loss = 2.14`

Final state of `exp6` (lighter-reg attempt, epoch 150):

- best EMA `val/acc = 0.4857` — **regression of 9.4 pt vs exp5**
- `train/acc = 0.329`, `train/loss = 1.81` — **fits train worse, not better**
- → the live model in exp5 wasn't choked by reg. The bottleneck is **backbone
  optimization** (from-scratch R50 with sparse temporal gradient at T=4) and
  **head capacity** (single Linear on the 512-d transformer pooled vector).

exp7 attacks those two bottlenecks while keeping the reg recipe that exp5
proved works.

## What changed vs. exp5_tsm_r50_transformer

Four orthogonal axes move at once. They don't touch each other (capacity, LR
groups, supervision target, schedule are independent), so a future ablation
can peel them back one at a time to attribute.

| field | exp5 | **exp7 (this)** | why |
|---|---|---|---|
| `model.classifier.name` | `linear` | **`mlp` (hidden=512, dropout=0.3)** | non-linear decision boundary on the 512-d pooled vector; +0.27M params |
| `training.lr` (head) | 5e-4 (single) | **5e-4** | unchanged — head/temporal/classifier need full LR |
| `training.backbone_lr` | — | **2e-4** | from-scratch R50 with AdamW + AMP wants a quieter LR; stabilizes BN/AMP, head still adapts fast |
| `training.aux_loss.enabled` | — | **true (weight 0.3)** | per-frame Linear on `(B,T,d)` spatial features; CE w/ same label, mixup-compatible. Feeds direct gradient into every frame of the backbone (main loss only sees 5 transformer tokens at T=4) |
| `training.epochs` | 150 | **220** | exp5 peaked at ep 100, then 50 ep of LR→0 with no further improvement. Stretching the cosine moves the high-LR phase right (LR ≥ 1e-4 lasts to ~ep 90 instead of ~ep 60) |
| `training.warmup_epochs` | 10 | **15** | MLP head + 2 LR groups + aux head need a longer ramp before cosine kicks in |

**Reg knobs — all unchanged from exp5 (deliberate):**
`label_smoothing=0.1`, `mixup_alpha=0.2`, `weight_decay=0.05`,
`temporal.dropout=0.15`. Touching these is what broke exp6.

**Other unchanged:** TSM-R50 backbone (`pretrained=false`), 2-layer transformer
head with `in_proj_dim=512`, `num_frames=4` (hard cap), `batch_size=16`,
AdamW, EMA (decay=0.9995), `strong_clip_aug=true`, `grad_clip=1.0`, `amp=true`,
cosine schedule.

## What's new in the codebase

- `src/models/modular.py`: added `AuxFrameHead`, `attach_aux_head`,
  `ModularVideoModel.forward_with_aux` returning `(main_logits, aux_logits)`.
  `aux_head` defaults to `None`; existing experiments are byte-identical at
  runtime.
- `src/configs/model/tsm_resnet50_transformer_mlp.yaml`: same as
  `tsm_resnet50_transformer.yaml` but `classifier.name: mlp`.
- `src/train.py`: builds the aux head before EMA so EMA mirrors it. Combined
  loss = `main_CE + w * mean_t CE(per_frame_logits, label)`, mixup-compatible
  (same `(λ, y_a, y_b)` mix replicated across frames). Eval path is
  unchanged — only the main classifier is used at val/test, so val_acc is
  directly comparable to exp5/exp6.

## What to watch in wandb / log

- `train/acc` should clear **0.55** by ep 100 (exp5 was at 0.39 there). If
  it's still at ~0.40 by ep 100, the aux loss may be dominating and biasing
  the backbone toward per-frame discriminability over temporal reasoning →
  drop `aux_loss.weight` to 0.1 or anneal it down.
- `val/acc` (EMA) should beat **0.5797** somewhere between ep 100 and ep 180.
  If it stalls below 0.55 by ep 60, the change set is net-harmful — kill
  the run (don't let the cosine play out).
- `val/loss` was diverging late in exp5 (2.14 vs train 1.24). Watch whether
  it stays closer to train_loss with the longer schedule.

## Sanity gates

- **Smoke (already passed):**
  ```
  python src/train.py experiment=exp7_tsm_r50_mlp_aux_long \
    dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
    wandb.enabled=false
  ```
  Confirmed: aux head wired, 2-group LR active (4 param groups), AMP+EMA
  stable, no NaN.

- **30-epoch gate:** if `Epoch 30/220 | ... val acc` is below **0.354**
  (exp5's ep-30 number), abort with `tmux kill-session -t exp7`. The change
  set is net-harmful and finishing 220 epochs won't recover it.

## Run

```bash
# Full run (tmux — required, SSH-drop safe)
tmux new -s exp7 "python src/train.py experiment=exp7_tsm_r50_mlp_aux_long 2>&1 | tee exp7.log"
# detach: Ctrl-b d   reattach: tmux attach -t exp7
```

Checkpoints: `checkpoints/exp7_tsm_r50_mlp_aux_long.pt` (best by EMA val acc),
`checkpoints/exp7_tsm_r50_mlp_aux_long_last.pt` (resume).

## Rough expected impact

- All four changes pull the right way: `train_acc` 0.55–0.65, best EMA
  `val_acc` **0.60–0.63** (+2 to +5 over exp5).
- Only head/LR help, aux is neutral: ~0.59–0.61.
- Lands at ~0.58: bottleneck is genuinely backbone pretraining; next move is
  iBOT/V-JEPA pretrained features (already on roadmap, remote).

## Cost

220 ep × ~5 min/ep ≈ **18 hours** wall-clock on the local CUDA GPU
(extrapolated from exp5's ~12 h for 150 ep). The 30-ep gate at ~3 h means we
risk at most 3 hours if the change set is net-harmful.
