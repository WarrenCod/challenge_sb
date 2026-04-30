# exp6_tsm_r50_lighter_reg — changelog

Single-axis ablation on top of `exp5_tsm_r50_transformer`. Same architecture,
same optimizer, same schedule — only the regularization knobs change.

## Why

Final state of `exp5` (wandb run `rlea1h8w`, 150/150 epochs):

- `train/acc = 0.504`
- `val/best_acc (EMA) = 0.5797`
- `train/loss = 1.24`, `val/loss = 2.14`
- `lr` ended at `≈ 2.4e-10` — cosine fully exhausted

`train_acc << val_acc(EMA)` after a full cosine: the live model never fit the
train set. With T=4 (hard-capped), from-scratch R50, and the stack
`label_smooth 0.1 + mixup 0.2 + dropout 0.15 + WD 0.05`, regularization is
choking optimization. EMA is masking this at eval — but the underlying weights
are stuck.

**More epochs would not have helped:** LR was already ~0 at epoch 150. The
question is "does the model fit better when we let it?", not "does it need
more steps?". This run keeps `epochs=150` so the comparison vs exp5 is clean
on a single axis.

## What changed vs. exp5_tsm_r50_transformer

| field | exp5 | **exp6 (this)** | why |
|---|---|---|---|
| `training.label_smoothing` | 0.10 | **0.05** | less target softening → easier to push train_acc up |
| `training.mixup_alpha` | 0.20 | **0.10** | weaker label/input perturbation, model still gets some |
| `training.weight_decay` | 0.05 | **0.02** | typical for from-scratch R50; 0.05 was timm-vit-flavored |
| `model.temporal.dropout` | 0.15 | **0.10** | head was bottlenecking gradient flow on a 5-token sequence |

Unchanged: optimizer `adamw`, `lr=5e-4`, `warmup_epochs=10`, cosine schedule,
`grad_clip=1.0`, EMA decay `0.9995`, `strong_clip_aug=true`, `batch_size=16`,
`epochs=150`, T=4, TSM-R50 backbone (from scratch, `pretrained=false`),
2-layer transformer head with `in_proj_dim=512`, linear classifier.

## What to watch in wandb

- `train/acc` should clear **0.55** well before epoch 100. If it plateaus
  around 0.60 by mid-run, reg is still too tight (or another bottleneck —
  head capacity, optimizer) → next ablation: MLP head + layer-wise LR + aux
  per-frame loss.
- `val/best_acc` should beat **0.5797**. If train climbs but val stalls /
  drops, we overshot the reg-down (back off mixup or WD).
- `val/loss` was diverging late in exp5 (2.14 vs train 1.24). Watch whether
  it now stays closer to train.

## Run / inspect

```bash
# Smoke test (short, foreground)
python src/train.py experiment=exp6_tsm_r50_lighter_reg \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  wandb.enabled=false

# Full run (tmux — required, SSH-drop safe)
tmux new -s exp6 "python src/train.py experiment=exp6_tsm_r50_lighter_reg 2>&1 | tee exp6.log"
# detach: Ctrl-b d   reattach: tmux attach -t exp6
```

Checkpoints: `checkpoints/exp6_tsm_r50_lighter_reg.pt` (best by val acc),
`checkpoints/exp6_tsm_r50_lighter_reg_last.pt` (resume).

## Rough expected impact

If the underfitting diagnosis is right: **+1 to +3 pt val acc** (0.59–0.61
range), with train_acc landing in the 0.65–0.75 band. If exp6 lands at
~0.58 with train_acc still around 0.50, the bottleneck is *not* regularization
— it's backbone optimization from scratch, and the right next move is iBOT /
V-JEPA pretraining (already on roadmap, remote).
