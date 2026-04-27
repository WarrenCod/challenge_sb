# exp6_tsm_r50_lighter_reg — results & post-mortem

Companion to [`exp6_tsm_r50_lighter_reg.md`](exp6_tsm_r50_lighter_reg.md) (the
pre-run changelog). This file is the **post-run reading**.

## TL;DR

The lighter-reg config **underperformed** exp5 by ~9 pts on val and ~17 pts on
train, even though both ran 150 epochs on the same arch / optimizer / schedule.
The diagnosis behind exp6 ("exp5 underfits — lighten reg to let train_acc
climb") was wrong. The lighter recipe didn't free up training; it made
optimization noisier and the live model converged to a *worse* basin (higher
train_loss, lower train_acc), and EMA inherited that.

## Headline numbers (both EMA-evaluated, same `evaluate_epoch` path)

| metric                       | exp5 (heavy reg)   | exp6 (lighter reg)  | Δ        |
| ---------------------------- | ------------------ | ------------------- | -------- |
| **best EMA val_acc**         | **0.5797** (ep 100)| **0.4857** (ep 139) | **−0.094** |
| final val_acc (ep 150)       | 0.5605             | 0.4827              | −0.078   |
| final train_acc (ep 150)     | 0.5044             | 0.3293              | −0.175   |
| final train_loss (ep 150)    | 1.2388             | 1.8098              | +0.571   |
| final val_loss (ep 150)      | 2.1374             | 2.0927              | −0.045   |
| epoch of best                | ~100               | ~139                | later    |

Trajectory snapshot (per-epoch print lines, EMA val):

| epoch | exp5 train_loss / acc | exp5 val_acc | exp6 train_loss / acc | exp6 val_acc |
| ----- | --------------------- | ------------ | --------------------- | ------------ |
|  10   | 3.21 / 0.102          | 0.198        | 3.28 / 0.080          | 0.061        |
|  30   | 2.88 / 0.166          | 0.354        | 3.10 / 0.108          | 0.180        |
|  50   | 2.55 / 0.232          | 0.475        | 2.91 / 0.142          | 0.245        |
|  75   | 2.20 / 0.301          | 0.557        | 2.58 / 0.196          | 0.365        |
| 100   | 1.76 / 0.393          | **0.580**    | 2.21 / 0.258          | 0.448        |
| 125   | 1.37 / 0.488          | 0.559        | 1.89 / 0.303          | 0.476        |
| 150   | 1.24 / 0.504          | 0.561        | 1.81 / 0.329          | 0.483        |

exp5 peaks at ~ep 100 then slightly drifts (mild late overfit, EMA snapshot
already locked). exp6 just lags the entire trajectory and never catches up — it
is **slower, not less overfit**.

## Why is exp6 worse? Three plausible causes, ranked

### 1. The "single-axis ablation" wasn't single-axis (most likely)

Four knobs moved at once:

| field                       | exp5  | exp6  |
| --------------------------- | ----- | ----- |
| `training.label_smoothing`  | 0.10  | 0.05  |
| `training.mixup_alpha`      | 0.20  | 0.10  |
| `training.weight_decay`     | 0.05  | 0.02  |
| `model.temporal.dropout`    | 0.15  | 0.10  |

Each of those is a "lighter reg" change individually, but they don't compose
linearly with **AdamW @ lr 5e-4 on a from-scratch ResNet50**. The
exp5 recipe was a coupled package keeping the loss surface smooth enough for
AdamW to navigate from random init; removing all four cushions at once
exposed a sharper / noisier landscape, the optimizer ping-ponged around a
higher-loss plateau, and the EMA averaged that noise instead of converging.
The smoking gun is **train_loss never dropping below 1.81** (vs 1.24) — that
is the optimizer failing to fit, not the regularizer suppressing fit.

### 2. WD 0.05 → 0.02 was the most load-bearing change

AdamW's decoupled WD shrinks parameter norms every step independently of
gradient magnitude. With from-scratch R50 + AMP, that shrinkage was likely
the dominant *implicit-LR* control, keeping pre-activations bounded so BN
stats stayed sane. Cutting WD by 2.5× lets weight norms grow, BN stats drift,
and effective updates become noisier — which fits the "trains slower, never
converges as far" pattern. (Mixup α and LS are unlikely to single-handedly
cost 17 train-acc points.)

### 3. The original underfitting framing mis-read EMA as ground truth

exp5's train_acc 0.50 << val_acc(EMA) 0.58 was read as "live model is choked
by reg." But that gap is mostly the **EMA averaging giving a smoother
classifier on val**, not the live model being held back. The live exp5 model
already had a cleanly-decreasing train_loss curve (3.21 → 1.24) — a model
that's actively fitting, not one stuck against a regularization wall.

## How to make exp7 beat exp5

Don't rerun the same 4-axis change. Pick the smallest move that respects what
exp5 already proved works.

### Best single bet (recommended): exp7 = exp5 + capacity, not less reg

Hold the entire exp5 reg recipe constant. Add fitting capacity in the only
two places that don't fight regularization:

- **MLP classifier head** after the transformer (Linear → GELU → Dropout(0.1)
  → Linear) instead of the current single Linear. The transformer outputs
  512-d already; an MLP head is ~0.5M params and gives a non-linear decision
  boundary the linear classifier can't.
- **Layer-wise LR decay** on the backbone (e.g. `lr_mult=0.7` per stage from
  head to stem). The transformer head and classifier learn at lr=5e-4; early
  conv stages learn at ~1.7e-4. This is the standard from-scratch fix when
  the backbone bottlenecks: the head adapts faster, the stem adapts more
  conservatively, train_loss drops further without destabilizing.

Expected gain if right: train_acc lands in 0.55-0.65, val 0.59-0.62.

### Second-best bet: aux per-frame loss (deep supervision)

Add a per-frame classifier branch on the TSM features (mean-pooled
spatially), supervised with the same label, weight 0.3-0.5. With T=4
(hard-capped) the temporal head sees only 4-5 tokens; the backbone gets
sparse end-to-end gradient. A per-frame aux loss gives the backbone direct
supervision at every frame, which historically helps short-clip
from-scratch training a lot.

Expected gain: +1-2 pt val. Cheap to add.

### What to do with the reg knobs (if anything)

If you do want to revisit reg, do it **truly single-axis** and small:

- **Only WD 0.05 → 0.03** (one step, not 2.5×). Hold LS=0.1, mixup=0.2,
  dropout=0.15 fixed.
- If that helps, then try mixup 0.2 → 0.15 next, alone.

Never change four reg knobs in the same run again — you can't attribute the
delta and you can't tell which one to undo. (This is the lesson worth
remembering from exp6.)

### Cheap inference-time wins (do regardless)

- **Multi-clip eval.** Sample N=3 random 4-frame clips per video at val/test
  and average logits. With T=4 (hard-capped) this is the standard way to
  reduce eval-time clip-sampling variance. Often worth 0.5-1.5 pt on val,
  free of training cost. Apply to exp5's existing checkpoint first to set a
  stronger baseline before exp7.

### What to **not** do

- Don't extend exp6 with more epochs — LR is already at 6.4e-6 by ep 140 and
  val_acc has been flat for ~30 epochs. The schedule is exhausted; the
  optimizer is parked.
- Don't re-launch exp6 hoping it was unlucky — train_loss at 1.81 at ep 150
  is a structural ceiling for this config, not a seed artifact.
- Don't add more frames; `num_frames=4` is the hard cap (per project memory).

## Run / artifact pointers

- Log: `exp6.log` (~86 MB, full 150 epochs, completed cleanly with
  `Done. Best validation accuracy: 0.4857`)
- Checkpoint: `checkpoints/exp6_tsm_r50_lighter_reg.pt`
- Compare: `exp5.log`, `checkpoints/exp5_tsm_r50_transformer.pt`
- Pre-run rationale (now superseded): [`exp6_tsm_r50_lighter_reg.md`](exp6_tsm_r50_lighter_reg.md)
