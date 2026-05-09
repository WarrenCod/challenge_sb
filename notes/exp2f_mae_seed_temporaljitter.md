# exp2f_mae_seed_temporaljitter — proposal

A diversity-only successor to `exp2d_mae_distill_cutmix`. Same architecture,
same teacher, same recipe — but **(a) a fresh RNG seed** and **(b) random
temporal jitter at training time**, two independent sources of structurally
new variance. The point is not to beat exp2d/exp2e on val; it is to produce a
fourth model whose **test-time errors are uncorrelated** with the existing
{exp2c, exp2d, exp2e} ensemble, so adding it to the ensemble pushes test
accuracy further.

## Why now

- Single-model val is saturated. exp2c→2d added +1.42 pp. 2d→2e added only
  +0.14 pp. Squeezing more from a single student is high-effort, low-payoff.
- The **test-time ensemble was the move that worked**. Pairwise disagreement
  among {2c, 2d, 2e} on the test set is 31–39 %, and the uniform softmax
  ensemble overrules each individual model on 19–25 % of clips. Adding a
  fourth model with **independent** errors should compound — same way
  2c+2d disagreed and 2c+2d+2e disagreed.
- We've never spent a training run on **temporal-offset augmentation**.
  Every prior Track-1 run uses `offset_frac=0.0` at train time, so all three
  current models see the exact same temporal sampling per video. That is
  the largest unexploited diversity axis remaining without changing the
  architecture.

## What changes vs exp2d

| field | exp2d | **exp2f (this)** | why |
|---|---|---|---|
| `dataset.seed` | `42` | **`1337`** | Different RNG → different init, batch order, mixup/cutmix masks, drop_path masks. Cheapest source of error decorrelation. |
| dataset temporal sampling at train time | fixed `offset_frac=0.0` | **uniform random `offset_frac` per item per epoch** | Genuine new aug axis. Each video is seen at slightly different temporal alignments across epochs, producing decision boundaries that focus on different sub-events. |
| teacher | exp2c (single) | **exp2c (single, unchanged)** | exp2e showed that ensemble-teacher buys ~0 pp; pick the simpler, faster recipe. Also keeps exp2f cleanly distinct from exp2e on the *teacher* axis (exp2f sees only exp2c soft targets, exp2e was distilled from {2c,2d}). |
| `training.distill.alpha` | `0.5` | `0.5` | unchanged |
| `training.distill.temperature` | `4.0` | `4.0` | unchanged |
| `training.epochs` | `100` | `100` | unchanged. exp2d's plateau at ep 83 was real; no reason to extend. |
| EMA, drop_path, dropout, LLRD, lr, WD, mixup, cutmix, label smoothing, num_frames, batch_size | exp2d values | **identical** | Single-axis discipline. Two changes (seed + temporal jitter) are already two; don't muddy. |

Total change: 2 axes (seed, temporal jitter), both targeting prediction
diversity rather than peak val accuracy.

## What this is *not*

- Not a born-again iteration on exp2e. exp2e showed compressing the
  ensemble back into one student is wasted complexity at this scale.
- Not a teacher-axis change. We've thoroughly characterized that axis
  (single 2c → ensemble 2c+2d).
- Not an architecture change. Reserved for the iBOT-init pivot or for
  Track 2 work (rouloul branch).
- Not a longer schedule. We have evidence longer schedules don't help here.

## Implementation

Two small edits, both backwards-compatible.

### 1. `src/dataset/video_dataset.py` — add training-time random offset

Add a `random_offset_frac: bool = False` flag to `VideoFrameDataset.__init__`.
When `True`, `__getitem__` ignores `self.frame_offset_frac` and samples
`offset_frac ~ Uniform[0, 1)` per call. This is independent of (and
mutually exclusive with) the deterministic TTA path used by
`create_submission.py`. ~5 lines.

### 2. `src/train.py` — wire a config flag through

Read `cfg.dataset.random_temporal_offset` (default `False`) and pass it as
`random_offset_frac=...` only to the **train** dataset. Val dataset
unchanged (deterministic eval). ~3 lines.

### 3. `src/configs/experiment/exp2f_mae_seed_temporaljitter.yaml`

```yaml
# @package _global_
# exp2f (Track 1): exp2d recipe + fresh seed + train-time random temporal offset.
# Goal is ensemble diversity, not single-model peak.
defaults:
  - override /model: vit_mae_transformer

model:
  spatial:
    drop_path: 0.2
  temporal:
    dropout: 0.2

dataset:
  num_frames: 4
  seed: 1337
  random_temporal_offset: true

training:
  batch_size: 16
  lr: 3.0e-4
  llrd: 0.75
  epochs: 100
  warmup_epochs: 8
  weight_decay: 0.05
  label_smoothing: 0.1
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
  strong_clip_aug: true
  amp: true
  grad_clip: 1.0
  ema: true
  ema_decay: 0.9999
  num_workers: 8
  checkpoint_path: ${hydra:runtime.cwd}/checkpoints/exp2f_mae_seed_temporaljitter.pt
  device: cuda
  distill:
    teacher_ckpt: ${hydra:runtime.cwd}/checkpoints/exp2c_mae_transformer.pt
    alpha: 0.5
    temperature: 4.0

wandb:
  enabled: true
  run_name: exp2f_mae_seed_temporaljitter
  tags: [track1, stage2, mae, transformer, regularized, ema, cutmix, distill, seed, temporaljitter, ensemble_diversity]
```

## Cost

- **Throughput:** identical to exp2d (one student forward + one teacher
  forward, exp2c only). ~6h22 on the A5000.
- **Inference:** add a TTA-3 + dump_probs run (~15 min) and a 4-way
  ensemble run (~5 s).

## Decision rules

- **val/best alone is not the target.** Acceptance is whether *adding
  exp2f to the ensemble* moves the leaderboard.
- ✓ exp2f val_best ∈ [0.595, 0.605] **and** 4-way ensemble disagrees
  with the 3-way ensemble on ≥ 12 % of test clips → keep, submit
  `ensemble_2c2d2e2f`.
- ◇ exp2f val_best < 0.59 → exp2f underperforms enough to drag the
  ensemble; drop it from the ensemble (or down-weight it).
- ✗ 4-way ensemble disagrees with 3-way on < 5 % of test clips →
  exp2f is too correlated with the existing models; not worth keeping.
  This would say train-time temporal jitter alone is not a strong enough
  diversity axis, and the next move should be an *architectural* axis
  (e.g., a different temporal head) or wait for iBOT.

## How to run

Smoke test first (must show train-time random offset):

```bash
cd /Data/challenge_sb
python src/train.py experiment=exp2f_mae_seed_temporaljitter \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  training.num_workers=2 training.resume=false wandb.enabled=false
```

Full run inside tmux:

```bash
tmux new -s exp2f "python src/train.py experiment=exp2f_mae_seed_temporaljitter 2>&1 | tee logs/exp2f_mae_seed_temporaljitter.log"
```

After training, mirror the exp2c/d/e flow to add it to the ensemble:

```bash
python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2f_mae_seed_temporaljitter.pt \
  +submission.tta_clips=3 +submission.dump_probs=true \
  dataset.submission_output=submissions/exp2f_mae_seed_temporaljitter_tta3_submission.csv

python src/ensemble_submission.py \
  --inputs submissions/exp2c_mae_transformer_tta3_submission.csv \
           submissions/exp2d_mae_distill_cutmix_tta3_submission.csv \
           submissions/exp2e_mae_bornagain_ensembleteacher_tta3_submission.csv \
           submissions/exp2f_mae_seed_temporaljitter_tta3_submission.csv \
  --output submissions/ensemble_2c2d2e2f_tta3_submission.csv
```
