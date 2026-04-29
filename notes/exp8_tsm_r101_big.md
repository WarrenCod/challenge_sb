# exp8_tsm_r101_big — changelog

Scale-up of [`exp7_tsm_r50_mlp_aux_long`](exp7_tsm_r50_mlp_aux_long.md) along
**four orthogonal axes** at once. First Track-2 entry in the TSM family —
ImageNet-pretrained backbone is now in scope (user switched to Track 2 on
2026-04-29).

Reg recipe is held **constant at exp7's settings** (label-smoothing 0.1,
mixup 0.2, weight-decay 0.05, temporal dropout 0.15, EMA 0.9995, strong clip
aug, aux-loss weight 0.3) so any delta vs exp7 attributes to the four
capacity / scale changes below, not to a regularization swing.

## Why

Pre-exp8 standings on val:

| run                              | best EMA val_acc | epoch  | track |
| -------------------------------- | ---------------- | ------ | ----- |
| exp5 (R50, transformer, linear)  | 0.5797           | ~100   | 1     |
| exp6 (lighter reg)               | 0.4857           | ~139   | 1     |
| **exp7 (R50, MLP+aux, 220 ep)**  | **0.5760**       | ~?     | 1     |

exp7's 4-knob capacity bundle on top of exp5 was a **wash** (−0.4 pt). That
matches the post-mortem hypothesis from
[`exp6_tsm_r50_lighter_reg_results.md`](exp6_tsm_r50_lighter_reg_results.md):
the bottleneck on this dataset isn't head capacity or schedule length; it's
**backbone representations** — a from-scratch R50 with sparse temporal
gradient (T=4 hard cap) cannot extract enough action-relevant features in
~12 h on 24k clips, no matter what head sits on top.

exp8 attacks that bottleneck head-on:

- **ImageNet-pretrained R101 backbone** — Track 2 unlocks this. Skips the
  hardest part of the from-scratch problem (early conv stages learning
  generic visual features) and lets every training step go to the harder
  question (temporal reasoning + SSv2-specific feature shaping).
- **Bigger backbone (R101 vs R50)** — +20M params, deeper hierarchy, better
  at fine-grained motion cues that SSv2 keys on.
- **Higher input resolution (256 vs 224)** — small-object hand/object
  manipulations that SSv2 labels turn on are pixel-level. +14% spatial area.
- **Bigger temporal head (4 layers, d=768, ff=3072)** — to actually use the
  richer per-frame features the bigger backbone produces.
- **Multi-clip TTA at inference** — separate to training; +0.5–1.5 pt is
  typical at T=4 from clip-sampling variance reduction.

## What changed vs. exp7

| field                              | exp7                | **exp8 (this)**         | why |
| ---------------------------------- | ------------------- | ----------------------- | --- |
| `model.spatial.variant`            | resnet50            | **resnet101**           | +20M params, deeper hierarchy |
| `model.spatial.pretrained`         | false (Track 1)     | **true (ImageNet)**     | Track 2; skips from-scratch low-level vision |
| `dataset.image_size`               | 224                 | **256**                 | +14% spatial area; SSv2 cues are pixel-level |
| `model.temporal.in_proj_dim`       | 512                 | **768**                 | wider transformer to absorb bigger backbone |
| `model.temporal.num_layers`        | 2                   | **4**                   | deeper temporal stack |
| `model.temporal.dim_feedforward`   | 2048                | **3072**                | scaled with `in_proj_dim` |
| `training.backbone_lr`             | 2.0e-4              | **1.0e-4**              | pretrained → standard fine-tuning rate, half exp7's |
| `training.lr` (head)               | 5.0e-4              | 5.0e-4                  | unchanged — head still learns from scratch |
| `training.batch_size`              | 16                  | **8**                   | R101 + 256² + T=4 ≈ 2.5× exp7 footprint |
| `inference (create_submission.py)` | single 4-frame clip | **4-clip temporal TTA** | average logits across 4 random 4-frame clips per video |

**Held constant from exp7 (deliberate):**
`label_smoothing=0.1`, `mixup_alpha=0.2`, `weight_decay=0.05`,
`temporal.dropout=0.15` (slight bump — see note), `ema_decay=0.9995`,
`strong_clip_aug=true`, `aux_loss.enabled=true (weight 0.3)`,
`epochs=220`, `warmup_epochs=15`, cosine schedule, AdamW, `grad_clip=1.0`,
`amp=true`, `num_frames=4` (hard cap).

> *Note on dropout:* `temporal.dropout` is 0.15 in both exp7 and exp8. The
> exp8 config sets it explicitly because the deeper transformer (4 vs 2
> layers) compounds dropout — same per-layer rate, more layers between
> input and output. If train_loss drops too fast late, this is the first
> knob to nudge up to 0.2.

## What's new in the codebase

- `src/models/spatial/resnet_tsm.py` — accepts `variant=resnet101`
  (already supported via the torchvision branch); `pretrained=True` loads
  ImageNet weights and TSM is folded into the right BasicBlock conv layers.
- `src/configs/data/default.yaml` — `image_size: 256` flows through to the
  train/val transforms in `src/utils.py`. ImageNet normalization stays on
  whenever `pretrained=true`.
- `src/create_submission.py` — multi-clip TTA path: 4 random 4-frame clip
  draws per test video, softmax-average the logits, argmax.
- `src/configs/experiment/exp8_tsm_r101_big.yaml` — single config bundling
  all four scale axes plus the TTA toggle.

The model registry, EMA handling, aux-loss wiring, and 2-group LR builder
are unchanged from exp7. Eval path uses only the main classifier (aux head
is training-only), so val_acc here is directly comparable to exp5/exp6/exp7.

## What to watch in wandb / log

- **GPU memory** — at bs=8, R101 + 256² + T=4 + AMP should land near
  ~16-18 GB on the A5000 (24 GB). If it OOMs, drop to bs=4 and double
  `aux_loss.weight` halving impact via grad accumulation is *not*
  configured — just lower the batch.
- **Iter speed** — exp7 ran ~3.5 it/s at bs=16. Expect ~1.5-2 it/s at bs=8
  here (fewer samples/iter but each sample is heavier). Per-epoch wall clock
  should land near 5-6 min, full run ≈ 18-22 h.
- **train/acc** vs **val/acc(EMA)** — pretrained backbones often show a much
  smaller train-val gap early than from-scratch. If train_acc rockets past
  0.7 by ep 30 while val stays near 0.45, the head is overfitting on top of
  frozen-ish features → the backbone LR may be too low (raise to 1.5e-4 in
  exp9).
- **First validation jump** — pretrained R101 typically lifts ep-1 val_acc
  to **0.20-0.30** out of the gate (vs ~0.05 from-scratch). If ep 1 lands
  below 0.10, something is wrong with the ImageNet weight load — check the
  log for `Downloading: ...resnet101...` or a `missing_keys` warning.

## Sanity gates

- **Smoke (passed):**
  ```
  python src/train.py experiment=exp8_tsm_r101_big \
    dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
    wandb.enabled=false
  ```
  Confirmed: ImageNet R101 weights downloaded (171 MB), aux head wired,
  4 param groups for the 2-group LR, AMP+EMA stable, no NaN, ep-1 train
  loss 4.59 → 3.81, no OOM. The first stale `_last.pt` from this smoke run
  caused a config-hash mismatch on the full launch — re-launched with
  `training.resume=false` to overwrite cleanly.

- **30-epoch gate:** if `Epoch 30/220 | ... val acc` is below **0.45**
  (a generous floor — pretrained R101 + TSM should clear this comfortably;
  exp5/from-scratch hit 0.354 here), abort with
  `tmux kill-session -t exp8`. Either the pretraining isn't transferring
  (check ImageNet weight load) or the optimization is diverging.

- **100-epoch gate:** if best EMA val_acc is below **0.58** (exp5's peak)
  by ep 100, the scale-up isn't paying for itself and it's worth
  killing — running the cosine to ep 220 won't recover a 220-epoch run
  that's underperforming the smaller model at its own peak.

## Run

```bash
# Full run (tmux — required, SSH-drop safe)
tmux new -s exp8 "python src/train.py experiment=exp8_tsm_r101_big training.resume=false 2>&1 | tee exp8.log"
# detach: Ctrl-b d   reattach: tmux attach -t exp8
```

(`training.resume=false` is needed once because the smoke test left a stale
`_last.pt` with a different config hash. Drop it on subsequent re-launches
once a real `_last.pt` exists.)

Checkpoints:
- `checkpoints/exp8_tsm_r101_big.pt` — best by EMA val acc (used by
  evaluate.py / create_submission.py)
- `checkpoints/exp8_tsm_r101_big_last.pt` — last-epoch state (resume)

## Rough expected impact

The four changes pull the same direction (more capacity / better
features / better inference). Plausible bands:

- **All four pay off cleanly:** best EMA val **0.66-0.70** (+8 to +12 pt
  over exp7). This is the optimistic case where pretrained features slot
  cleanly into the SSv2 task and the bigger temporal head uses them.
- **Pretraining is the only meaningful win:** **0.62-0.65** (+4 to +7).
  Backbone width, resolution, and temporal depth contribute marginally.
- **Pretraining helps but the deeper temporal head overfits at T=4:**
  **0.59-0.62** (+1 to +4). In this case exp9 should drop temporal back to
  2 layers and isolate the backbone change.
- **Wash / regression:** below 0.58. Most likely cause is the bigger
  temporal head overfitting to 4 tokens — peel temporal back to exp7's
  2-layer / 512-d / 2048-ff in exp9 and keep R101+256+pretrained.

## Cost

220 ep × ~5-6 min/ep ≈ **18-22 hours** wall-clock on the local A5000.
(Smoke-test extrapolated: 4500 batches/epoch at ~2 it/s ≈ 38 min/ep at
warm-up speed; once AMP/JIT settles, expect ~5-6 min.) The 30-ep gate at
~3 h means we risk at most 3 hours if the change set is net-harmful.

---

# Results

*(filled in after the run completes)*

## TL;DR

TBD.

## Headline numbers

| metric                       | exp7 (R50, T=2)    | **exp8 (R101, T=4)**  | Δ        |
| ---------------------------- | ------------------ | --------------------- | -------- |
| best EMA val_acc             | 0.5760             | TBD                   | TBD      |
| epoch of best                | TBD                | TBD                   | —        |
| final val_acc (ep 220)       | 0.5750             | TBD                   | TBD      |
| final train_acc (ep 220)     | 0.5330             | TBD                   | TBD      |
| final train_loss (ep 220)    | 1.5233             | TBD                   | TBD      |
| final val_loss (ep 220)      | 2.4447             | TBD                   | TBD      |

## Trajectory snapshot

| epoch | exp7 train loss / acc | exp7 val_acc | exp8 train loss / acc | exp8 val_acc |
| ----- | --------------------- | ------------ | --------------------- | ------------ |
|  10   | TBD                   | TBD          | TBD                   | TBD          |
|  30   | TBD                   | TBD          | TBD                   | TBD          |
|  50   | TBD                   | TBD          | TBD                   | TBD          |
|  75   | TBD                   | TBD          | TBD                   | TBD          |
| 100   | TBD                   | TBD          | TBD                   | TBD          |
| 150   | TBD                   | TBD          | TBD                   | TBD          |
| 200   | TBD                   | TBD          | TBD                   | TBD          |
| 220   | TBD                   | TBD          | TBD                   | TBD          |

## What attribution can / cannot be made

Four axes moved at once (backbone size, pretraining, resolution, temporal
depth). If exp8 wins, the contribution split is **not identified** from
this run alone. Single-axis ablations to peel back, ranked by suspected
contribution:

1. **exp9a:** drop R101 → R50 (keep pretrained=true, 256², T=4 head). Tests
   whether width matters above pretraining.
2. **exp9b:** drop image_size 256 → 224 (keep R101, pretrained, T=4 head).
   Tests resolution gain.
3. **exp9c:** drop temporal head 4-layer/768 → 2-layer/512 (keep R101,
   pretrained, 256). Tests whether the deeper temporal stack is helping or
   just adding parameters.
4. **exp9d:** turn off TTA in `create_submission.py`. Cheap — just a flag
   at submission time. Tests TTA contribution to the leaderboard score
   (val/acc here is single-clip already).

Run order if exp8 wins: 9a first (biggest suspected contributor), then 9c
(highest overfit risk), then 9b. 9d is a 5-minute test, do it last.

## Diagnosis (if exp8 underperforms)

- **best EMA val < 0.55 by ep 100:** most likely the deeper temporal head
  (4 layers on 4 tokens) is overfitting. Peel to exp9c first.
- **train_acc << val_acc throughout:** unusual for pretrained — would mean
  the backbone is locked too cold by `backbone_lr=1e-4`. Try 1.5e-4 or
  2e-4 in exp9.
- **train_acc >> val_acc by ep 50:** the pretrained backbone is memorizing.
  Bump `temporal.dropout` to 0.2 and/or `weight_decay` to 0.07.
- **NaN / loss explosion:** AMP + R101 + bs=8 can occasionally trigger this
  on the first warmup epochs. Drop `lr` peak by 30% and re-launch.

## Run / artifact pointers

- Log: `exp8.log`
- Best checkpoint: `checkpoints/exp8_tsm_r101_big.pt`
- Last checkpoint: `checkpoints/exp8_tsm_r101_big_last.pt`
- Pre-run rationale: this file (sections above the `# Results` heading)
- Compare:
  - `exp7.log`, `checkpoints/exp7_tsm_r50_mlp_aux_long.pt` (immediate parent)
  - `exp5.log`, `checkpoints/exp5_tsm_r50_transformer.pt` (Track-1 high-water mark)
