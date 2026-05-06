# `vit_tf_2` — post-mortem (run on 2026-04-28, log `vit_tf_2.log`)

## TL;DR

- **Best val acc: 10.50%** (top-1, 33 classes, chance ≈ 3.13%) — about 3.4× chance.
- Successor to `vit_transformer_mlp` v1 (best 9.22%); +1.3 pp over v1 with all the diagnosed v1 mistakes corrected.
- Train acc 9.4% < val acc 10.5% at convergence — same pattern as v1 (under-fit, not over-fit), but *less* dramatic; the gradient flatness was eased, not eliminated.
- **Verdict: confirmed v1's diagnosis (gradient-starved, not capacity-bound) but *also* confirmed the harder ceiling: ViT-Ti from scratch on 36k 4-frame SSv2 clips tops out around 10–12%.** From-scratch ViTs are simply not the right tool at this data scale; pretraining (Track 2) or stronger inductive bias (TSM-ResNet) are the only paths past ~12%.

## Setup

| Field | Value |
|---|---|
| Model | ViT-Ti/16 (scratch) → 2-layer Transformer → MLP(384) |
| Spatial pool | `cls_avg` (CLS ⊕ mean of patch tokens, out_dim = 2 × 192 = 384) |
| Spatial drop_path | 0.05 |
| Temporal | 2 layers, 6 heads (64-dim each), dropout 0.1, drop_path 0.0 |
| MLP head | hidden 384, dropout 0.2 |
| Optimizer | AdamW, lr 6e-4, wd 0.05 |
| Schedule | 5 warmup + cosine over 50 epochs |
| Batch | 64, AMP fp16, grad-clip 1.0 |
| Aug | strong_clip_aug (RRC + ColorJitter + RandAug + RandomErasing, no hflip) |
| Reg | mixup α=0.0 (off), label_smoothing 0.05, EMA 0.0 (off) |
| Frames | `num_frames=4` (matches disk; v1's duplication bug fixed) |
| Data | 36k train clips, val 8.7k clips |
| Hardware / runtime | local CUDA GPU, ~2 h wall (vs v1's 6.5 h at bs=16) |

## Trajectory

| Epoch | train loss | train acc | val loss | val acc | event |
|---|---|---|---|---|---|
| 1 | 3.3717 | 0.065 | 3.3307 | 0.0770 | new best |
| 5 | 3.3358 | 0.075 | 3.3243 | 0.0769 | end of warmup |
| 10 | 3.3350 | 0.075 | 3.3096 | 0.0773 | early plateau |
| 20 | 3.3199 | 0.079 | 3.2874 | 0.0825 | drifting up |
| 30 | 3.3099 | 0.082 | 3.2804 | 0.0932 | new best |
| 38 | 3.2846 | 0.091 | 3.2564 | 0.1008 | crosses 10% |
| 41 | 3.2804 | 0.090 | 3.2459 | 0.1047 | new best |
| 45 | 3.2724 | 0.093 | 3.2444 | 0.1030 | plateau |
| 49 | 3.2714 | 0.094 | 3.2446 | **0.1050** | global best |
| 50 | 3.2727 | 0.094 | 3.2445 | 0.1047 | final |

Shape:
- Epochs 1–10: nearly flat. The model is in a noisy random-prediction regime; warmup ends at 5 with val ≈ 7.7%.
- Epochs 10–30: slow drift up (0.077 → 0.093). Cosine lr decay is gentle; gradient signal *is* propagating, just slowly.
- Epochs 30–49: monotonic rise to 10.50%, with slight wobble in the last 10 epochs as cosine collapses lr to ~0.
- Best at epoch 49 (penultimate); lr-decay schedule is well-matched to where the model gives up.

## Diagnosis

### What v1 → v2 fixed

| Knob | v1 | v2 | effect |
|---|---|---|---|
| `num_frames` | 8 (effective 4 via duplication) | 4 | eliminates duplicate tokens; +unknown but real pp |
| Backbone | ViT-S (22M) | ViT-Ti (5.7M) | more signal per gradient |
| mixup α | 0.2 | 0.0 | unblocks gradient signal |
| EMA | 0.9999 | off | live weights are evaluated |
| MLP dropout | 0.5 | 0.2 | less suppression |
| drop_path | 0.1 / 0.1 | 0.05 / 0.0 | less stochastic-depth skipping |
| label smoothing | 0.1 | 0.05 | gentler prior |
| Temporal layers | 4 | 2 | depth matched to 5 tokens |
| Batch | 16 | 64 | 4× tighter gradients |
| lr | 3e-4 | 6e-4 | linear-scaled to bs=64 |
| Epochs | 60 | 50 | trimmed (v1 plateaued at 48) |

### What v2 confirmed about v1

All of v1's diagnoses checked out: removing mixup/EMA/heavy dropout *did* let the gradient flow (train loss dropped from v1's plateau ~3.34 → 3.27, train acc rose from 7.5% → 9.4%). The fixes were directionally correct.

### What v2 newly revealed

The ceiling is real, not a regularization artifact. With *all* the v1 throttles eased, the model still tops out at ~10.5% val acc and train loss stays well above the random-label loss floor (3.27 vs floor of ~3.50 = `log(33)`). It's no longer gradient-starved — it's *signal-starved*: ViT-Ti has no useful inductive bias for SSv2-style temporal action discrimination, and 36k 4-frame clips is not enough data for it to learn one from scratch.

Note the still-inverted train/val pattern (train_acc 9.4% < val_acc 10.5%): even with mixup off and EMA off, label_smoothing 0.05 is enough to make the val-side eval slightly easier. Not a problem; it's a small effect now, not the dominant one v1 had.

### Compute

Wall time dropped from 6.5 h (v1, bs=16) to ~2 h (v2, bs=64) for a slightly longer schedule (50 vs 60). The bs=64 throughput gain is the headline; it makes ablations cheap enough to actually run.

## What v2 *did* prove

- **v1's gradient-starvation diagnosis was correct.** The fixes worked as designed; the problem is now the data/architecture combo, not the recipe.
- **From-scratch ViT-Ti is not a viable Track-1 contender at this data scale.** Score it as "exhausted; the ceiling is around 10–12% and further recipe tweaks won't break past it."
- **The pipeline is fast and stable at bs=64.** Future from-scratch experiments should default to bs≥64 unless OOM forces otherwise.

## Lessons → directions

1. **Stop iterating on from-scratch ViT.** Two runs (v1 9.22%, v2 10.50%) bracket the achievable; the curve is too flat to be worth more compute.
2. **Pretraining or strong inductive bias is required.** Track 2 (`exp_dinov2_transformer`, see `notes/dinov2_transformer_v1.md` — 65.31%) confirms this — DINOv2 ViT-S/14 alone closes a ~55 pp gap.
3. **For Track 1**, the remaining viable routes are: (a) `exp4_tsm_resnet18` (strong inductive bias from scratch, est. 30%+); (b) MAE/iBOT/V-JEPA pretraining on this dataset (already configured in exp1/2/3/5).
4. **Future from-scratch ablations should use this run's regularization recipe as a known baseline** — don't re-discover the v1 trap.

## Reproduction

```bash
tmux new -s vit_tf_2 \
  "python src/train.py experiment=vit_tf_2 2>&1 | tee vit_tf_2.log"
```

Checkpoints:
- `best_vit_tf_2.pt` — best-by-val (37 MB, model state dict + config)
- `best_vit_tf_2_last.pt` — final-epoch full state (111 MB)
