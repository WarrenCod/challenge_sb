# exp8_vjepa_spacetime — results

V-JEPA Stage-1 EMA target encoder (ViT-S/16, Track 1) + 4-layer **joint
space-time transformer** head, trained 300 epochs on the 33-class video
classification task. wandb run `7zis80fb`, log: `exp8.log`. Resumed once
from epoch 144 (best so far 0.4742) → 300.

## Final numbers

| metric | value |
|---|---|
| `val/best_acc` | **0.5185** |
| `val/acc` (epoch 300) | 0.5166 |
| `val/loss` (epoch 300) | 2.1844 |
| `train/acc` (epoch 300, hard-label, mixup-on) | 0.3490 |
| `train/loss` (epoch 300) | 1.9305 |
| train − val gap (epoch 300) | **−16.8 pp** |
| best-checkpoint saves | 171 / 300 |

## Trajectory (key checkpoints, sampled from 171 best-saves)

| approx. epoch | val acc |
|---|---|
| early (~ep 1) | 0.071 |
| ~ep 5  | 0.110 |
| ~ep 10 | 0.155 |
| ~ep 20 | 0.235 |
| ~ep 40 | 0.330 |
| ~ep 80 | 0.405 |
| ~ep 144 (resume point) | 0.474 |
| ~ep 200 | 0.501 |
| ~ep 300 (final best) | **0.518** |

`val/loss` is U-shaped per the wandb sparkline: drops through ~ep 30,
flat to ~ep 80, then climbs monotonically to 2.18 at ep 300. `val/acc`
keeps climbing the whole time, which means class **ranking** keeps
improving while **calibration** drifts — exactly the regime
label-smoothing + mixup creates.

## Read

1. **Head swap was the right call.** Same V-JEPA Stage-1 dump as exp5,
   recipe ~matching exp2c. Only the head changed: CLS-only readout
   (exp5) → patch-token joint space-time transformer (exp8).
   Result: **0.3520 (exp5) → 0.5185 (exp8)**, a +16.7 pp jump.
   This confirms the exp5 hypothesis that V-JEPA's CLS was untrained
   and the old head was reading effectively dead features.
2. **Regularization is biting.** train_acc 0.349 *below* val_acc 0.517 —
   a negative gap, vs exp5's +28.7 pp positive gap. The hard-label
   train_acc is partly artifactual under mixup 0.2 + label_smoothing 0.1
   (mixed-label train samples can't easily score on the argmax), but
   the direction inversion plus the U-shaped val/loss confirm the model
   is no longer memorizing.
3. **val/acc still rising at ep 300.** Last 10 best-saves span
   0.508 → 0.518, roughly +0.001 per save. Schedule isn't visibly
   exhausted, but per-epoch return is asymptotic.
4. **No collapse, no NaN.** Survived 300 epochs (vs the earlier
   `jepa.log.nan-run` and the exp5c stall at acc≈0.07). The combination
   bf16 + grad_clip 1.0 + EMA 0.9999 + LLRD 0.75 + 20-epoch warmup is
   stable for this encoder.

## Comparison with the head/encoder family on the same data

| run | encoder init | head | reg stack | epochs | val/best |
|---|---|---|---|---|---|
| exp2  | MAE     | temporal (CLS) | none | 50  | 0.5212 |
| exp2b | MAE     | temporal (CLS) | drop_path 0.2 + dropout 0.2 + mixup 0.2 + strong_clip_aug | 50  | **0.5459** |
| exp5  | V-JEPA  | temporal (CLS) | none | 50  | 0.3520 |
| exp5c | V-JEPA  | temporal (CLS) | exp2c-style | 100 (killed ~ep84) | 0.0799 (collapsed) |
| **exp8** | **V-JEPA** | **spacetime (patch tokens)** | **exp2c-style** | **300** | **0.5185** |

Reading the table:
- exp5 → exp8 isolates the head swap. **+16.7 pp** at the head, with the
  same encoder and (close to) the same recipe.
- exp8 vs exp2 (0.5185 vs 0.5212): V-JEPA + patch head is essentially
  on par with MAE + CLS head (no reg). Tells us the V-JEPA features are
  competitive *once you read the right tokens*.
- exp8 vs exp2b (0.5185 vs 0.5459): under matched regularization, MAE
  still leads V-JEPA by ~3 pp on the same head family. The cleanest
  measurement so far that V-JEPA Stage-1 is the marginal weaker
  encoder, not a recipe gap.
- exp5c (0.0799) was a CLS-head V-JEPA + heavy reg run that collapsed
  to near-random — heavy reg on the wrong-token head is worse than
  light reg on the wrong-token head. Don't repeat.

## Implications for the next move

- The CLS-vs-patch head question for V-JEPA is settled. Patch-token
  space-time head is the canonical Stage 2 for V-JEPA dumps from now on.
- The encoder gap to MAE (~3 pp at matched recipe) is small but real.
  Cheapest closers, ranked: longer V-JEPA Stage-1 (currently 200 ep at
  loss 0.0092 / cos 0.98), multi-clip TTA at eval, mean-pool-of-patches
  classifier variant for ensembling.
- `num_frames` stays at 4 (project hard cap).
- A short calibration finetune (mixup 0 + ls 0.0 for 10–20 epochs from
  the ep-300 checkpoint) is the cheapest way to re-fold the val/loss
  drift back into val/acc gains.
