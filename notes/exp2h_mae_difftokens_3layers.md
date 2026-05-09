# exp2h_mae_difftokens_3layers — proposal + result

Architecture-axis successor to exp2f. Adds **difference tokens** in the
temporal head and bumps the temporal transformer depth **2 → 3 layers**.
Two coupled changes; everything else identical to exp2f (single exp2c
teacher, seed 1337, random temporal offset, 100-epoch schedule).

## Sketch

```
[clip frames f1..f4] -> ViT-S/16 (MAE-init) -> per-frame CLS z1..z4
                                                    │
                       difference tokens d_i = z_{i+1}-z_i (i=1..3)
                                                    │
              temporal seq = [CLS, z1, z2, z3, z4, d1, d2, d3]   (8 tokens)
                                                    │
                            3-layer transformer  ◄── (was 2 layers, 5 tokens)
                                                    │
                                                  CLS -> MLP -> 33 logits
```

## Why coupled (architecture-axis only)

- 3 delta tokens lift the temporal head's effective sequence length from
  5 → 8. The 2-layer head was sized for 5; adding depth at 5 tokens is
  saturated. With 8 tokens a 3rd layer has new relational pairs to mix,
  so it isn't dead capacity. Coupling keeps the depth/token-count ratio
  meaningful.
- Difference tokens give the head an explicit "what changed between
  frames" signal — the SSv2-style task is direction-sensitive (the
  no-hflip rule exists for this reason), so explicit Δ-tokens are a
  cheap inductive bias for that.

## Diff vs exp2g (preceding launched run)

| field | exp2g | **exp2h (this)** | why |
|---|---|---|---|
| temporal arch | `vit_mae_transformer` (2 layers, 5 tokens) | **`vit_mae_relational_transformer` (3 layers, 8 tokens, +Δ-tokens)** | architecture axis |
| teacher | 4-way ensemble {2c,2d,2e,2f}, softmax_mean | **single exp2c, α=0.5, T=4** | exp2g's ensemble teacher gave ~0 pp over single 2c on holdout val; revert to simpler recipe to isolate arch change |
| schedule | 200 ep / warmup 16 | **100 ep / warmup 8** | 200-ep regime was for the longer-cosine probe; not the right backdrop for an architecture comparison |
| `drop_path` | 0.3 | **0.2** | matches exp2f baseline (compensation for longer schedule no longer needed) |
| `mixup_alpha` | 0.5 | **0.2** | same reason |
| seed, LLRD, lr, WD, EMA, cutmix, label smoothing, num_frames=4, batch=16 | exp2g values | **identical** | single-axis discipline — only the temporal head architecture moves |

Net change vs exp2f (the closer baseline recipe-wise): just the two
architectural edits in the temporal head.

## Important: first **real_val** run

As of this experiment, `train.py` validates on the **real `processed_data/val/`
held-out set** (6,746 videos), not the in-train 80/20 holdout that exp2c–2g
used. Per the train/val split fix on 2026-05-06, pre-fix `val/acc` numbers
ran ~+20 pp high. Do **not** compare exp2h's val/best to exp2c–2g directly.
The W&B `real_val` tag on the run is the marker.

## Result

- **Best val top-1: 0.3908** at epoch **73/100** on real val_dir (6,746 videos)
- Final epoch: train loss 1.18 / acc 0.42 | val loss 2.62 / acc 0.384
- Trajectory: smooth ascent, peak around ep 66–74, then a shallow plateau
  (val acc held 0.384–0.391 from ep ~60 onwards, val loss drifted up
  slightly from 2.55 → 2.62 — mild late overfitting offset by EMA).
- ~29 best-model saves over the run; no NaNs, no divergence, ran clean
  to ep 100. Run survived in tmux, log at
  `logs/exp2h_mae_difftokens_3layers.log`.
- vs the **bar to beat ≈10.5 %** from prior real-val runs: **+28.6 pp**.
  This is the new from-scratch real-val baseline on the Jabiru branch.
- vs exp2f/2g holdout-val (~0.62–0.63): **not comparable** — different
  val set. The ~+20 pp holdout-vs-real shift in user memory implies
  exp2f/2g would land in the ~0.42–0.43 region on real val, so exp2h's
  0.39 is in that same neighborhood; the architectural change has neither
  obviously helped nor hurt vs the (re-projected) prior recipe. To make
  this an apples-to-apples comparison we would need to re-eval exp2f/2g
  on real val_dir via `evaluate.py`.

## Checkpoints

- best (val=0.3908): `checkpoints/exp2h_mae_difftokens_3layers.pt`
- last (ep 100):     `checkpoints/exp2h_mae_difftokens_3layers_last.pt`

## Submission

- File: `submissions/exp2h_mae_difftokens_3layers_tta3_submission.csv`
- TTA=3 (offsets 0.000 / 0.333 / 0.667), softmax-mean over the 3 clips
- Rows: 6,913 + header (matches test set)
- Probs/names dumped for ensembling:
  `submissions/exp2h_mae_difftokens_3layers_tta3_submission.{probs.npy,names.txt}`
- Class coverage: 32 of 33 indices used; **class 27 absent** (matches the
  known dataset quirk — index 27 has no examples). No class is collapsed
  to ~0 predictions.
- Class distribution (top/bottom 3, counts /6913):
  - top: 30 (439, 6.4 %), 31 (379, 5.5 %), 12 (375, 5.4 %)
  - bottom: 26 (19, 0.3 %), 16 (43, 0.6 %), 25 (61, 0.9 %)
- Sanity OK: shape, header, no NaN cells, classes within `[0,32]`,
  no duplicate `video_name`s implied by the row count.

## Real-val landscape (TTA=1 / TTA=3, all on `processed_data/val/`, 6,745–6,746 videos)

| run | TTA=1 top-1 | TTA=3 top-1 |
|---|---|---|
| exp2c_mae_transformer | 0.3644 | 0.3671 |
| exp2d_mae_distill_cutmix | 0.3742 | 0.3687 |
| exp2e_mae_bornagain_ensembleteacher | 0.3800 | 0.3801 |
| exp2f_mae_seed_temporaljitter | 0.3788 | 0.3812 |
| exp2g_mae_4wayteacher_long (200 ep, 4-way teacher) | 0.3770 | 0.3752 |
| **exp2h_mae_difftokens_3layers** | **0.3908** | **0.3874** |

- TTA=3 is approximately a wash on val (helps 3 runs, hurts 2). TTA still
  worth using on test because it's cheap insurance.
- exp2h is +1.08 pp ahead of the next best (exp2e); the field is tightly
  clustered within a 2.6 pp band.
- exp2g's long-schedule + 4-way ensemble teacher recipe **underperformed**
  exp2f and exp2e on real val, retroactively justifying exp2h's reversion
  to a single (exp2c) teacher and 100-epoch schedule.
- Holdout-vs-real gap was **~24 pp**, not the ~20 pp recorded in memory
  (exp2f went 0.628 holdout → 0.379 real). Memory entry to refresh.

## Ensemble results (val NLL-optimized + acc-weighted, applied to test TTA=3 probs)

| ensemble | val top-1 | val NLL | Δ vs exp2h alone |
|---|---|---|---|
| exp2h alone (TTA=3) | 0.3874 | 2.2850 | — |
| 3-model {h,e,f} acc-weighted | 0.4001 | 2.2441 | +1.27 pp |
| 3-model {h,e,f} NLL-optimal | 0.3999 | 2.2330 | +1.25 pp |
| 6-model {c..h} acc-weighted | **0.4004** | 2.2312 | **+1.30 pp** |
| 6-model {c..h} NLL-optimal | 0.4000 | 2.2152 | +1.26 pp |

- NLL-optimal weights for 6-model: exp2h **0.455**, exp2g 0.159, exp2f
  0.152, exp2c 0.110, exp2e 0.086, exp2d 0.039. Optimizer correctly
  drowns out the weakest model (exp2d).
- Adding the bottom 3 models on top of {h,e,f} buys ~0.03 pp top-1 —
  diminishing returns. The 3-model and 6-model ensembles are equivalent
  on top-1 within noise.
- Acc-weighting beats NLL-optimal on top-1 by ~0.04 pp because NLL
  minimization recalibrates probabilities at the cost of a few argmax
  flips. For top-1 leaderboard scoring, acc-weighting is the better play.

## Official submission

- `submissions/ensemble_2c2d2e2f2g2h_realvalw_tta3_submission.csv`
- 6-model acc-weighted ensemble (weights = each model's TTA=3 real-val
  top-1, normalized): {0.165, 0.165, 0.171, 0.171, 0.169, 0.174}
- Expected leaderboard: ~0.40–0.41 top-1, ~+1.3 pp over exp2h alone
- Sanity: 6,913 rows, 32 distinct classes (class 27 absent as expected),
  no degenerate predictions.

Backup CSVs also written for comparison:
- `ensemble_2c2d2e2f2g2h_optimal_tta3_submission.csv` (6-model, NLL-optimal)
- `ensemble_2h2e2f_realvalw_tta3_submission.csv` (3-model, acc-weighted)
- `ensemble_2h2e2f_optimal_tta3_submission.csv` (3-model, NLL-optimal)

## Next

The ensemble is a one-shot win; further improvements need a **new training
run** on top of exp2h. Diagnosis: exp2h is **underfitting the task** (train
0.42, val 0.39, narrow gap), so extra regularization is unlikely to help —
the wins are in *useful signal* and *temporal-aware capacity*. Candidates
in next-experiment.md.

