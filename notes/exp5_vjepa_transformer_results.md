# exp5_vjepa_transformer — results

V-JEPA Stage-1 EMA target encoder (ViT-S/16, Track 1) + 2-layer temporal
transformer head, trained 50 epochs on the 33-class video classification
task. Run id `ra5awwor`, log: `exp5.log` / `wandb/run-20260428_035047-ra5awwor`.

## Final numbers

| metric | value |
|---|---|
| `val/best_acc` | **0.3520** (epoch 47) |
| `val/acc` (epoch 50) | 0.3505 |
| `val/loss` (epoch 50) | 2.9550 |
| `train/acc` (epoch 50) | 0.6374 |
| `train/loss` (epoch 50) | 1.6590 |
| train − val gap (epoch 50) | **+28.7 pp** |
| best-checkpoint saves | 27 / 50 epochs |

## Trajectory (every 5 epochs)

| epoch | train acc | val acc | train loss | val loss | gap |
|---|---|---|---|---|---|
|  1 | 0.0650 | 0.0940 | 3.3816 | 3.2951 | −2.9 pp |
|  5 | 0.1422 | 0.1358 | 3.1581 | 3.1394 | +0.6 pp |
| 10 | 0.2167 | 0.2269 | 2.9350 | 2.9298 | −1.0 pp |
| 15 | 0.2755 | 0.2617 | 2.7613 | 2.8061 | +1.4 pp |
| 20 | 0.3387 | 0.2888 | 2.5630 | 2.7552 | +5.0 pp |
| 25 | 0.3948 | 0.3206 | 2.3838 | 2.7423 | +7.4 pp |
| 30 | 0.4632 | 0.3338 | 2.1700 | 2.7417 | +12.9 pp |
| 35 | 0.5360 | 0.3403 | 1.9534 | 2.7801 | +19.6 pp |
| 40 | 0.5952 | 0.3491 | 1.7786 | 2.8980 | +24.6 pp |
| 45 | 0.6254 | 0.3481 | 1.6839 | 2.9409 | +27.7 pp |
| 50 | 0.6374 | 0.3505 | 1.6590 | 2.9550 | +28.7 pp |

## Read

1. **Massive overfit, second half is wasted**. Per-decade `val/acc` slope:
   ep 1→10 +13 pp · ep 10→20 +6 pp · ep 20→30 +5 pp · ep 30→40 +1.5 pp ·
   ep 40→50 +0.1 pp. Train accuracy keeps climbing linearly until the last
   epoch. The model has plenty of capacity but no incentive to generalize.
2. **`val/loss` U-shape**. Bottoms at 2.69 around epoch 24, then climbs
   monotonically to 2.96 by epoch 50 — confidently-wrong predictions
   accumulating, the classic over-confidence signature of an under-regularized
   recipe. `val/acc` only inches up because the *ranking* of the top class
   stays correct while the calibration degrades.
3. **No regularization stack**. The recipe matches exp3 (iBOT-style:
   `lr=5e-4`, `llrd=0.65`, `drop_path=0.1`, temporal `dropout=0.1`,
   `weight_decay=0.05`, `label_smoothing=0.1`) — none of `mixup`,
   `strong_clip_aug`, or `ema` are on. exp2 → exp2b on MAE has already shown
   that this stack on top of the same head + data closes the gap and lifts
   `val/best` by ≈ +2.5 pp.
4. **V-JEPA Stage-1 itself looks healthy**. Stage-1 finished 200 epochs at
   `loss=0.0092`, `cos_sim=0.98`, `target_std=0.526` — predictor converged
   without target collapse. The Stage-2 ceiling is set by the recipe, not
   by a wedged encoder.

## Comparison with the MAE family on the same head + data

| run | encoder init | reg stack | epochs | val/best |
|---|---|---|---|---|
| exp2 | MAE | none | 50 | 0.5212 |
| exp2b | MAE | drop_path 0.2 + dropout 0.2 + mixup 0.2 + strong_clip_aug | 50 | **0.5459** |
| exp2c | MAE | exp2b + EMA + 100 ep | 100 | TBD (in flight) |
| **exp5** | **V-JEPA** | **none** | 50 | **0.3520** |

V-JEPA-with-exp3-recipe lands ~17 pp below MAE-with-exp2-recipe. Two non-
exclusive explanations:

- **Recipe-limited.** exp5 inherited exp3's gentler iBOT recipe but skipped
  the regularization stack that closes the gap on exp2 → exp2b. The MAE
  comparison shows ≈ +2.5 pp from regularization, plus another ≈ +1–2 pp
  expected from EMA + longer schedule. That alone could lift exp5 toward
  ≈ 0.39–0.41.
- **Encoder-limited.** V-JEPA Stage-1 was trained on 4-frame clips (the
  hard project cap) with bidirectional Δt ∈ {±1, ±2, ±3} predictor
  conditioning. Compared to V-JEPA at scale (64+ frames per clip, millions
  of videos), the predictive signal here is brittle and may produce
  features that satisfy the JEPA loss without transferring strongly to
  action classification. MAE's per-pixel objective is locally denser and
  appears more robust on this size of data.

The two are not separable from exp5 alone. The next experiment isolates
the recipe axis by running V-JEPA under the exp2c recipe; the residual
gap vs exp2c is then a clean measurement of encoder quality.

## Implications for the next move

- The dominant problem is regularization, not capacity or learning rate.
  exp5's plateau is over by epoch ~32; the remaining 18 epochs add nothing
  to `val/acc` and actively hurt `val/loss`.
- Architecture / head changes should wait until the regularization recipe
  is matched against the MAE/iBOT family. Mean-pool head, multi-clip TTA,
  or larger temporal head are all *after* this lever.
- `num_frames` stays at 4 (project hard cap).
