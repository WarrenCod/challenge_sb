# exp3b_noisystudent_pseudo_test — proposal

Single Stage-2 run that uses every empirical finding from the exp2*/exp3a
ladder. The ensemble probe (2026-05-14) showed errors correlated at 0.87
across {exp2m, exp2n, exp3a} — encoder is the bottleneck and TTA/ensemble
gain has saturated. The only proposed lever that adds **new information**
is pseudo-labels on the unlabeled test set; everything else here is a
small polish on the exp3a recipe, traceable to the curve.

Three orthogonal levers:

1. **Pseudo-label distillation on test** (the main lever). 6913 test
   videos → softmax-mean{exp2m, exp2n, exp3a} → filter `max_p ≥ 0.7` AND
   all-3-argmax-agree. Expected retention 30–50%. Soft targets (not
   argmax) used as supervision via CE-soft.
2. **Perceiver 2 cross-attn** (1 → 2). The actual *reader* against patch
   tokens; self-attn on 16 queries was saturating already at exp3a's 4.
   Second block zero-init (out_proj + MLP final) → step-0 forward
   bit-identical to exp3a's Perceiver.
3. **Recipe surgery.** exp3a val-loss bottomed @ ep 25 then climbed
   (over-reg cocktail). Cut epochs 55 → 35, LS 0.15 → 0.10, dropout
   0.30 → 0.20, drop_path 0.30 → 0.25, drop multi_clip (suspected wash).

K stays at 6 (ladder shows diminishing returns past K=4 → 6). PDH stays
with weight 0.3 → 0.2 (pseudo carries direction signal too).

## Sketch

```
Stage 1 (frozen): MAE Stage-1 ViT-S/16  →  checkpoints/mae_stage1.pt

One-off, pre-training (~30 min):
   softmax-mean{exp2m, exp2n, exp3a} on processed_data/test  (TTA=1)
   filter: max_p ≥ 0.70 AND all 3 argmaxes agree
   → processed_data/pseudo_labels_v1.pt
      {video_name → soft_target (33-d softmax)}

Stage 2 (exp3b):

  Spatial:    ViT-S/16 + K=6 divided ST blocks      (unchanged)
              drop_path linear 0.0 → 0.25            (was 0.30)

  Temporal:   Perceiver head
              ├─ 16 learnable queries
              ├─ cross-attn × 2                       ← NEW: 1 → 2 (2nd block zero-init)
              ├─ self-attn × 4                       (unchanged — saturating)
              ├─ temporal_pos sinusoidal + scale     (kept from exp3a)
              └─ dropout 0.20                        (was 0.30)

  Classifier: Linear(384, 33)

  Aux heads:
    └─ PairDirectionAuxHead (λ=0.2, was 0.3)   [labeled samples only]

  Data per step (bs=16):  12 labeled + 4 pseudo
                          (random sampling each side, drop_last on pseudo)

  Losses:
    labeled  : CE(student, y) [LS 0.10]
             + α · KD(student || ensemble teacher, T=4)    α = 0.4
             + λ_pair · CE_pair                            λ_pair = 0.2
    pseudo   : CE_soft(student, pseudo_softmax)             weight = 1.0
    total    : (L · loss_l + w · P · loss_p) / (L + P)

  Distill teacher: ensemble {exp2m, exp2n, exp3a}, softmax-mean
                   (same trio used to generate pseudo-labels — by design;
                    we want both supervision signals coherent)

  Schedule:   epochs 35, warmup 5
              LR 1.5e-4, LLRD 0.70, WD 0.05
              LS 0.10, mixup 0.2, cutmix 1.0, no hflip
              EMA 0.9999, grad_clip 0.5, AMP

  REMOVED from exp3a:
    • multi_clip consistency_kl  (suspected wash; pseudo replaces its role)
    • predict_next/prev_cls      (already gone)
    • SWA                        (was already off)
```

## Diff vs exp3a (full table)

| field | exp3a | **exp3b** | rationale |
|---|---|---|---|
| backbone | ViT-S/16 + K=6 ST | unchanged | K-scaling diminishing returns confirmed |
| drop_path | linear 0.0→0.30 | **linear 0.0→0.25** | recipe surgery |
| Perceiver depth (self-attn) | 4 | 4 | saturating |
| Perceiver cross-attn | 1 | **2** | NEW: cross-attn is the reader; never tested |
| Perceiver dropout | 0.30 | **0.20** | recipe surgery; less reg for shorter run |
| Perceiver temporal pos | sinusoidal+scale | unchanged | order-aware @ step 0 |
| distill teacher | {exp2m, exp2n} | **{exp2m, exp2n, exp3a}** | best ensemble we have |
| distill α | 0.5 | **0.4** | leaner KD (pseudo carries supervision now) |
| pair_direction | λ=0.3 | **λ=0.2** | pseudo soft targets overlap with directional signal |
| multi_clip | 2 clips, λ_cons=0.2 | **off** | suspected wash; pseudo replaces |
| pseudo_labels | — | **NEW** | the lever — see below |
| epochs | 55 | **35** | exp3a val-loss bottomed @ ep 25 |
| warmup | 8 | **5** | scaled with shorter total |
| batch_size | 8 (×2 clips=16) | **16** (=12 labeled + 4 pseudo) | total forward unchanged |
| label_smoothing | 0.15 | **0.10** | recipe surgery |
| TTA at submission | 5 | **1** | TTA=5 measured as wash on this line |
| reg knobs not listed | exp3a | unchanged | hold else equal |

## Step-0 invariant

- Spatial K=6 ST blocks: temporal MHA out_proj zero-init (unchanged).
- Perceiver second cross-attn block: out_proj **and** MLP-final Linear
  zero-init → contributes exactly 0 to query updates at t=0.
- PDH `fc_out` zero-init.
- Pseudo loss term only fires when `pseudo_per_batch>0` — the labeled-loss
  path at step 0 matches exp3a's step-0 forward modulo recipe constants
  (LS, dropouts) that don't affect parameter values.

## Code changes (already wired)

1. `src/models/temporal/perceiver.py` — `num_cross_attn:int=1` arg, extra
   blocks in `self.cross_extra` (zero-init). State-dict back-compat with
   exp2*/exp3a checkpoints preserved (block-0 stays at `self.cross`).
2. `scripts/generate_pseudo_labels.py` — new. CLI: `--models <ckpts>
   --threshold 0.7 --require_argmax_agreement --output <path>`.
3. `src/dataset/pseudo_label_dataset.py` — new. Returns
   `(video, soft_target)` from test/video_<id>/.
4. `src/train.py` — new `train_one_epoch_pseudo` parallel-loader path,
   dispatched on `training.pseudo_labels.enabled`. Two forwards per step
   (labeled gets PDH/KD; pseudo gets CE-soft only). Sample-weighted total
   loss.
5. `src/configs/experiment/exp3b_noisystudent_pseudo_test.yaml` — new.

No edits to `evaluate.py` / `create_submission.py` — pseudo is training-time only.

## Pre-launch order

```
# 1. Pseudo-label generation (~30 min, FG)
.venv/bin/python scripts/generate_pseudo_labels.py \
  --models checkpoints/exp2m_st_perceiver.pt \
           checkpoints/exp2n_bornagain_bidir_swa.pt \
           checkpoints/exp3a_vivit_pairwise_multiclip.pt \
  --threshold 0.7 --require_argmax_agreement \
  --output processed_data/pseudo_labels_v1.pt

# 2. Smoke (~5 min, FG)
rm -fv checkpoints/exp3b_noisystudent_pseudo_test{,_last}.pt
.venv/bin/python src/train.py experiment=exp3b_noisystudent_pseudo_test \
  dataset.max_samples=64 training.epochs=1 training.batch_size=8 \
  training.pseudo_labels.pseudo_per_batch=2 wandb.enabled=false

# 3. Real run (~24-30h, BG, watchdog)
rm -fv checkpoints/exp3b_noisystudent_pseudo_test{,_last}.pt
nohup setsid bash scripts/train_robust.sh \
  python src/train.py experiment=exp3b_noisystudent_pseudo_test \
  > /Data/challenge_sb/logs/exp3b_noisystudent_pseudo_test.log 2>&1 < /dev/null &
disown
# + cron @reboot and */2 watchdog calling scripts/ensure_running.sh exp3b_noisystudent_pseudo_test
```

W&B run_id pinned in yaml; `resume="allow"`.

## Expected gain (honest, with overlap discount)

| lever | optimistic | conservative |
|---|---|---|
| pseudo-label distillation | +3.0 | +1.5 |
| 2 cross-attn vs 1 | +0.8 | +0.3 |
| recipe surgery | +0.6 | +0.2 |
| **Sum** | **+4.4** | **+2.0** |
| **× 50% overlap** | **+2.2** | **+1.0** |

Landing band: **0.50 floor → 0.51 mid → 0.53 stretch** real val. Hitting
0.55 in one shot is unlikely; a second iteration (exp3c) using exp3b's
predictions as the next teacher historically pays half the previous gain:
0.49 → 0.51 → 0.53 → 0.54 across rounds is realistic; 0.55 likely needs
either round 3 or a new Stage-1.

## Failure modes & fallbacks

1. **Pseudo confirmation bias.** val_acc lags exp3a by ep 10. Fix:
   tighten threshold to 0.8, or drop agreement filter (kept anyway via
   the τ gate).
2. **Pseudo class imbalance.** generate-script prints histogram; if any
   class < 5 pseudo or > 800, cap/balance offline and regenerate.
3. **OOM from 3-teacher ensemble + pseudo loader.** Each teacher is ViT-S
   + Perceiver ≈ 30 M params + activations. fp16 inference should fit on
   24 GiB. If tight, set `distill.alpha=0` (rely only on labels + PDH +
   pseudo) — saves one teacher forward per step.
4. **2-cross-attn diverges early.** Second block zero-init should keep
   step-0 invariant; if val crashes < ep 5, drop to 1 cross-attn (one
   yaml line, no rebuild).
5. **Pseudo loader can't keep up** (CPU bottleneck). Increase
   `num_workers` on pseudo loader (currently `max(2, train_workers//2)`).

## Run log

- *(awaiting pseudo-label generation + smoke)*
