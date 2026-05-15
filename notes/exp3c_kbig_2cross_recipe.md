# exp3c_kbig_2cross_recipe — proposal

Isolation run on top of exp3a (real val 0.4937), **no pseudo-labels**. Tests
whether the K-bump + Perceiver-2-cross-attn + recipe-surgery bundle is +pp
or wash, independent of the noisy-student lever exp3b stacked on top.

User explicitly chose this *bundled* variant ("exp3c'") over the cleaner
K=6 attribution. We accept the trade: one run instead of two, but the
landing number doesn't factor into "K vs cross-attn vs recipe."

Three orthogonal levers, all NEW vs exp3a:

1. **K = 6 → 8 ST blocks** (last 8: blocks 4..11). Sweet spot on the
   K-axis above K=6. Historically K=2→4 paid +5.27 pp (single largest
   exp2*/exp3a win); K=4→6 paid ≤+1 pp; K=6→8 conservative is −0.5 to
   +1.0. Zero-init temporal MHA out_proj keeps step-0 identical to
   K=6.
2. **Perceiver 1 → 2 cross-attn.** Cross-attn is the *reader* against
   patch tokens; self-attn on 16 queries was saturating at exp3a's 4.
   Second block zero-init (attn out_proj + MLP final Linear) — step-0
   forward bit-identical to exp3a.
3. **Recipe surgery.** exp3a val-loss bottomed at ep 25 then climbed.
   ep 55→35, warmup 8→5, LS 0.15→0.10, drop_path 0.30→0.25, Perceiver
   dropout 0.30→0.20, drop multi_clip (suspected wash on exp3a).

PDH stays at λ=0.3 (not the lever). Distillation stays as exp3a's
{exp2m, exp2n} ensemble at α=0.5, T=4.

## Sketch

```
Stage 1 (frozen): MAE Stage-1 ViT-S/16  →  checkpoints/mae_stage1.pt

Stage 2 (exp3c):

  Spatial:    ViT-S/16 + K=8 divided ST blocks      ← NEW (was 6)
              blocks 0..3 spatial-only; 4..11 ST
              drop_path linear 0.0 → 0.25            ← recipe surgery

  Temporal:   Perceiver head
              ├─ 16 learnable queries
              ├─ cross-attn × 2                       ← NEW (1 → 2)
              ├─ self-attn × 4                       (unchanged)
              ├─ temporal_pos sinusoidal + scale     (unchanged)
              └─ dropout 0.20                         ← recipe surgery

  Classifier: Linear(384, 33)

  Aux head:   PairDirectionAuxHead (λ=0.3)            (unchanged)

  Data:       1 augmented clip per video; bs=16; random temporal offset
              (no multi-clip; NO pseudo-label batch half)

  Losses:     CE(y) [LS 0.10]                          ← recipe surgery (was 0.15)
            + α · KD(student || {exp2m, exp2n}, T=4)    α = 0.5  (= exp3a)
            + λ_pair · CE_pair                          λ_pair = 0.3  (= exp3a)

  Schedule:   35 ep, 5 warmup                          ← recipe surgery
              LR 1.5e-4, LLRD 0.70, WD 0.05
              EMA 0.9999, grad_clip 0.5, AMP, mixup 0.2, cutmix 1.0
```

## Diff vs exp3a

| field | exp3a | **exp3c** | rationale |
|---|---|---|---|
| K (ST blocks) | 6 | **8** | NEW: K-axis bump |
| Perceiver cross-attn | 1 | **2** | NEW: architectural lever (extra zero-init) |
| Perceiver self-attn | 4 | 4 | saturating; unchanged |
| Perceiver dropout | 0.30 | **0.20** | recipe surgery |
| drop_path | 0.0→0.30 | **0.0→0.25** | recipe surgery |
| epochs | 55 | **35** | val-loss bottomed @ ep 25 |
| warmup_epochs | 8 | **5** | scaled with shorter total |
| label_smoothing | 0.15 | **0.10** | recipe surgery |
| multi_clip | on (λ_cons=0.2) | **off** | suspected wash |
| batch_size | 8 (×2=16) | **16** straight | follows multi_clip drop |
| distill teacher | {exp2m, exp2n} | **{exp2m, exp2n}** | unchanged |
| distill α | 0.5 | 0.5 | unchanged |
| pair_direction λ | 0.3 | 0.3 | unchanged (not the lever) |
| pseudo_labels | — | — (off) | THE ISOLATION (exp3b had this on) |
| TTA at submit | 5 | 1 | TTA=5 confirmed wash |
| reg knobs not listed | exp3a | unchanged | hold else equal |

## Step-0 invariant

- Spatial K=8 ST blocks: temporal MHA out_proj zero-init + temporal-pos
  zero-init on the *new* blocks 4..5 → those two extra blocks contribute
  exactly 0 at step 0; blocks 6..11 still match exp3a's K=6 forward.
- Perceiver second cross-attn block: out_proj **and** MLP-final Linear
  zero-init → contributes exactly 0 to query updates at t=0
  (verified empirically on 2026-05-14: state-dict transfer from
  num_cross_attn=1 weights into num_cross_attn=2 model gave exact zero
  output diff).
- PDH `fc_out` zero-init.

Net: t=0 forward is bit-identical to exp3a at K=6 (modulo recipe
constants — LS, dropouts — which don't affect parameter values).

## Honest expected gain (with overlap discount)

| lever | optimistic | conservative |
|---|---|---|
| K = 6 → 8 | +1.0 | −0.5 |
| Perceiver 1 → 2 cross-attn | +0.8 | +0.3 |
| Recipe surgery (ep, LS, dropouts, drop_path) | +0.6 | −0.2 |
| Drop multi_clip (suspected wash) | +0.2 | −0.1 |
| **Sum** | **+2.6** | **−0.5** |
| **× 50% overlap** | **+1.3** | **−0.3** |

**Landing band: 0.49 floor → 0.50 mid → 0.51 stretch.**

The conservative case is *negative* because:
- exp3b mid-run (as of ep 18) is already 0.5 pp behind exp3a on top-1
  with these same architectural + recipe levers in place. If the
  underperformance comes from the recipe/arch (not the pseudo), exp3c
  could land below exp3a.
- More capacity (K=8) on a model already showing widening train-val gap
  is risky.

This run is **diagnostic** more than performance-chasing. The information
content is high regardless of outcome:
- Lands ≥ 0.50 → recipe+arch+K is genuinely positive; **pseudo is a
  wash or worse** on this line. Drop pseudo, push K and recipe further.
- Lands 0.49–0.50 (~ exp3a) → bundle is neutral; only pseudo could have
  paid (and exp3b will tell us). Either way, the line is near its ceiling
  on this Stage-1.
- Lands < 0.49 → bundle is a regression. The recipe surgery
  over-loosened reg for the shorter schedule, or K=8 broke MAE features.
  Abandon this lever set; go to a new Stage-1.

## Code changes

**None.** Everything required is already wired:
- `src/models/spatial/vit_mae.py` accepts `space_time_layers: 8` (was
  used by exp3a with 6).
- `src/models/temporal/perceiver.py` accepts `num_cross_attn: 2` (wired
  for exp3b).
- `src/configs/experiment/exp3c_kbig_2cross_recipe.yaml` (new) — single
  yaml.

`evaluate.py` / `create_submission.py` need no edits.

## Pre-launch checklist

1. exp3b finishes (touches `checkpoints/exp3b_noisystudent_pseudo_test.done`).
2. Smoke (~3 min): `python src/train.py experiment=exp3c_kbig_2cross_recipe
   dataset.max_samples=64 training.epochs=1 training.batch_size=4
   wandb.enabled=false`. Verify finite loss, K=8 wrap message, 2 teachers
   loaded, no pseudo loader allocated, PDH at λ=0.3.
3. `rm -fv checkpoints/exp3c_kbig_2cross_recipe{,_last}.pt` after smoke.
4. Real launch under `train_robust.sh` + cron watchdog.

## Launch (after exp3b.done)

```
# 1. Clean smoke ckpts
rm -fv /Data/challenge_sb/checkpoints/exp3c_kbig_2cross_recipe{,_last}.pt \
       /Data/challenge_sb/checkpoints/exp3c_kbig_2cross_recipe.done

# 2. Launch detached + watchdog
cd /Data/challenge_sb
nohup setsid bash scripts/train_robust.sh \
  .venv/bin/python src/train.py experiment=exp3c_kbig_2cross_recipe \
  > logs/exp3c_kbig_2cross_recipe.log 2>&1 < /dev/null &
disown

# 3. Replace cron watchdog (one run at a time on this A5000)
( crontab -l 2>/dev/null | grep -v 'ensure_running.sh exp3'; \
  echo '*/2 * * * * bash scripts/ensure_running.sh exp3c_kbig_2cross_recipe >> logs/ensure_running.log 2>&1'; \
  echo '@reboot cd /Data/challenge_sb && sleep 30 && bash scripts/ensure_running.sh exp3c_kbig_2cross_recipe >> logs/ensure_running.log 2>&1' \
) | crontab -
```

## Failure modes & fallbacks

1. **K=8 broke MAE features.** Detect: val/acc < 0.30 by ep 10. Fix: drop
   to K=6 (one yaml line). At step 0 the K=8 forward is identical to K=6
   *given fresh init*, but training trajectory diverges; check the
   first ~10 ep curve against exp3a's first 10 ep.
2. **Recipe over-loosened.** Detect: train/val gap > 15 pp by ep 15. Fix:
   restore drop_path 0.30 + Perceiver dropout 0.30.
3. **2-cross-attn diverged.** Detect: val crash < ep 5. Fix:
   `temporal.num_cross_attn=1`.
4. **OOM (K=8 + 2 teachers + 2-cross + bs=16).** Predicted ~7–8 GB / 24
   GB on A5000 (exp3b at 5.5 GB). If tight: bs=12.

## Run log

- *(awaiting exp3b finish + smoke)*
