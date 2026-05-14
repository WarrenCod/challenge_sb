# exp2n_bornagain_bidir_swa — proposal

Stage-2-only iteration on **exp2m** (current bar). Stage-1 features unchanged
(`checkpoints/mae_stage1.pt`, exp2c MAE init). Architecture is *almost*
identical to exp2m — the only structural diff is a second auxiliary head
(predict-prev-CLS) wired symmetrically to the existing predict-next. All other
gains come from a **born-again teacher**, **stronger reg**, **shorter cosine
schedule**, and **SWA** averaged over the plateau region.

Goal: cross **50% real val** by stacking four small wins on exp2m's working
geometry. Each individual move is conservative; the bet is on stacking.

## Sketch

```
Stage 1 (frozen): MAE Stage-1 ViT-S/16  →  checkpoints/mae_stage1.pt
                                            (same as exp2k, exp2m)

Stage 2 (exp2n): identical to exp2m, except:

  • Distill teacher:  exp2c (~38% real)  →  exp2m (~46% real, born-again)
  • Aux head:         predict-NEXT-CLS only       →  predict-NEXT + predict-PREV
                       (cosine, stop-grad target)      (mirror, λ each = 0.05)
  • Reg:              drop_path 0.2 flat           →  linear 0.0→0.3 (heavier on ST blocks)
                      temporal MHA dropout 0.2     →  0.3
                      label smoothing 0.10         →  0.15
                      LLRD 0.75                    →  0.70
  • Schedule:         100 ep, warmup 12            →  70 ep, warmup 8
  • Checkpoint:       best-by-val + EMA            →  + SWA over ep 35..55 (saved separately)
  • Submission TTA:   tta_clips 3 (1-crop)         →  tta_clips 5 × 5-crop (25 views)

The backbone (ViT-S/16 + K=4 divided ST blocks) and head (Perceiver, 16Q × 788KV)
are unchanged. Zero-init temporal residual still in force.
```

## Why this design (lessons from exp2m)

1. **exp2m is reg-bound, not capacity-bound.** Train acc kept climbing past
   ep 60 (0.48 → 0.54) while val plateaued and slipped (0.47 → 0.45). Every
   change here attacks the gap, not the ceiling.
2. **Zero-init residual is the design pattern.** predict-prev mirrors
   predict-next which is already zero-init in `fc2`. exp2n is bit-identical to
   exp2m at step 0 modulo the new aux head's *zero-output* contribution.
3. **Distillation is the weakest signal in exp2m.** Teacher was exp2c (~38%
   real); upgrading to exp2m itself is textbook born-again.
4. **Plateau-aware schedule.** exp2m peaked at ep 61/100. A 70-ep cosine puts
   peak at ~ep 43, leaves ep 35–55 as a plateau window for SWA, and avoids the
   30 wasted epochs of exp2m's tail.

## Init (the non-regression guarantee, preserved)

- ViT-S/16 backbone: MAE Stage-1 weights (unchanged from exp2m).
- Last 4 blocks: divided ST. Temporal MHA out_proj **zero-init**, temporal
  pos-embed **zero-init**. → t=0 forward = exp2k = MAE Stage-1.
- Perceiver head: same init as exp2m.
- predict-NEXT aux head: existing `PredictNextCLSAuxHead` (zero-init `fc2`).
- predict-PREV aux head: new `PredictPrevCLSAuxHead`, structurally identical
  to predict-next with the input/target slicing flipped:
  - input: CLS at frames 1..T-1
  - target: stopgrad(CLS at frame 0)
  - same 2-layer MLP, same cosine loss, same fp32 cast, same zero-init `fc2`.
- Distill teacher = exp2m. Loaded frozen at start, run in eval mode, AMP-safe.

Step-0 invariant: every new piece is identity or zero, so the first forward
of exp2n is the same as exp2m's first forward. Any val drift below exp2m is
earned by gradient steps.

## Diff vs exp2m (full table)

| field | exp2m | **exp2n (this)** | rationale |
|---|---|---|---|
| backbone | ViT-S/16 + K=4 ST | unchanged | exp2m's geometry won; don't gamble |
| temporal head | Perceiver 16Q × 788KV | unchanged | clean A/B |
| Stage-1 init | MAE `mae_stage1.pt` | unchanged | user constraint |
| distill teacher | `exp2c_mae_transformer.pt` | **`exp2m_st_perceiver.pt`** | born-again, +1–1.5 pp |
| distill α | 0.5 | **0.6** | stronger teacher → trust it more |
| distill T | 4.0 | unchanged | |
| aux: predict-next | λ=0.1 | **λ=0.05** | halve to make room for prev, total aux unchanged |
| aux: predict-prev | — | **λ=0.05 (NEW)** | bidirectional symmetric signal |
| drop_path | 0.2 (flat) | **linear 0.0→0.30 (per-block)** | reg gap is the bottleneck |
| temporal MHA dropout | 0.2 | **0.3** | reg on the new attention path |
| label_smoothing | 0.10 | **0.15** | reg |
| LLRD | 0.75 | **0.70** | spatial layers move slightly more, helps ST integration |
| epochs | 100 | **70** | exp2m peaked at 61/100; tail wasted |
| warmup | 12 | **8** | scaled with shorter total |
| EMA | 0.9999 | unchanged | already optimal |
| SWA | — | **window ep 35..55, every-epoch update, saved as `_swa.pt`** | average plateau weights |
| batch / lr / WD / grad_clip / mixup / cutmix / strong_clip / amp | exp2m | unchanged | hold everything else still |
| `num_frames`, `random_temporal_offset`, `seed` | 4 / true / 1337 | unchanged | hard cap respected |
| TTA at submission | `tta_clips=3`, 1-crop | **`tta_clips=5`, 5-crop = 25 views** | submission-time only |

Three checkpoints will exist at end-of-run: `exp2n_*.pt` (best-by-val EMA),
`exp2n_*_swa.pt` (SWA average), `exp2n_*_last.pt` (resume). Submission picks
whichever wins on `evaluate.py`. The ensemble step (next note) will use both.

## Code changes required (small)

1. **`src/models/aux/predict_prev_cls.py`** (new, ~80 lines): clone
   `predict_next_cls.py`, swap the input/target slicing. Same constructor
   signature so train.py wires it identically.
2. **`src/train.py`**: add a `predict_prev_cls` cfg block paralleling
   `predict_next_cls` (instantiate aux, attach to model as
   `predict_prev_cls_head`, accumulate its scalar loss with its own λ). ~15
   lines, mirrors lines 461–476 and 195–199.
3. **`src/models/spatial/vit_mae.py`**: accept `drop_path` as either a float
   (current) or a list[float] (per-block). When a float `>0` is passed
   alongside a new `drop_path_schedule="linear"` flag, expand to per-block
   linspace internally. Default behavior preserved. ~20 lines.
4. **`src/train.py`**: SWA hook. Use `torch.optim.swa_utils.AveragedModel`
   (PyTorch built-in). After each epoch ≥ `swa_start`, call
   `swa.update_parameters(model)`. At end-of-train, save
   `swa.module.state_dict()` to `*_swa.pt`. ViT has LayerNorm only — no
   `update_bn` pass needed. ~25 lines.
5. **`src/configs/experiment/exp2n_bornagain_bidir_swa.yaml`** (new): all the
   field overrides above.

No changes needed to `evaluate.py` or `create_submission.py`: aux heads are
attached only at training time and the saved checkpoint contains the merged
Hydra config, so eval/submit pick the right architecture automatically.

## Smoke tests (must pass before launch)

- `bs=4 max_samples=64 ep=1 training.resume=false`: verify loss is finite,
  both aux heads contribute non-zero scalar losses by step 0 (they should be
  small constants — zero-init `fc2` makes both auxes ≈ 1.0 cosine-distance at
  init, multiplied by λ=0.05 each).
- `bs=16 max_samples=64 ep=1 training.resume=false`: verify no OOM at full
  batch (exp2m fit at bs=16 on the A5000; exp2n adds a frozen exp2m teacher
  → +~127 MB params + activations). If tight, drop teacher to fp16 inference
  (it's already in eval mode, no grad).
- Confirm SWA hook activates at the right epoch and saves `*_swa.pt`. Test
  by running 5-epoch toy with `swa_start=2`, `swa_end=4`.

## Failure modes / fallbacks

- **OOM from frozen teacher.** Cast teacher to fp16 in eval mode. If still
  tight, drop `tta_clips` from the *training-time* sanity ckpt only — should
  not happen, exp2m had headroom.
- **Aux loss explodes.** Predict-prev fires before backbone has any temporal
  signal (early epochs). Both aux heads are `fc2`-zero-init; loss is bounded
  by λ × 1.0 = 0.05 each at init. If exp2n's val is below exp2m by ep 10,
  drop λ to 0.025 each; if still bad, disable predict-prev and keep the
  born-again-distill + reg + SWA piece.
- **Born-again distill drags student to teacher's mistakes.** Lower α to 0.5
  (= exp2m setting). If exp2n is still worse than exp2m by ep 30, exp2m's
  ceiling on this Stage-1 is close to the real bound and we'll need an
  ensemble (exp2o) instead of more single-model training.
- **SWA window mis-tuned.** If exp2n's val curve peaks earlier (ep 30 ish),
  shift SWA window to ep 25..45. Cheap to adjust on a relaunch — the SWA
  state can be re-derived offline from per-epoch snapshots if we save them.

## Expected gain (honest budget)

- Born-again distill: +0.7–1.5 pp
- Stronger reg (drop_path schedule + dropout 0.3 + LS 0.15 + LLRD 0.70): +0.4–0.8 pp
- Bidirectional aux: +0.2–0.5 pp
- SWA over plateau: +0.3–0.7 pp
- 5-crop × 5-clip TTA (vs 1-crop × 3): +0.3–0.6 pp
- **Stack (with overlap discount, ~60%):** +1.5–2.6 pp on real val

If exp2m's real val is ≈46.5%, exp2n lands around **48–49%**. Crossing 50%
likely needs the ensemble step (exp2o, drafted separately).

## Launch (after approval and smoke tests)

```
nohup setsid bash scripts/train_robust.sh \
  python src/train.py experiment=exp2n_bornagain_bidir_swa \
  > /Data/challenge_sb/logs/exp2n_bornagain_bidir_swa.log 2>&1 < /dev/null &
disown
```

W&B run id = `exp2n_bornagain_bidir_swa`, `resume="allow"`. Add a crontab
`@reboot` line and a `*/2` watchdog calling `scripts/ensure_running.sh
exp2n_bornagain_bidir_swa`, mirroring the exp2k pattern. Replace the existing
exp2k @reboot entry — only one run at a time on the A5000.

## Open decisions before we wire this up

D1. **SWA implementation.** PyTorch built-in `AveragedModel` (clean, ~25 lines
    in train.py) vs offline averaging from per-epoch saved snapshots (zero
    train.py change, but needs disk for ~20× ~125 MB ≈ 2.5 GB). Default:
    built-in.
D2. **Aux λ split.** Halve to 0.05/0.05 (default — total aux weight equal to
    exp2m) vs keep 0.1/0.1 (double the aux signal but might dominate KD).
    Default: 0.05/0.05.
D3. **Epoch budget.** 70 ep (default — fits exp2m's curve) vs 80 ep (more
    buffer if SWA shifts the peak right).
D4. **5-crop TTA.** Adds ~5× submission cost (still trivial on the A5000).
    Keep at submission-time only? Default: yes.

## Run log

- **2026-05-10 23:27** — launched (W&B `d6c7x5rz`). Reached epoch 50/70 cleanly. SWA averaging began at ep 35 (configured).
- **2026-05-11 06:33** — process killed externally (no traceback, no OOM, no host reboot). `_last.pt` saved at end of ep 50 (06:31). In-memory SWA averager lost — never flushed to `_swa.pt`.
- **Mid-train real val** (best ckpt at kill time): **top-1 0.4746**, top-5 0.7881. Compare to **exp2m: 0.4689 / 0.7804**. exp2n leads by +0.57 pp top-1.
- In-train val/acc (0.4750) vs real val (0.4746): essentially zero gap on this setup — contradicts the prior +20 pp leakage memo. Worth verifying whether `train.py` was changed.
- **2026-05-11 23:15** — resumed from `_last.pt` (start_epoch=51). W&B run id pinned in yaml (`wandb.run_id: d6c7x5rz`) so the dashboard re-attaches. New watchdog: `scripts/ensure_running.sh` + cron `*/2` + `@reboot` (uses `.venv/bin/python` explicitly — prior `@reboot` with bare `python` would have hit /usr/bin/python 3.9, no torch). SWA averager restarts from scratch on this resume; with `start_epoch=35` ≤ current_epoch=51 it begins immediately → **20 SWA averages instead of the planned 35** (ep 51–70).
- ETA to finish: ~3 h based on the 7 h / 50 ep cadence.
