# exp2e_mae_bornagain_ensembleteacher — proposal

Born-again successor to `exp2d_mae_distill_cutmix`. Same encoder (MAE-pretrained
ViT-S, Track 1), same temporal head, same `num_frames=4`, same CutMix+Mixup
recipe, same EMA. **One axis changes: the distillation teacher.** Instead of
the single exp2c teacher used in exp2d, we use a **frozen ensemble teacher
that averages exp2c + exp2d** (both are EMA snapshots) — and crank `distill.alpha`
up to `0.7` because the teacher is now stronger and better calibrated. We also
shorten the schedule to 80 epochs (exp2d plateaued at ep 83) and add 3-clip
temporal TTA at submission time.

This **diverges from the exp2d note's own recommendation** ("don't run exp2e/exp2f
ablations yet, wait for iBOT"). That recommendation was about the originally
scoped exp2e/exp2f (CutMix-only / distill-only attribution). The proposal here
is a different move — a higher-leverage one — that does not depend on iBOT and
exploits a specific signal exp2d surfaced (38 % test-clip disagreement with
exp2c at similar accuracy → complementary errors).

## Why

exp2d hit `val/best = 0.6010` (+1.42 pp over exp2c) by adding CutMix and
self-distillation from exp2c. The Stage-2 trick budget is now showing
diminishing returns per axis (4.09 → 1.42 pp), and the exp2d note correctly
flags architecture (iBOT, joint space-time) as the next big lever. But:

- **iBOT Stage-1 is blocked** on the remote machine; we can't schedule it here.
- **V-JEPA is on a separate path** (exp5, different family).
- The user wants the **best results ever** *now*, with the GPU we have.

So the question is: what is the highest-leverage single move that doesn't
depend on a missing checkpoint? Three signals point at iterative
self-distillation with an ensemble teacher:

1. **exp2c and exp2d disagree on 38 % of test clips at similar accuracy.**
   The exp2d note already flagged this: their errors are largely complementary,
   not redundant noise. A logit-average ensemble at submission would be a free
   probe — but a *better* use of that signal is to **distill the ensemble into
   a single student**, so we get the ensemble accuracy with single-model
   deployment cost. This is exactly the born-again-networks setup.

2. **Born-again networks (Furlanello et al. 2018) consistently produce
   +0.5–1.0 pp per generation** on image classification, with diminishing
   returns. exp2d is gen-1 (taught by gen-0 = exp2c). exp2e is gen-2.

3. **Ensemble teachers add +0.2–0.5 pp on top of single-teacher KD**
   (Hinton 2015, mean-teacher follow-ups). exp2c+exp2d is a particularly
   appealing ensemble because they share the architecture but learned
   different decision boundaries — high diversity, low calibration drift.

These compound; they don't substitute. Stacked, the conservative band is
**+0.5–1.5 pp over exp2d** at val time, plus another **+0.3–0.7 pp on test**
from temporal TTA at submission. Optimistic case lands at **val_best ≥ 0.62**.

## What changed vs `exp2d_mae_distill_cutmix`

| field                              | exp2d                                    | **exp2e (this)**                                                         | why |
|------------------------------------|------------------------------------------|--------------------------------------------------------------------------|-----|
| `training.distill.teacher_ckpt`    | `checkpoints/exp2c_mae_transformer.pt`   | **`[exp2c…pt, exp2d…pt]`** (list — averaged softmax outputs)             | Born-again + ensemble teacher. exp2d's 0.601 is a stronger floor than exp2c's 0.587, and averaging captures the 38 % disagreement as complementary signal. |
| `training.distill.alpha`           | `0.5`                                    | **`0.7`**                                                                | Teacher is now stronger and better calibrated; lean harder on its targets. Hard CE still gets 30 % of the loss for grounding. |
| `training.distill.temperature`     | `4.0`                                    | `4.0`                                                                    | unchanged — DeiT default, KL term still scaled by `T²`. |
| `training.distill.combine`         | n/a                                      | **`softmax_mean`**                                                       | New field. Average teacher *probabilities* (not logits), then re-log for KL. Avoids one teacher's logit scale dominating. |
| `training.epochs`                  | `100`                                    | **`120`**                                                                | exp2d's plateau at ep 83/100 was likely LR-floor limited (cosine had decayed near zero). With a richer ensemble-teacher signal, lengthening the cosine window should push the plateau later. Wall-clock ≈ 9h on A5000 (vs exp2d's 6h22). User explicitly opted in to spend more epochs. |
| `submission.tta_clips`             | absent                                   | **`3`**                                                                  | New flag in `create_submission.py`. Sample 3 temporally-jittered clips per test video, average softmax. Pure inference change — no training cost. |

Unchanged (intentionally — keep the delta interpretable):
`num_frames=4`, `batch_size=16`, `lr=3e-4`, `llrd=0.75`, `weight_decay=0.05`,
`label_smoothing=0.1`, `mixup_alpha=0.2`, `cutmix_alpha=1.0`,
`strong_clip_aug=true`, `drop_path=0.2`, `dropout=0.2`, `grad_clip=1.0`,
`amp=true`, `warmup_epochs=8`, `ema=true`, `ema_decay=0.9999`. Same MAE
Stage-1 init, same temporal head shape, same student architecture.

## Implementation

Three small, localized edits, all backwards-compatible.

### 1. `src/train.py` — extend `load_teacher` to support an ensemble list

Currently `load_teacher` takes a single path and returns an `nn.Module`.
Extend the contract:

- `teacher_ckpt: str | list[str]` accepted in config.
- If a single path → unchanged.
- If a list → load each, build a tiny `EnsembleTeacher(nn.Module)` wrapper
  whose `forward(x)` runs each teacher under `torch.no_grad()`, applies
  `softmax(logits / 1.0, dim=-1)`, averages, and returns `log` of the
  average (so the existing KL code in `_forward_loss` keeps working when
  it does `F.softmax(teacher_logits / T, dim=-1)`).

  Important detail: the existing code does `softmax(teacher_logits / T)`,
  which assumes `teacher_logits` is in pre-softmax (logit) space. To keep
  that contract, the ensemble wrapper instead returns the **mean of
  per-teacher logits** by default (`mean_logits`), or the
  **log of the mean of per-teacher softmax probabilities** when
  `combine=softmax_mean`. The `_forward_loss` site is unchanged either
  way; the ensemble wrapper just produces the right tensor.

  Pseudocode:

  ```python
  class EnsembleTeacher(nn.Module):
      def __init__(self, teachers: list[nn.Module], combine: str = "softmax_mean"):
          super().__init__()
          self.teachers = nn.ModuleList(teachers)
          self.combine = combine
          for t in self.teachers:
              t.eval(); t.requires_grad_(False)

      @torch.no_grad()
      def forward(self, x):
          logits_list = [t(x) for t in self.teachers]
          if self.combine == "mean_logits":
              return torch.stack(logits_list, dim=0).mean(dim=0)
          # "softmax_mean": mean of probs, return as logits via log()
          probs = torch.stack([F.softmax(l, dim=-1) for l in logits_list], dim=0).mean(dim=0)
          return torch.log(probs.clamp_min(1e-8))
  ```

### 2. `src/configs/experiment/exp2e_mae_bornagain_ensembleteacher.yaml`

```yaml
# @package _global_
# Exp2e (Track 1): exp2d + born-again teacher = ensemble(exp2c, exp2d).
# alpha bumped 0.5 -> 0.7 (teacher stronger), epochs 100 -> 80 (exp2d plateaued).
# 3-clip temporal TTA at submission. Single-axis vs exp2d on the teacher only.
defaults:
  - override /model: vit_mae_transformer

model:
  spatial:
    drop_path: 0.2
  temporal:
    dropout: 0.2

dataset:
  num_frames: 4

training:
  batch_size: 16
  lr: 3.0e-4
  llrd: 0.75
  epochs: 80
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
  checkpoint_path: ${hydra:runtime.cwd}/checkpoints/exp2e_mae_bornagain_ensembleteacher.pt
  device: cuda
  distill:
    teacher_ckpt:
      - ${hydra:runtime.cwd}/checkpoints/exp2c_mae_transformer.pt
      - ${hydra:runtime.cwd}/checkpoints/exp2d_mae_distill_cutmix.pt
    combine: softmax_mean
    alpha: 0.7
    temperature: 4.0

wandb:
  enabled: true
  run_name: exp2e_mae_bornagain_ensembleteacher
  tags: [track1, stage2, mae, transformer, regularized, ema, cutmix, distill, bornagain, ensemble_teacher]
```

### 3. `src/create_submission.py` — optional 3-clip temporal TTA

Currently the submission path is single-clip and deterministic
(`_pick_frame_indices` evenly spaces frames). Add a `submission.tta_clips=N`
flag (default `1`, opt-in):

- For `N=1`: unchanged.
- For `N>1`: build the test dataset `N` times with `N` distinct temporal
  offsets. Cleanest: introduce a `pick_strategy` argument on the dataset
  (`even` (current) | `even_offset_k` for k in [0..N-1]) that shifts the
  linspace by `k / N` of one frame stride before rounding. Run the model
  on each variant; average the per-clip softmax; argmax. Cost: ~N× the
  current submission inference (~5 min × 3 = ~15 min on A5000), no
  training impact.

This is a strictly additive flag — exp2c, exp2d, exp2e submissions can
all opt in or stay single-clip.

## What was *not* changed and why

- **Architecture identical to exp2d.** This run is a controlled axis
  swap on the teacher only. Exp2e being a clean delta vs exp2d is more
  valuable than a "kitchen sink" run that mixes architectural and
  distillation changes.
- **Mixup α / CutMix α / drop_path / dropout / LLRD / lr / WD all unchanged.**
  Same reason. We already paid for these knobs; muddling them now would
  waste the controlled comparison.
- **No multi-clip consistency loss at training time.** Tempting (it's
  effectively a learned form of TTA), but doubles per-step cost; the
  ensemble-teacher forward already adds +50 % vs exp2d's single teacher.
  Net wall-clock would land near 8h, which is past the comfort zone.
- **No new architecture (joint space-time, RelPos, TokenLearner).**
  Reserved for exp3 (iBOT-init) or exp4 once iBOT lands.
- **Hard-distillation token (DeiT III).** Soft KL is enough for born-again;
  the hard-distill token requires touching the model and a fresh teacher
  forward path, not worth it for ~+0.1–0.2 pp.

## Files

- New: `src/configs/experiment/exp2e_mae_bornagain_ensembleteacher.yaml`
- Edited: `src/train.py` — accept list-typed `distill.teacher_ckpt`, add
  `EnsembleTeacher` wrapper class, route through `load_teacher` /
  `main()`. ~30 lines, no behavioural change for single-teacher configs.
- Edited: `src/create_submission.py` — `submission.tta_clips` flag + a
  per-call temporal offset on the dataset. ~25 lines.
- Edited (minor): `src/dataset/video_dataset.py` — add an
  `index_offset_frac: float = 0.0` parameter to `_pick_frame_indices`
  (or a sibling helper), threaded through `VideoFrameDataset.__init__`.
- Reuses: everything else from exp2d.
- Teachers: `checkpoints/exp2c_mae_transformer.pt` (frozen) +
  `checkpoints/exp2d_mae_distill_cutmix.pt` (frozen).
- Output: `checkpoints/exp2e_mae_bornagain_ensembleteacher.pt`.

## How to run

Smoke test first (must show **two** teacher loads and the ensemble line):

```bash
cd /Data/challenge_sb
python src/train.py experiment=exp2e_mae_bornagain_ensembleteacher \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  training.num_workers=2 training.resume=false wandb.enabled=false
```

Expected smoke-test log lines:

```
[train] loading teacher from /Data/challenge_sb/checkpoints/exp2c_mae_transformer.pt
[train] loading teacher from /Data/challenge_sb/checkpoints/exp2d_mae_distill_cutmix.pt
[train] ensemble teacher: 2 models, combine=softmax_mean
[train] distillation enabled (alpha=0.7, T=4.0)
```

Full run inside tmux:

```bash
tmux new -s exp2e "python src/train.py experiment=exp2e_mae_bornagain_ensembleteacher 2>&1 | tee logs/exp2e_mae_bornagain_ensembleteacher.log"
```

Evaluate / submit after training:

```bash
python src/evaluate.py training.checkpoint_path=checkpoints/exp2e_mae_bornagain_ensembleteacher.pt

# Single-clip submission (baseline for the new model)
python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2e_mae_bornagain_ensembleteacher.pt \
  dataset.submission_output=submissions/exp2e_mae_bornagain_ensembleteacher_submission.csv

# 3-clip TTA submission (the leaderboard candidate)
python src/create_submission.py \
  training.checkpoint_path=checkpoints/exp2e_mae_bornagain_ensembleteacher.pt \
  submission.tta_clips=3 \
  dataset.submission_output=submissions/exp2e_mae_bornagain_ensembleteacher_tta3_submission.csv
```

## Expected behaviour

- **Throughput.** Same student forward as exp2d, but **two** frozen teacher
  forwards instead of one (the ensemble wrapper runs both serially under
  `no_grad`). Wall-clock per epoch ~1.2–1.3× exp2d. With 120 epochs (vs
  exp2d's 100), **total ≈ 9h on A5000** (vs exp2d's 6h22). User opted in
  to the longer schedule — the cosine window is the lever, not the recipe.
- **Train_loss not directly comparable** to exp2c/exp2d — the KL term
  weight changed (0.5 → 0.7) and the teacher distribution changed.
  Trust val/best.
- **val/best lands above exp2d's `0.6010`.** Conservative band
  **0.605–0.615** (born-again ~+0.5 pp, ensemble-teacher ~+0.3 pp on
  top, with interaction). Optimistic ≥ 0.62 if the ensemble teacher's
  complementary signal pays off as advertised.
- **TTA on submission gives +0.3–0.7 pp on test.** Independent of training,
  so even if val_best disappoints, the TTA flag stands on its own and
  also retroactively helps exp2c/exp2d submissions if we want them.
- **Decision rules:**
  - ✓ `val/best ≥ 0.615` → exp2e is the new Stage-2 baseline; consider a
    second seed for SWA across exp2d+exp2e (free further gain at
    submission via simple average).
  - ◇ `0.605 ≤ val/best < 0.615` → ensemble-teacher gave a real but
    modest gain; carry the recipe forward but don't iterate (born-again
    gen-3 is unlikely to clear noise).
  - ✗ `val/best ≤ 0.605` → ensemble teacher is wasted complexity. Roll
    back to single-teacher exp2d-as-teacher (`exp2f`), or pivot to exp3
    (iBOT) once that checkpoint lands.

## Why this is the best single shot we can run *today*

Constraints we have to live inside:
- Track 1 only. No ImageNet.
- `num_frames=4` capped.
- No hflip/vflip ever.
- iBOT (exp3) blocked on a remote machine.
- V-JEPA (exp5) is a separate-family path.

Within those constraints, the Stage-2 trick budget is the only lever, and
born-again with an ensemble teacher is **the move that maximally exploits
information we have already paid GPU time for**: exp2c and exp2d are sunk
cost, and their 38 % test-clip disagreement is a measured, not assumed,
signal of complementary errors. Compressing that complementarity into one
student is strictly better than running CutMix-only / distill-only
ablations (which would just re-attribute the existing exp2d gain) and
strictly safer than firing exp3 without iBOT weights.

If exp2e clears 0.615 we can stop adding tricks; if it doesn't, exp3
(iBOT) is the next call and this proposal is shelved.
