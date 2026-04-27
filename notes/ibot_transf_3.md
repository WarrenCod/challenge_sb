# ibot_transf_3 — changelog

Third pass on iBOT-Stage1 + temporal-transformer (Track 1). Built directly on the
diagnosis of v2 (`ibot_transf_2`), which collapsed.

## What happened in v2

`ibot_transf_2` (wandb run `ibot_transf_2`) **never trained**. From the log:

| epoch | train loss | train acc | val loss | val acc |
|---|---|---|---|---|
| 1   | 3.3885 | 0.0689 | 3.4543 | 0.0706 |
| 40  | 3.3671 | 0.0704 | 3.3602 | 0.0706 |
| 80  | 3.3664 | 0.0704 | 3.3585 | **0.0706** |

`log(33) ≈ 3.4965`. Both losses sit at ≈ 3.37 for 80 epochs — the model has
barely learned the class prior, and val acc is **identical to four decimal
places every epoch**, which means the classifier collapsed to a single
constant prediction (the dominant class is ~7% of val).

For comparison, v1 (`exp3_ibot_transformer`, run `ibot_transformer_cleaned`)
from the **same** `checkpoints/ibot_stage1.pt` went 0.071 → 0.302 cleanly.
So the iBOT pretraining is fine. **The v2 Stage 2 recipe broke it.**

## Why v2 broke (most likely → least likely)

v2 changed five things at once on top of v1. Walking the failure modes:

1. **CutMix α=1.0 on SSv2.** With α=1, the mix coefficient is ~Uniform(0,1).
   Half of all CutMix batches glue two unrelated motion clips at ~50/50 with
   a soft label like `0.5 × Pulling + 0.5 × Closing-something`. SSv2 labels
   are direction-and-motion-defined, so spatial CutMix obliterates the
   target signal. Strongest single suspect.
2. **Strong-aug + LLRD 0.78 + lr=5e-4 stacked.** LLRD 0.65 → 0.78 raised
   layer-0 LR ~7.5× and layer-6 LR ~3×. The encoder is pushed harder while
   simultaneously being fed RandAugment(m=9), RRC scale (0.5, 1.0),
   RandomErasing — far from the input distribution iBOT pretrained on.
   Combined with label smoothing 0.1 and mixup soft labels, the gradient
   SNR drops below what's needed to escape the constant-prediction basin.
3. **EMA decay 0.9998 + EMA-only eval.** Val numbers are computed only on
   the EMA copy; with decay 0.9998 the EMA lags the live model by several
   epochs early on. Wouldn't cause 80 epochs of flat val on its own, but
   masks any short-lived recovery.
4. Schedule (50 → 80) and batch (16 → 32): cosmetic, not the issue.
5. bf16 vs fp16: bf16 has fp32 exponent range; `train.py` correctly drops
   the GradScaler in that branch. Not the issue.

**Pretrain or Stage 2 problem?** Stage 2. The encoder produces useful
features (proven by v1). Don't relaunch iBOT — fix the FT recipe.

## What changes in v3

The smallest set of high-impact, low-risk levers from the diagnosis:

| field | v1 (exp3) | v2 (broken) | **v3** | why |
|---|---|---|---|---|
| `aug_strength`         | (none)    | strong      | **soft** | RRC (0.7,1.0), CJ 0.3/0.05, RandAug n=2 m=6, RE p=0.15. ViT-S iBOT trunk likes inputs closer to its pretraining distribution. |
| `mixup_alpha`          | 0         | 0.2         | **0.1** | Mass strongly near {0,1}; gentle interp only. |
| `cutmix_alpha`         | 0         | 1.0         | **0.0** | Removed entirely — destroys motion direction. |
| `freeze_backbone_epochs` | 0       | 0           | **5**   | LP-FT: head warms up against a stable encoder before any FT. Kills the "first-epoch random head wrecks the encoder" failure mode. |
| `ema`                  | false     | true (.9998) | **true (.999)** | EMA used as a *teacher*, not eval target. Decay 0.9998 is too slow at 90k steps. |
| `distill_alpha`        | 0         | 0           | **0.5** | EMA self-distillation: `loss = 0.5·CE_mix + 0.5·T²·KL(student/T ‖ teacher/T)`. Soft labels track the moving model and give better-shaped targets than label smoothing alone. |
| `distill_temperature`  | —         | —           | **2.0** | Standard. |
| `eval_on_ema`          | (n/a)     | true        | **false** | EMA is the teacher now — eval the live student so val measures the thing being trained. |
| `llrd`                 | 0.65      | 0.78        | **0.70** | Mild encoder adaptation, not aggressive — v2 showed 0.78 + strong-aug is unstable. |
| `epochs`               | 50        | 80          | **80**  | Keep the longer schedule. v1 was still climbing at ep48. |
| `batch_size`           | 16        | 32          | **32**  | Fits in bf16 on the A5000. |
| `amp_dtype`            | fp16      | bf16        | **bf16** | Keep — Ampere-safe, drops GradScaler. |

Unchanged from v2: `lr=5e-4`, `weight_decay=0.05`, `label_smoothing=0.1`,
`warmup_epochs=5`, `grad_clip=1.0`, `drop_path=0.1`, 2-layer transformer
head, linear classifier, no flips.

## Why LP-FT + EMA-distill, mechanically

- **LP-FT** (Kumar et al., "Fine-Tuning Can Distort Pretrained Features"):
  for the first 5 epochs only the temporal head + classifier train, then the
  encoder unfreezes. The optimizer/LLRD groups are built upfront with all
  params present; during freeze the spatial params have `requires_grad=False`,
  so AdamW skips them (their `.grad` stays `None`). On unfreeze
  (`set_backbone_frozen(model, False)`), they re-enter the gradient flow at
  whatever LR the cosine schedule has reached — i.e. just past peak.
- **EMA-teacher KD**: at every step, the EMA copy (in `.eval()`, no grad)
  is forwarded on the same (post-mixup) batch as the student. The KL term
  pulls the student towards the EMA's softmax — a smoothed, slightly-stale
  view of itself. Empirically worth +1–2 pt in low-data ViT FT and
  cushions the noise from heavy-aug/mixup.

## Augmentation contract (still SSv2-safe, soft preset)

`build_soft_clip_transform` (`src/utils.py`) — every random parameter is
sampled once per clip, applied to all 4 frames:

- **RandomResizedCrop** scale `(0.7, 1.0)`, ratio `(3/4, 4/3)` → 224×224
- **ColorJitter** brightness/contrast/saturation 0.3, hue 0.05
- **RandAugment** `n=2, m=6` (no hflip/vflip in torchvision's op set)
- **RandomErasing** prob 0.15, same rectangle on every frame

Eval pipeline unchanged: deterministic Resize(224) + Normalize.

## Run / inspect

```bash
# Smoke (1 frozen epoch + 1 unfrozen epoch on 64 samples)
python src/train.py experiment=ibot_transf_3 \
  dataset.max_samples=64 training.epochs=2 training.batch_size=4 \
  training.freeze_backbone_epochs=1 training.num_workers=2 \
  wandb.enabled=false

# Full run, in tmux so an SSH drop can't kill it
tmux new -s ibot3 "python src/train.py experiment=ibot_transf_3 2>&1 | tee logs/ibot_transf_3.log"
```

Checkpoints: `checkpoints/ibot_transf_3.pt` (best by val acc on the live
student), `checkpoints/ibot_transf_3_last.pt` (resumable).

## Rough expected impact

Independent estimates (rough, prior literature on SSv2-style FT):

| change | rough Δ val acc |
|---|---|
| LP-FT (5 ep freeze)             | +2 to +4  |
| EMA-teacher distill (α=0.5,T=2) | +1 to +2  |
| Soft aug (vs none)              | +2 to +4  |
| Drop CutMix, soften Mixup       | recovers v2's 23-pt collapse |
| LLRD 0.78 → 0.70                | stability, not accuracy |

Realistic target: **0.40 – 0.46** val acc. If the recipe lands cleanly, the
remaining levers (3-crop TTA, repeated augmentation, reverse-direction
auxiliary head) can be layered on as v4 / v5 — each on its own.
