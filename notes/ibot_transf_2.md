# ibot_transf_2 — changelog

Second pass on the iBOT-Stage1 + temporal transformer recipe. Baseline run
`ibot_transformer_cleaned` (wandb id `sgq2mf2l`) finished at:

- best val acc **0.3021** (epoch 48/50)
- final train acc 0.3166, final val acc 0.2996
- train ≈ val → **underfitting**, both still trending up at the schedule end

## Diagnosis driving the changes

1. The previous config used `build_transforms` (Resize+Normalize only) — so
   train and val saw the *same* deterministic pipeline. Zero augmentation.
2. `LLRD=0.65` on a 12-block ViT-S leaves the bottom layer at lr `≈ 4.6e-6`,
   essentially frozen. iBOT's SSL objective ≠ classification, so the encoder
   does want to move.
3. Cosine ended at ~0 LR while val was still climbing → schedule too short.
4. No EMA, no Mixup/CutMix despite both being supported by `train.py`.
5. Frame ceiling: every video on disk has exactly 4 frames, so `num_frames=4`
   is forced and TSN-style segmental random sampling is a no-op (1 frame per
   segment ⇒ deterministic). `video_dataset.py` left unchanged.

## What changed vs. exp3_ibot_transformer

| field | v1 (cleaned) | v2 (this) | why |
|---|---|---|---|
| `training.strong_clip_aug` | (false) | **true** | clip-consistent RandomResizedCrop + ColorJitter + RandAugment + RandomErasing. **No** horizontal or vertical flip — SSv2 labels are direction-sensitive. |
| `training.mixup_alpha` | 0 | **0.2** | standard timm video recipe |
| `training.cutmix_alpha` | 0 | **1.0** | 50/50 with mixup per batch (`train.py:123-131`) |
| `training.ema` | false | **true** (decay 0.9998) | usually +0.5–2 pt on video classifiers |
| `training.llrd` | 0.65 | **0.78** | unfreeze bottom of the ViT — encoder needs to adapt |
| `training.epochs` | 50 | **80** | best was epoch 48/50 → schedule was clipped |
| `training.batch_size` | 16 | **32** | A5000 has headroom at `num_frames=4` + bf16 |
| `training.amp_dtype` | fp16 (default) | **bf16** | Ampere-safe; drops GradScaler |

Unchanged: `lr=5e-4`, `weight_decay=0.05`, `label_smoothing=0.1`,
`warmup_epochs=5`, `grad_clip=1.0`, `drop_path=0.1` on the spatial trunk,
2-layer transformer head, linear classifier.

## Augmentation contract (no flips)

`ConsistentClipAug` (`src/utils.py:250`) is enabled. Per-clip, the same random
parameters are applied to all 4 frames so motion stays coherent:

- **RandomResizedCrop** scale `(0.5, 1.0)`, ratio `(3/4, 4/3)` → 224×224
- **ColorJitter** brightness/contrast/saturation 0.4, hue 0.1
- **RandAugment** `n=2, m=9` — *no* hflip/vflip in torchvision's op set
- **RandomErasing** with probability 0.25, same rectangle on every frame

Eval pipeline is unchanged: deterministic Resize(224)+Normalize.

## Run / inspect

```bash
# Smoke test (fits on any GPU)
python src/train.py experiment=ibot_transf_2 \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
  wandb.enabled=false

# Full run (in tmux so an SSH drop won't kill it)
tmux new -s ibot2 "python src/train.py experiment=ibot_transf_2 2>&1 | tee logs/ibot_transf_2.log"
```

Checkpoints: `checkpoints/ibot_transf_2.pt` (best by val acc),
`checkpoints/ibot_transf_2_last.pt` (resume).

## Rough expected impact

Each lever roughly independent, ballpark from prior literature on
SSv2-style fine-tuning:

| change | rough +pt val acc |
|---|---|
| strong clip aug | +3 to +5 |
| longer schedule (50→80) | +1 to +2 |
| LLRD 0.65 → 0.78 | +0.5 to +2 |
| EMA | +0.5 to +1 |
| Mixup + CutMix | +0 to +1 |

Realistic next-run target: **0.36 – 0.42** val acc.
Stretch target with TTA at submission time: a touch higher.
