# `dinov2_transformer` v1 — post-mortem (run `u6mz76ai`, 2026-04-29)

## TL;DR

- **Best val acc: 65.31%** (top-1, 33 classes, chance ≈ 3.13%) — first viable Track-2 result; ~7× the best from-scratch run (`vit_tf_2` at 10.50%, `vit_transformer_mlp` v1 at 9.22%).
- DINOv2 ViT-S/14 (reg4, LVD-142M pretrained) + 2-layer temporal Transformer + MLP head, full fine-tune with LLRD 0.65.
- Schedule was well-budgeted: warmup → fast climb (44% by epoch 1, 60% by epoch 5), wobble through the schedule's mid-epochs, then steady cosine descent to a peak at the last epoch (65.31%).
- `train_acc 36% < val_acc 65%` at convergence — a clean sign of *under-fit train view, well-fit val view* due to mixup α=0.2 + label-smoothing 0.1 on a strongly-pretrained backbone. Not a problem; just means there's still room to push by easing reg.

## Setup

| Field | Value |
|---|---|
| Model | DINOv2 ViT-S/14 reg4 (LVD-142M pretrained, timm `vit_small_patch14_reg4_dinov2.lvd142m`) → 2-layer Transformer → MLP(768) |
| Spatial pool | `cls_avg` (CLS ⊕ mean of patch tokens, out_dim = 2 × 384 = 768) |
| Spatial drop_path | 0.1 |
| Temporal | 2 layers, 12 heads (64-dim each), dropout 0.1, drop_path 0.0 |
| MLP head | hidden 768, dropout 0.2 |
| Optimizer | AdamW, lr 5e-4, wd 0.05, **LLRD 0.65** (31 param groups) |
| Schedule | 5 warmup + cosine over 30 epochs |
| Batch | 32, AMP fp16, grad-clip 1.0 |
| Aug | strong_clip_aug (RRC + ColorJitter + RandAug + RandomErasing, no hflip) |
| Reg | mixup α=0.2, label_smoothing 0.1, **EMA 0.9999** |
| Frames | `num_frames=4` (matches disk) |
| Image size | 224 (16 × 14 = 224, matches DINOv2 patch grid) |
| Data | 36k train clips, val 8.7k clips |
| Hardware / runtime | local CUDA GPU, ~2 h wall (30 epochs × ~4 min/epoch from log timestamps) |

## Trajectory

| Epoch | train loss | train acc | val loss | val acc | event |
|---|---|---|---|---|---|
| 1 | 3.0089 | 0.132 | 2.2477 | 0.4414 | new best |
| 2 | 2.6111 | 0.214 | 1.9114 | 0.5469 | new best |
| 3 | 2.4802 | 0.235 | 1.8016 | 0.5795 | new best |
| 4 | 2.4485 | 0.235 | 1.7632 | 0.5915 | new best |
| 5 | 2.5138 | 0.236 | 1.7744 | 0.5972 | end of warmup, new best |
| 6 | 2.5291 | 0.229 | 1.8134 | 0.5813 | dip (cosine kicks in) |
| 8 | 2.6217 | 0.206 | 1.8745 | 0.5694 | local minimum |
| 12 | 2.4337 | 0.247 | 1.8229 | 0.5787 | recovering |
| 17 | 2.1827 | 0.300 | 1.7159 | 0.6147 | new best |
| 20 | 2.0616 | 0.305 | 1.6561 | 0.6346 | new best |
| 24 | 1.9042 | 0.341 | 1.6048 | 0.6490 | new best |
| 28 | 1.8176 | 0.383 | 1.5870 | 0.6522 | new best |
| 30 | 1.8283 | 0.361 | 1.5847 | **0.6531** | final / global best |

Shape:
- Warmup pulled the head + last LLRD groups very fast — **44% val acc after a single epoch** is the pretrained-backbone signature.
- Epochs 5–11 wobbled (val 56–60%) as cosine started lowering lr from peak; the model was still re-balancing the freshly-perturbed pretrained features against the new head.
- Epochs 12–30 are a clean monotonic climb (with mild noise) typical of mid-late cosine on a stable optimization. Best is on the final epoch — schedule was well-budgeted.
- `val_loss < train_loss` and `val_acc > train_acc` throughout: training side is artificially harder because of mixup mixing + label smoothing flattening. EMA also smooths only val (the EMA shadow is what's evaluated). All three together mean the train curve here understates real fit quality.

## Diagnosis / what worked

### Pretrained DINOv2-S/14 was the right backbone

DINOv2 LVD-142M features generalize unusually well downstream — the +55 pp gap vs the from-scratch ViT-Ti runs is mostly attributable to the backbone, not the head or the recipe. The reg4 variant was a deliberate choice (paper-recommended for downstream); `cls_avg` pooling kept both global (CLS) and local-motion (mean-of-patches) signal.

### LLRD 0.65 was aggressive but correct

31 param groups with 0.65 decay → the deepest blocks see lr 5e-4 while the early blocks see ~5e-4 × 0.65^11 ≈ 2.4e-6. That's near-frozen for the early DINOv2 layers, which is exactly what you want when the pretrained features are this strong on 36k clips. No sign of catastrophic forgetting (no val-acc collapse, no late-stage divergence).

### Image size 224 (= 16 × 14) avoided pos-embed weirdness

The model loaded at 14×14 patches over 224 px (16×16 patch grid). timm logs show `Resized position embedding: (37, 37) to (16, 16)` — bilinear interp from DINOv2's training grid. This is fine but means the model is slightly OOD on grid layout vs its 518-px pretraining; results suggest the resize is benign at this scale.

## What this run did *not* prove

- **Whether mixup/EMA/label-smoothing are pulling their weight here.** The train-loss-below-val-loss pattern hints they may be over-flattening the train signal. Worth A/B testing on a v2 with these dialed back.
- **Whether 30 epochs is the right budget.** Best is at epoch 30 (final), so the schedule was *not* obviously too long; arguably could be pushed to 40 to see if val acc keeps climbing past 65.31%.
- **Whether the temporal head is doing real work.** With only 5 tokens (CLS + T=4) and DINOv2's already-strong per-frame features, it's plausible mean-pool of frame embeddings would be within 1–2 pp of this number. Worth checking.

## Lessons → constraints for v2

1. **Keep DINOv2 ViT-S/14 reg4 as backbone.** No reason to change the dominant factor.
2. **Try lower mixup / smoothing** (e.g. mixup 0.1, smoothing 0.05) to close the train/val gap and see if val acc moves up.
3. **Push to 40–50 epochs.** Best on final epoch ⇒ schedule may be undersized.
4. **Sanity-check the temporal contribution.** Run an `avg_pool` temporal head ablation; if it lands within 1 pp of 65.31% the temporal Transformer is overhead.
5. **Try DINOv2 ViT-B/14** if memory allows. Bigger backbone, same pretraining, often +3–5 pp downstream.

## Reproduction

```bash
tmux new -s dinov2 \
  "python src/train.py experiment=exp_dinov2_transformer 2>&1 | tee dinov2.log"
```

Checkpoints (saved at repo root):
- `best_dinov2_transformer.pt` — EMA best, 145 MB (model state dict only)
- `best_dinov2_transformer_last.pt` — final-epoch full state (584 MB, includes optimizer)

## Submission

This is the strongest checkpoint produced on the rouloul branch so far (Track 2). Submission generated with:

```bash
python src/create_submission.py training.checkpoint_path=best_dinov2_transformer.pt
```

→ writes `processed_data/submission.csv` (6913 test videos).
