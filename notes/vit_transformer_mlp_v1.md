# `vit_transformer_mlp` v1 — post-mortem (run `0qjn39xs`, 2026-04-25 → 04-26)

## TL;DR

- **Best val acc: 9.22%** (top-1, 33 classes, chance ≈ 3.13%) — about 3× chance, far below competitive.
- **Train acc 7.5% < val acc 9.0%** at epoch 60 → severely **under-fitting**, not over-fitting.
- Two compounding root causes:
  1. **Frame duplication bug.** Config had `num_frames: 8` but every video on disk has exactly 4 frames, so `_pick_frame_indices` returned `[0,0,1,1,2,2,3,3]`. The temporal transformer saw 8 tokens with each frame duplicated — half the input was redundant.
  2. **Regularization stack too aggressive for the available signal.** Mixup α=0.2 + label smoothing 0.1 + drop_path 0.1 (×2 places) + MLP dropout 0.5 + EMA 0.9999 + RandAugment + RandomErasing simultaneously suppressed the gradient enough that the model never escaped the label-smoothed-mixup loss floor (~3.31).

## Setup

| Field | Value |
|---|---|
| Model | ViT-S/16 (scratch) → 4-layer Transformer → MLP(768) |
| Spatial pool | `cls_avg` (CLS ⊕ mean of patch tokens, out_dim = 768) |
| drop_path | 0.1 (ViT) + 0.1 (temporal) |
| Temporal | 4 layers, 8 heads, dropout 0.1 |
| MLP head | hidden 768, dropout 0.5 |
| Optimizer | AdamW, lr 3e-4, wd 0.05 |
| Schedule | 5 warmup + cosine over 60 epochs |
| Batch | 16, AMP fp16, grad-clip 1.0 |
| Aug | strong_clip_aug (RRC + ColorJitter + RandAug(2,9) + RandomErasing) |
| Reg | mixup α=0.2, label_smoothing 0.1, EMA 0.9999 |
| Frames | `num_frames=8` (config) → effective 4 unique frames duplicated to 8 |
| Data | 36k train videos, val 8.7k clips |
| Hardware / runtime | A5000 24 GB, **6.5 h wall**, ~1080 MiB peak (very low — could've used a much bigger batch) |

## Trajectory

Loss/acc (selected epochs from `vit_xform.log`):

| Epoch | train loss | train acc | val loss | val acc | event |
|---|---|---|---|---|---|
| 31 | 3.352 | 0.073 | 3.326 | 0.079 | |
| 38 | 3.345 | 0.073 | 3.317 | 0.082 | new best |
| 41 | 3.344 | 0.073 | 3.317 | 0.088 | new best |
| 44 | 3.342 | 0.073 | 3.315 | 0.090 | new best |
| 46 | 3.340 | 0.074 | 3.314 | 0.091 | new best |
| 48 | 3.340 | 0.073 | 3.312 | **0.0922** | last best |
| 50 | 3.339 | 0.075 | 3.312 | 0.090 | plateau |
| 60 | 3.338 | 0.075 | 3.311 | 0.090 | end |

Shape:
- Train loss drifts from 3.35 → 3.34 over 30 epochs — virtually flat. With 33 classes and label smoothing 0.1, the *minimum-achievable* mixup train loss with random labels is ~3.30, so the train loss sat ~0.04 nats above the floor for the full back half of the run.
- Val acc improved monotonically (with noise) until epoch 48, then plateaued. The schedule still had 12 epochs of cosine decay left at that point — extra compute was wasted.
- `val_loss < train_loss` and `val_acc > train_acc`: the model is being asked to *un-smooth* its training labels and *un-mix* its training inputs at evaluation time, so the val side looks better. This pattern is diagnostic of over-regularization rather than over-fitting.

## Diagnosis

### 1. Frame duplication (largest single confound)

`dataset.num_frames=8` interacted with the on-disk reality of 4 frames per clip via:

```python
# src/dataset/video_dataset.py: _pick_frame_indices
positions = torch.linspace(0, num_available - 1, steps=num_frames)
indices = [int(round(float(x))) for x in positions]
# num_available=4, num_frames=8 → [0, 0, 1, 1, 2, 2, 3, 3]
```

The temporal transformer's job — learning a function over the temporal sequence — was structurally sabotaged: half its tokens are exact duplicates of their neighbours. Positional encodings made this worse, not better, because the model had to learn that adjacent positions sometimes carry identical content.

This is also documented in the project memory: clip length is hard-capped at T=4. Future configs should hard-set `num_frames: 4`.

### 2. Regularization budget too high for the data scale

Each individual reg knob is reasonable for ViT-from-scratch on, say, 1M ImageNet images. **Stacking all of them on 36k clips × 4 frames** means the gradient signal that survives mixup mixing + label smoothing flattening + drop_path skipping + MLP dropout + EMA temporal smoothing is too small to drive a 22M-param ViT off random initialization in 60 epochs.

Evidence: train loss is essentially the loss-floor under mixup+smoothing. The model is converging toward "predict the marginal class distribution," not toward "discriminate classes."

### 3. Architecture mismatched to the temporal token budget

A 4-layer Transformer over 5 tokens (CLS + 4 frames) is over-parameterized: each layer has full self-attention and FFN over a sequence of length 5. Two layers would do the same work with half the params and half the depth-wise drop_path skip risk.

### 4. Compute under-utilized

Peak GPU memory was ~1.08 GB on a 24 GB card. Batch size could have been 64–128 with no OOM. Higher batches → tighter gradient estimates → faster convergence and the option to scale lr with √batch.

## What v1 *did* prove

- The pipeline (modular ViT + Transformer + MLP, EMA, mixup, AMP, resume, checkpointing-with-config) runs end-to-end and W&B-logs cleanly. Infra is fine.
- The new TTA + EMA + drop_path code paths exercise without errors over a full 60-epoch run.
- ViT-S/16 from scratch with this regularization recipe is **not** a viable Track-1 contender at this data scale. Score it as "exhausted; move on."

## Lessons → constraints for v2

1. **`num_frames: 4`** — match disk reality; never duplicate frames.
2. **Lower the regularization floor.** Drop mixup, drop EMA, drop label smoothing (or 0.05), drop_path 0 / 0.05, MLP dropout 0.2.
3. **Match temporal depth to token count.** 5 tokens → 2 transformer layers max.
4. **Use the GPU.** Batch 64+, lr scaled accordingly.
5. **Stop the schedule when val acc plateaus.** 60 epochs of cosine when val flatlines at epoch 48 wastes 1.5 GPU-hours.
6. **Re-introduce reg only after you've seen the model overfit on a smaller-reg run** — the right time to add mixup is when train acc > val acc.

## Proposed v2 — `vit_tf_2` (created in this session)

Files: `src/configs/experiment/vit_tf_2.yaml`, `src/configs/model/vit_tf_2.yaml`.

Beyond the table below, v2 also switches the spatial backbone from `vit_s_16` (22M params) to `vit_ti_16` (5.7M). v1 was gradient-bound, not capacity-bound (train acc 7.5% means even ViT-S couldn't fit train), so a smaller model gives more signal per gradient on 36k clips and gets to its ceiling faster.

Goal: get the same backbone over its current floor by removing the regularization that is starving it.

| Knob | v1 | v2 | reason |
|---|---|---|---|
| `num_frames` | 8 (effective 4) | **4** | match disk; eliminate duplicates |
| ViT pool | `cls_avg` | `cls_avg` | keep — patch tokens carry motion |
| ViT drop_path | 0.1 | **0.05** | barely-learning model needs gradient |
| Temporal layers | 4 | **2** | 5 tokens, depth wasted |
| Temporal drop_path | 0.1 | **0.0** | same |
| MLP hidden | 768 | 512 | match temporal out_dim drop |
| MLP dropout | 0.5 | **0.2** | severe; dial down |
| Optimizer | AdamW | AdamW | keep |
| lr | 3e-4 | **6e-4** | bigger batch + less mixup damping |
| Batch | 16 | **64** | use the GPU |
| Epochs | 60 | **40** | shorter; v1 plateaued at 48 |
| Warmup | 5 | 5 | keep |
| Mixup α | 0.2 | **0.0** | off — was eating signal |
| Label smoothing | 0.1 | **0.05** | mild prior, doesn't crush gradient |
| EMA decay | 0.9999 | **0.0** | off — too slow for under-fit model |
| Strong aug | true | **false** (light: RRC(0.7,1.0)+mild ColorJitter, no RandAug, no Erasing) | bring signal back |
| Hflip | off | off | SSv2 direction-sensitive, keep off |
| AMP | fp16 | fp16 | keep |
| Grad clip | 1.0 | 1.0 | keep |

**Expected outcome.** Honest range: **15–25% val acc** at convergence. ViT-from-scratch on 36k 4-frame clips has a real ceiling; this recipe just gets the model *to* that ceiling instead of starving it on the way there. Beating ~25% likely needs MAE/iBOT/V-JEPA pretraining (the documented exp1/exp2/exp3/exp5 routes) or a stronger inductive-bias backbone (TSM-ResNet18, exp4).

**Smoke test (passed in this session, train loss 3.21→2.51 in 7 batches):**
```bash
python src/train.py experiment=vit_tf_2 \
  dataset.max_samples=64 training.epochs=1 training.batch_size=8 wandb.enabled=false
```

**Full run (tmux, per repo convention):**
```bash
tmux new -s vit_tf_2 \
  "python src/train.py experiment=vit_tf_2 2>&1 | tee vit_tf_2.log"
```

## Higher-ceiling alternatives (not v2, but worth flagging)

If the target is "best Track-1 score I can produce now" rather than "iterate the ViT scratch recipe", these dominate v2 in expectation:

1. **`exp4_tsm_resnet18`** (already exists) — ResNet-18 + TSM has way better inductive bias for from-scratch on this data scale. Already configured for 150 epochs SGD. Probably hits 30%+.
2. **TSM-ResNet18 + Transformer head** (does **not** yet exist) — combine the strong scratch trunk from exp4 with the learned temporal aggregation from this experiment family. Modular pieces are already registered (`spatial.name=resnet_tsm`, `temporal.name=transformer`); only a new yaml is needed. Highest-ceiling **net-new** experiment for Track 1 from scratch.
3. **`exp2_mae_transformer` / `exp3_ibot_transformer` / `exp5_vjepa_transformer`** — once the SSL pretraining checkpoints land, these are the documented winning routes.
