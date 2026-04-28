# V-JEPA Stage 2 — supervised finetune

Stage 2 takes the V-JEPA EMA target encoder from Stage 1 and finetunes it,
together with a 2-layer temporal transformer head and a linear classifier, on
the 33-class video classification task. Track 1.

- Entry point: `python src/train.py experiment=exp5_vjepa_transformer`
- Config: [`src/configs/experiment/exp5_vjepa_transformer.yaml`](src/configs/experiment/exp5_vjepa_transformer.yaml)
- Reused model config: [`src/configs/model/vit_mae_transformer.yaml`](src/configs/model/vit_mae_transformer.yaml)
- Output: `checkpoints/exp5_vjepa_transformer.pt` (best by val acc)

## Why no new code

The Stage-1 V-JEPA target encoder is dumped in **the same key layout** as the
MAE and iBOT encoders (`patch_embed.*`, `cls_token`, `blocks.0..11.*`,
`norm.*`, 149 tensors total at ViT-S/16). The Stage-2 spatial encoder
[`ViTMAEEncoder`](src/models/spatial/vit_mae.py) already loads this layout with
`strict=False`. So Stage 2 for V-JEPA is one new YAML file:

```yaml
defaults:
  - override /model: vit_mae_transformer

model:
  spatial:
    checkpoint_path: ${hydra:runtime.cwd}/checkpoints/vjepa_stage1.pt
```

Identical to how `exp3_ibot_transformer.yaml` swaps in the iBOT checkpoint.

## Architecture (modular, unchanged from exp2/exp3)

```
(B, T=4, 3, 224, 224)
        │
        ▼
ViTMAEEncoder (ViT-S/16, init from vjepa_stage1.pt)   ← spatial, per-frame
        │   CLS-token output per frame
        ▼
(B, T, 384)
        │
        ▼
TemporalTransformer (2 layers, 6 heads, dim_ff=1536)  ← order-aware temporal
        │
        ▼
(B, 384)
        │
        ▼
Linear(384 → 33)                                       ← classifier
```

- **Spatial:** 12-block ViT-S/16, CLS-token output per frame, weights initialised
  from the V-JEPA target encoder.
- **Temporal:** 2-layer transformer with `max_len ≥ 4`, dropout 0.1.
- **Classifier:** linear head on the temporal-pooled embedding.

`num_frames=4` (project hard cap).

## Recipe

| Setting | Value | Note |
|---|---|---|
| `lr` (head) | `5.0e-4` | Matches exp3 (iBOT). V-JEPA features are predictive / contextual rather than pixel-reconstructive, so the gentler iBOT-style LR fits better than MAE's 1e-3 starting point. |
| `llrd` (layer-wise LR decay) | `0.65` | Matches exp3. Stronger decay preserves more of the Stage-1 representation in early blocks. |
| `epochs` | 50 | |
| `warmup_epochs` | 5 | |
| `batch_size` | 16 clips × 4 frames = 64 image-passes/iter (fits A5000) |
| `weight_decay` | 0.05 | |
| `label_smoothing` | 0.1 | |
| `grad_clip` | 1.0 | |
| `amp` | `true` (bf16) | |
| `num_workers` | 8 | |

LLRD param groups are built by `utils.build_llrd_param_groups` over the
encoder's `ordered_layers()` (`embed → block_0 → … → block_11 → norm`), with
the head trained at the full base LR.

This recipe is deliberately the same as exp3 so that the **exp2 (MAE) ↔ exp3
(iBOT) ↔ exp5 (V-JEPA)** comparison isolates the pretraining objective —
encoder pretraining is the only axis that varies; head, data, and finetuning
recipe are held fixed.

## Run

**Smoke test first** (always, on this repo):

```bash
python src/train.py experiment=exp5_vjepa_transformer \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4
```

Confirm the encoder load message in the log:

```
[vit_mae] loaded encoder from .../checkpoints/vjepa_stage1.pt
```

with no missing or unexpected keys (the V-JEPA dump matches the ViTMAEEncoder
layout exactly — all 149 tensors).

**Full run, in tmux:**

```bash
tmux new -s exp5 "python src/train.py experiment=exp5_vjepa_transformer 2>&1 | tee exp5.log"
# detach: Ctrl-b d    reattach: tmux attach -t exp5
tail -f exp5.log
```

`train.py` saves the best-by-val-acc checkpoint (with the full merged Hydra
config embedded in the `.pt`), so the next steps need no architecture flags.

## Evaluate / submit

```bash
# Top-1 / top-5 on the held-out val set.
python src/evaluate.py training.checkpoint_path=checkpoints/exp5_vjepa_transformer.pt

# Test-set submission CSV (video_name,predicted_class).
python src/create_submission.py training.checkpoint_path=checkpoints/exp5_vjepa_transformer.pt
```

Both scripts rebuild the model from the config embedded in the checkpoint, so
they pick up the V-JEPA spatial init automatically.

## Comparison protocol

To cleanly compare encoder objectives:

| Experiment | Spatial init | Temporal head | Head recipe | LR | LLRD |
|---|---|---|---|---|---|
| exp2 | MAE Stage-1 | 2-layer transformer | same | 1.0e-3 | 0.75 |
| exp3 | iBOT Stage-1 | 2-layer transformer | same | 5.0e-4 | 0.65 |
| **exp5** | **V-JEPA Stage-1** | **2-layer transformer** | **same** | **5.0e-4** | **0.65** |

Note: exp2 vs exp3/exp5 deliberately differs on LR/LLRD. Encoder vs
finetune-recipe contributions to the deltas are not separable across the
MAE/iBOT boundary; *within* the iBOT/V-JEPA pair, the recipe is identical and
the delta isolates the pretraining objective.

## Variants worth considering later

- **`vit_mae_meanpool` head** (analog to exp1): order-blind temporal pool. If
  V-JEPA already encodes temporal context per-frame via the Δt-aware
  predictor's training signal, mean-pooling might already get most of the
  benefit. Cheap experiment.
- **Higher LR** (e.g., 1e-3) if the run plateaus too low. iBOT-style pretraining
  is typically more "ready" than V-JEPA's; 5e-4 is the conservative starting
  point. Bump if val accuracy is clearly underfitting after warmup.
- **More aggressive LLRD (0.5)** if early blocks are degrading; or weaker
  (0.75) if the encoder needs to adapt more. Watch the train/val gap by epoch.
