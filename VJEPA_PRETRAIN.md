# V-JEPA Stage 1 — frame-pair pretraining

Self-supervised pretraining of a ViT-S/16 spatial encoder on the challenge's own
training clips, with no external data and no ImageNet weights (Track 1 rules).
Stage 1 produces the encoder weights that Stage 2 finetunes for the 33-class
classifier.

- Entry point: `python src/pretrain_vjepa.py experiment=vjepa_pretrain`
- Code: [`src/pretrain_vjepa.py`](src/pretrain_vjepa.py), [`src/models/vjepa.py`](src/models/vjepa.py), [`src/dataset/vjepa_dataset.py`](src/dataset/vjepa_dataset.py)
- Config: [`src/configs/experiment/vjepa_pretrain.yaml`](src/configs/experiment/vjepa_pretrain.yaml)
- Output: `checkpoints/vjepa_stage1.pt` (best by train loss) + `checkpoints/vjepa_stage1_last.pt`

## Why V-JEPA, on this dataset

MAE (pixel reconstruction) and iBOT (token-discriminative) both proved viable
in earlier experiments (exp1/exp2/exp3). V-JEPA targets the *temporal*
prediction signal that those objectives ignore: predicting the **embedding of
one frame from another frame** of the same clip. This pushes the encoder to
encode *what is constant about a moment in this video* rather than the photometric
content of any single frame. Because SSv2 labels are direction-sensitive, that
inductive bias matches the downstream task closely.

Tier-0 means: the simplest viable JEPA. We predict frame-to-frame in feature
space; we do not do tubelet/spatial masking on top. The hard cap of 4 frames
(see project memory) makes a frame-pair formulation natural.

## Objective

Per training step, for each clip in the batch:

1. Sample two distinct frame indices `(i, j)`, `i ≠ j`, in `{0, …, T-1}`.
2. **Context encoder** `f_θ` (trainable ViT-S/16) runs on frame `i`.
3. **Target encoder** `f_θ̄` (EMA of `f_θ`, no grad) runs on frame `j`.
4. **Predictor** `g_φ` takes `z_i` plus a Δt embedding and predicts `f_θ̄`'s
   patch tokens for frame `j`.
5. Loss: `smooth_L1( LN(z_pred), LN(z_target).detach() )` with `β = 2.0`.

Both prediction and target pass through a **parameter-free LayerNorm** before
the loss. This makes the matching scale-invariant, which is the standard
defence against representation collapse in JEPA-style training. We also log
`cos_sim` (alignment) and `target_std` (per-dim std of the target after LN) as
collapse-monitor signals.

## Architecture

| Component | Spec |
|---|---|
| Context encoder | ViT-S/16, 12 blocks, depth=12, heads=6, embed=384, patch=16, image=224. Outputs `(B, 1+196, 384)`: CLS + patch tokens. |
| Target encoder | Same architecture; weights are EMA-updated from the context encoder, gradients off. |
| Predictor | "Narrow" 4-block transformer at width 128 (`predictor_dim=128`, `predictor_depth=4`, `predictor_heads=4`). `proj_in` 384→128, learnable `mask_token` queries, `dt_embed` lookup, sin-cos positional, LN, `proj_out` 128→384. Concat-and-self-attend pattern: input is `[context_tokens; query_tokens]`, output reads only the query positions. |

Why a *narrow* predictor: V-JEPA papers report that giving the predictor too
much capacity lets it model the input rather than be forced to ride on the
target encoder's representation. Width 128 / depth 4 keeps it small relative to
the 384-wide encoder.

The Δt embedding is a learnable 6-entry table (`n_dt = 2·(T-1) = 6` for T=4),
indexed by signed Δt with the zero-bin removed. So the same predictor handles
forward and backward prediction at distances 1, 2, 3 frames.

The EMA target encoder is saved in **the same key layout as MAE / iBOT**
(`patch_embed.*`, `cls_token`, `blocks.*`, `norm.*`) — see
`VJEPAEncoder.encoder_state_dict` — so Stage 2 reuses the existing
`ViTMAEEncoder` loader without any renaming.

## Augmentation (clip-consistent)

Same `RandomResizedCrop` coordinates, same `ColorJitter` factors, same
grayscale decision, and same Gaussian-blur radius are applied to **every frame
of a clip**. The only thing that varies across frames within a clip is the
actual underlying motion — which is precisely the signal V-JEPA learns from.

**No horizontal or vertical flip.** SSv2 labels are direction-sensitive, so
flipping silently mislabels data; this rule applies everywhere in the project.

## Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW, no weight decay on 1-D params (bias, LayerNorm, CLS / mask tokens, pos / dt embeds) |
| Effective LR | `base_lr · batch_size / 256 = 1.5e-4` at `batch_size=256` |
| LR schedule | Linear warmup over 10 epochs, cosine decay to `final_lr = 1e-6` |
| Weight decay | 0.05 |
| Batch size | 256 clips |
| Gradient clip | L2 = 3.0 |
| Mixed precision | `bf16` autocast (no GradScaler — bf16 is range-stable) |
| Epochs | 200 (with early stop) |
| EMA momentum | Cosine schedule from `0.996` to `1.0` over the run |
| Loss | smooth-L1, `β = 2.0` |

**Robustness guards:**
- **NaN/Inf guard.** If the loss is non-finite at a step, the backward pass,
  optimizer step, and EMA update are all skipped for that step. After
  `max_nonfinite_streak = 50` consecutive non-finite steps, training aborts —
  the predictor is wedged.
- **Early stop.** If best train loss hasn't improved by `min_delta = 1e-4` for
  `patience = 25` epochs, training stops. Floored at `min_epochs = 60` so we
  never bail before warmup + the early plateau has settled.

## Run / monitor / resume

Always launch under tmux so an SSH drop can't kill the run:

```bash
tmux new -s vjepa "python src/pretrain_vjepa.py experiment=vjepa_pretrain 2>&1 | tee jepa.log"
# detach: Ctrl-b d   reattach: tmux attach -t vjepa
tail -f jepa.log
```

The script writes both a best-by-loss checkpoint (`vjepa_stage1.pt`) and a
last-epoch checkpoint (`vjepa_stage1_last.pt`). Resume picks up from
`*_last.pt` automatically; the resume hash guards against config drift on
restart.

Smoke test before any long run:

```bash
python src/pretrain_vjepa.py experiment=vjepa_pretrain \
  dataset.max_samples=64 training.epochs=1 training.batch_size=8
```

## Output schema

`checkpoints/vjepa_stage1.pt` is a single `dict`:

```python
{
  "encoder_state_dict": {  # 149 tensors, ViT-S/16, MAE/iBOT key layout
      "cls_token": ..., "patch_embed.proj.weight": ...,
      "blocks.0.norm1.weight": ..., ..., "norm.weight": ..., "norm.bias": ...,
  },
  "config": {...},                # full merged Hydra cfg, for reproducibility
  "epoch": 178,                   # checkpoint epoch
  "vjepa_train_loss": 0.0093,     # best loss at save time
}
```

`Stage 2 → src/models/spatial/vit_mae.py::ViTMAEEncoder._load_mae_checkpoint`
reads `encoder_state_dict` directly with `strict=False`. No adapter required.

## Health signals to watch in W&B / `jepa.log`

- **`loss`** — should decrease smoothly through warmup and continue dropping
  for ~60–100 epochs before plateauing in the low 0.01 range. Floors near
  0.009 on this dataset.
- **`cos_sim`** — cosine similarity between predicted and target patch tokens
  after LN. Climbs to ~0.98 by mid-training. If it pegs near 1.0 *too early*,
  suspect representation collapse.
- **`target_std`** — per-dim std of LN'd target tokens. If it crashes toward
  zero, the target encoder is collapsing — check EMA momentum and predictor
  capacity.
- **non-finite streak** — should stay at 0; non-zero values are logged.

## Following stage

After Stage 1 finishes (or early-stops), the encoder is loaded by Stage 2
finetuning — see [`VJEPA_STAGE2.md`](VJEPA_STAGE2.md).
