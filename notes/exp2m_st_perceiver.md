# exp2m_st_perceiver — proposal

Architecture-axis change to **exp2k** (current bar: 41.62% real val) on a
single axis: TimeSformer-style **divided space-time attention** in the last
4 ViT-S/16 blocks. Recipe is otherwise identical to exp2k v2.

## Sketch

```
ViT-S/16, 12 blocks, MAE Stage-1 init, T=4

blocks 1..8   (unchanged, pure per-frame spatial)
  Norm → SpatialAttn → + → Norm → MLP → +

blocks 9..12  (NEW: divided space-time, TimeSformer-style)
  reshape (B,T,N+1,D) → (B,N+1,T,D)
  Norm → TemporalAttn (over T=4 tokens at same patch position) → +
  reshape back → (B*T,N+1,D)
  Norm → SpatialAttn (MAE-init)              → +    [timm Block]
  Norm → MLP        (MAE-init)               → +    [timm Block]

head = exp2k Perceiver, unchanged (16 queries × 788 KV)
aux  = predict-next-CLS, unchanged (λ=0.1)
```

## Init (the key risk mitigation)

- Spatial attn + MLP in late blocks: copy MAE Stage-1 weights as in exp2k.
- Temporal MHA QKV: standard PyTorch init.
- **Temporal MHA output projection: zero-init.** t=0 forward is bit-identical
  to exp2k's spatial-only block.
- Temporal pos embed (T per block, learnable): zero-init.
- Temporal MHA softmax runs in fp32 (project-wide stability convention).

## Diff vs exp2k (architecture-only)

| field | exp2k | **exp2m (this)** | why |
|---|---|---|---|
| backbone | per-frame spatial ViT-S | **last 4 blocks gain divided ST attn** | mix across frames *inside* the backbone, not only in the head |
| temporal head | Perceiver (3-layer, 16 queries) | unchanged | clean A/B |
| aux loss | predict-next-CLS λ=0.1 | unchanged | clean A/B |
| seed, LLRD, lr 1.5e-4, WD 0.05, EMA 0.9999, mixup/cutmix, drop_path 0.2, label smoothing, num_frames=4, batch=16, 100 ep, distill (exp2c, α=0.5, T=4), warmup 12, grad_clip 0.5 | exp2k values | **identical** | only the backbone moves |

## Hyperparameters

- `model.spatial.space_time_layers: 4` — last K blocks upgraded.
- `model.spatial.space_time_num_frames: 4` — pinned to dataset.num_frames.
- Temporal MHA dim: 384 (ViT-S width), heads=6, dropout=0 in MHA.
- LLRD: temporal sub-block shares the bucket of its host block (same lr group
  as block_{i}'s spatial half), so 30 param groups same as exp2k.

## Smoke tests (passed)

- bs=4 max_samples=64 ep=1 resume=false: train loss 2.40, val acc 0.4375 on
  toy val, no NaN, no OOM. `[vit_mae] wrapped last 4 block(s) (8..11)`.
- bs=16 max_samples=64 ep=1 resume=false: train loss 2.39 on 64 samples.
  No OOM at full batch. ~5% FLOPs overhead vs exp2k as expected.

## Failure modes to monitor

- NaN at step 0 → temporal init bug. (Should be impossible: out_proj zeroed.)
- Val_acc ≤ exp2k by epoch 10 → MAE features being disrupted by temporal mix.
  Fall back to K=2, or freeze blocks 1..8 entirely.
- Flat val_acc → temporal lr too low; bump LLRD on those buckets.

## Launch

```
nohup setsid bash scripts/train_robust.sh \
  python src/train.py experiment=exp2m_st_perceiver \
  > /Data/challenge_sb/logs/exp2m_st_perceiver.log 2>&1 < /dev/null &
disown
```

Same robust pattern as exp2k v2 final segment (parent = PID 1, immune to
tmux/SSH death). @reboot crontab from exp2k still active.

## Mid-train submission (2026-05-10 13:47)

- Training still in progress (epoch 50/100, PID 306112) — not interrupted.
- Source: `checkpoints/exp2m_st_perceiver.pt` (best-by-val so far, val_acc=0.4660 from earlier today). Snapshotted to `checkpoints/exp2m_st_perceiver.snapshot.pt` to avoid race with training; ran inference on the snapshot at `training.batch_size=4` to keep GPU headroom.
- Output: `processed_data/submission_exp2m_mid_train.csv` (6913 rows).
- Sanity: 32 unique classes predicted (class 27 correctly absent), top-1 class is 30 (605 preds), min nonzero class is 26 (19 preds). Distribution is broad — no pathological collapse.
- Final submission will still be created by `scripts/submit_after_exp2m.sh` once training exits, against the (possibly-improved) `exp2m_st_perceiver.pt`.
