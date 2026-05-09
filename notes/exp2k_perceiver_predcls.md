# exp2k_perceiver_predcls — proposal

Architecture-axis change to exp2h on **two coupled axes** chosen on the same
information-bandwidth/inductive-bias theme:

1. **A — full per-frame patch tokens to the head.** exp2h sees only the
   per-frame CLS (1 vector × T frames). exp2k passes the whole token
   sequence (1 CLS + 196 patch tokens per frame, T=4 → 788 tokens) to a
   **Perceiver-style head**: 16 learnable queries cross-attend to all 788
   KV tokens, then a 3-layer self-attention tower runs on the queries,
   mean pool, linear classifier.
2. **D — predict-next-CLS auxiliary loss.** From frames 1..3 spatial CLS,
   predict frame 4's CLS via a 2-layer MLP (3·D → 512 → D); cosine loss
   vs `stopgrad(CLS_4)`. Weight λ_pred = 0.1. Task-matched inductive bias
   for "What Happens Next?".

## Sketch

```
spatial backbone (exp2h ViT-S/16, MAE-init, drop_path=0.2)
        │  return_all_tokens=True
        ▼  per frame: [CLS, p1..p196] (D=384)
flatten -> KV: (B, T*(N+1), D) = (B, 788, 384)
        + temporal pos embed (T=4, learnable, zero-init)
        │
        ▼
[16 learnable queries (D=384)] ── cross-attention ── KV
        │
        ▼
3-layer self-attn block on the 16 queries
        │
        ▼
mean pool over 16 -> Linear -> 33 logits  (main classifier)

Aux head (independent path):
[CLS_1, CLS_2, CLS_3] -> 2-layer MLP (3D -> 512 -> D) -> predicted CLS_4
        ↘ 1 - cos vs stopgrad(CLS_4)
```

## Diff vs exp2h (only architecture + aux changes; recipe identical)

| field | exp2h | **exp2k (this)** | why |
|---|---|---|---|
| spatial output | per-frame CLS (B, T, D) | **full tokens (B, T, N+1, D)** | 100× more spatial-temporal information |
| temporal head | RelationalTransformer (3 layers, 8 tokens incl. diff tokens) | **Perceiver (3 layers self-attn on 16 queries; 1 cross-attn over 788 KV)** | exposes per-position patterns, compresses to small learnable code |
| aux loss | none | **predict-next-CLS (cosine, λ=0.1)** | task-matched; predicts the future, which is the challenge |
| seed, LLRD, lr, WD, EMA, cutmix, label smoothing, num_frames=4, batch=16, 100 ep, drop_path=0.2, distill (exp2c, α=0.5, T=4) | exp2h values | **identical** | clean A/B; only the head + aux move |

## Hyperparameters

- Perceiver: num_queries=16, num_heads=6, num_layers=3, mlp_ratio=4.0,
  dropout=0.2, max_frames=4. Temporal pos embed learnable, zero-init.
- Aux: hidden=512, num_input_frames=3 (predict frame 4 from frames 1..3),
  λ=0.1, output layer zero-init.
- All other recipe params identical to exp2h.

## Smoke test (passed)

- bs=4 max_samples=64 epochs=1: train loss 2.44, finite, no OOM, MAE init
  loaded cleanly, predict-next-CLS aux engaged, distill teacher loaded.
- bs=16 max_samples=64 epochs=1: train loss 2.40, no OOM at full batch.
- LLRD picked up the new params: 30 param groups (vs exp2h's similar ~30).

## Result

### v1 attempt (2026-05-07 19:13 → 2026-05-08 00:33) — diverged

- Healthy through epoch 9: train acc 0.0668→0.0692, train loss 2.21→2.22.
- **Diverged at epoch 10**: train loss → NaN mid-epoch, acc collapsed to
  chance (~0.024). Only "new best" ever saved was epoch 1, val_acc=0.0532.
- Loop kept running ~50 NaN epochs before manual kill; `_last.pt` got
  overwritten with NaN weights every epoch.
- Likely cause: fp16 cross-attention softmax over T·(N+1)=788 KV tokens
  overflows; predict-next-CLS cosine on fp16 backbone CLS amplifies it.
  No NaN guard in train.py to short-circuit the run.

### v2 fixes (2026-05-08)

Code:
- `perceiver._CrossAttentionBlock.forward` runs the MultiheadAttention call
  in fp32 (autocast disabled).
- `predict_next_cls` MLP + cosine in fp32; explicit `F.normalize(eps=1e-6)`
  on pred and target before the dot-product cosine.
- `train.py`: per-step `torch.isfinite(loss)` skip + 20-streak abort; per-epoch
  refusal to overwrite `_last.pt` if `train_loss`/`val_loss` is non-finite,
  exits 2 so the watchdog (`scripts/train_robust.sh`) re-launches and resume
  picks up the previous healthy `_last.pt`.

Recipe:
- lr 3e-4 → 1.5e-4
- warmup_epochs 8 → 12
- grad_clip 1.0 → 0.5
- everything else (LLRD, EMA, mixup/cutmix, distill, 100 ep, num_frames=4,
  drop_path 0.2) unchanged.

### v2 smoke test (passed)

bs=4 max_samples=64 epochs=1 resume=false: train loss 2.40 (finite), val loss
3.50, no NaN, checkpoints saved cleanly.

### v2 run

- Launched 2026-05-08 ~14:24 under tmux + train_robust.sh.
- Healthy through epoch 67 (train_loss 1.14, in-train val_loss 2.53,
  in-train val_acc 0.4136, **best in-train val_acc 0.4165**). No NaN, no
  drift, log shows steady improvement.
- **Interrupted 2026-05-08 ~22:00** mid-epoch 68 — log truncated cleanly
  with no traceback / no `[robust]` retry message → tmux server died,
  taking the watchdog with it. (Same failure mode logged in
  `feedback_tmux_training.md` from 2026-05-08.) Run was idle 2026-05-08
  22:00 → 2026-05-09 13:12.
- **Resumed 2026-05-09 13:12** under `nohup setsid bash scripts/train_robust.sh
  …` so the watchdog's parent is PID 1, immune to tmux/SSH death. Added
  matching `@reboot` crontab for host-reboot resilience.
- `_last.pt` at epoch 67 is finite, train.py picked it up cleanly:
  `resuming from epoch 68/100 (best so far 0.4165)`. ~32 epochs (~3 h)
  remaining at 8 it/s.

### v2 final result (2026-05-09 16:13, exit 0)

- Run finished cleanly. 100 epochs, no further NaN.
- **Best in-train val_acc 0.4165** (from epoch ~63, pre-tmux-death; the
  resumed 68→100 segment did not beat it).
- Last epoch 100/100: train loss 1.067, train acc 0.4970 / val loss 2.591,
  val acc 0.4031 (slight overfit at the tail; EMA + best-by-val pick is fine).
- **Real held-out val_dir: top-1 0.4162, top-5 0.7364** (full 6745 videos, no TTA).
  Matches the in-train metric — unusual relative to past runs where in-train
  was inflated; this recipe generalizes cleanly.
- vs prior best on real val: exp2h was 0.39x; exp2k delivers ≈+2 pp from
  full-token Perceiver + predict-next-CLS aux. New high-water mark.

### Submission

- File: `submissions/exp2k_perceiver_predcls.csv` (6913 rows, 32 distinct
  classes; class 27 correctly absent).
- Inference: 3-clip TTA (offset_frac ∈ {0.0, 0.333, 0.667}), softmax-averaged.
- Top class skew: class 30 = 521 preds (~2.4× uniform mean of 216) — within
  normal SSv2-style class-imbalance range.

### Inference-script fix (alongside this run)

`evaluate.py` and `create_submission.py` rebuild the model via
`build_model(cfg)`, but the `predict_next_cls_head` aux module is attached
in `train.py` *after* `build_model`, so the checkpoint state_dict carried 4
extra keys that broke strict-load. Both scripts now load with `strict=False`
and print a one-liner listing dropped (aux) keys + any genuinely missing
ones. Aux heads are training-only and never needed at inference.
