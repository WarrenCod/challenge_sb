# exp3a_vivit_pairwise_multiclip — proposal

Stage-2 v3 on top of exp2n (real val 0.4746). exp2n curve peaked ep 41/70 and
declined through ep 70 → recipe ceiling on this architecture is ~0.475.
Path to 55% needs a **structural** lift, not more reg.

Four orthogonal levers, each on a different axis exp2n didn't move:

1. **K=4 → K=6 divided ST blocks** in the backbone. Zero-init residual; step-0
   forward unchanged. Mirrors the exp2k→exp2m lever (+5.27 pp) that was the
   single largest historical win.
2. **Perceiver: sinusoidal+learnable_scale temporal pos, 3→4 layers.** exp2n
   had a zero-init learnable temporal pos → permutation-symmetric at step 0.
   Sinusoidal init means the head is order-aware from the first forward.
3. **Pairwise direction head (PDH).** 6 ordered (i<j) frame-CLS pairs →
   classify the action from `[cls_i, cls_j, Δ, Σ]` → 33-class CE, λ=0.3.
   Replaces exp2n's predict-next + predict-prev (which were cosine-on-CLS,
   weak signal). PDH puts a *classification* gradient on pairwise frame
   deltas — directly aligned with "what happens next".
4. **Multi-clip consistency.** 2 augmented views per video, symmetric KL
   between predictions (λ_cons=0.2). Mixup/cutmix applied per half with
   shared (lam, perm) so the pairing stays intact. Doubles per-video signal
   without violating num_frames=4.

Distill: born-again ENSEMBLE teacher = softmax-mean of {exp2m, exp2n_bare},
α=0.5, T=4. SWA dropped (wash on exp2n).

## Sketch

```
Stage 1 (frozen): MAE Stage-1 ViT-S/16  →  checkpoints/mae_stage1.pt

Stage 2 (exp3a):

  Spatial:    ViT-S/16 + K=6 divided ST blocks (was K=4)
              drop_path linear 0.0→0.30, per-block

  Temporal:   PerceiverHead
              ├─ 16 learnable queries (std=0.02)
              ├─ temporal_pos: sinusoidal init + learnable scalar  ← NEW
              ├─ cross-attn (1×) Q ←→ KV=788 tokens (fp32 softmax)
              └─ self-attn encoder, 4 layers (was 3)
              + LayerNorm + mean-pool over queries → 384-d video vec

  Classifier: Linear(384, 33)

  Aux heads (training-only):
    └─ PairDirectionAuxHead  (λ=0.3)
       6 pairs (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
       feat = [cls_i, cls_j, cls_j-cls_i, cls_i+cls_j]
       MLP(4D=1536 → 768 → 33), zero-init fc_out
       average per-pair logits → CE w/ label_smoothing=0.15

  Training-time signals:
    • main CE (label_smoothing=0.15)
    • PDH CE                                       λ_pair  = 0.3
    • multi-clip consistency: 0.5*(KL(a||b)+KL(b||a))   λ_cons  = 0.2
    • distill KD from ensemble{exp2m, exp2n}       α       = 0.5, T=4
```

Step-0 invariant: PDH (zero-init `fc_out`) emits zero logits → CE≈log(33)≈3.5
bounded by λ_pair=0.3 → ~1.0 contribution at init. K=6 ST blocks all
zero-init residual → output unchanged vs exp2n architecture. Sinusoidal pos is
not zero (intentional: order-aware) but is bounded ‖pos‖≤1.

## Diff vs exp2n (full table)

| field | exp2n | **exp3a** | rationale |
|---|---|---|---|
| backbone | ViT-S/16 + K=4 ST | **K=6 ST** | broader temporal coverage |
| Perceiver depth | 3 | **4** | bigger relational head |
| Perceiver temporal pos | zero-init learnable | **sinusoidal+learnable_scale** | order-aware @ step 0 |
| distill teacher | exp2m | **[exp2m, exp2n] softmax_mean** | banked ensemble gain |
| distill α | 0.6 | **0.5** | leaner KD (PDH carries more aux mass) |
| predict_next_cls | λ=0.05 | **off** | superseded by PDH |
| predict_prev_cls | λ=0.05 | **off** | superseded by PDH |
| pair_direction | — | **λ=0.3** | NEW: directional aux |
| multi_clip | — | **2 clips, λ_cons=0.2** | NEW: consistency reg |
| SWA | ep 35..70 | **off** | wash on exp2n |
| epochs | 70 | **55** | exp2n peaked ep 41; trim tail |
| warmup | 8 | 8 | unchanged |
| batch_size | 16 | **8 (×2 clips=16)** | same per-step optimizer count |
| reg knobs (drop_path/dropout/LS/LLRD) | exp2n | unchanged | exp2n's reg held; don't perturb |
| mixup/cutmix/EMA/grad_clip/AMP | exp2n | unchanged | |
| TTA at submission | 5 | 5 | unchanged (temporal-only; 5-crop deferred) |

## Honest expected gain (with 50% overlap discount)

| lever | optimistic | conservative |
|---|---|---|
| K=4 → K=6 ST | +1.5 | +0.3 |
| Perceiver sinusoidal + 4 layers | +1.5 | +0.3 |
| Pair direction head | +2.0 | +0.5 |
| Multi-clip consistency | +3.0 | +0.5 |
| Ensemble teacher | +1.0 | +0.2 |
| **Sum** | +9.0 | +1.8 |
| **× 50% overlap** | **+4.5** | **+0.9** |

Landing band: **0.49 floor → 0.53 mid → 0.55 stretch** real val. The 55%
hard target likely needs an exp3b (FixMatch on test data) or exp3c (ensemble
distillation including exp3a) follow-on.

## Failure modes & fallbacks

1. Consistency-KL collapse (both halves converge to uniform). Detect: KL
   < 1e-3 within 5 ep while CE flat. Fix: λ_cons 0.2 → 0.1, or one-sided KL.
2. PDH dominates main CE. Detect: PDH-loss > 1.5× main CE @ ep 5. Fix: halve
   λ_pair to 0.15.
3. Ensemble teacher overconfidence drag. Detect: val_acc lags exp2n curve by
   ep 20. Fix: α 0.5 → 0.3.
4. OOM at bs=8×2. Fix: bs=6×2; halve PDH hidden_dim to 512.

## Run log

- *(awaiting smoke + launch)*
