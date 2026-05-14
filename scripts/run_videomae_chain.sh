#!/usr/bin/env bash
# Stage 1 (VideoMAE pretrain) → Stage 1 probe → Stage 2a (meanpool) →
# Stage 2b (transformer) → ensemble submission.
# Idempotent: every stage resumes from its `*_last.pt` if present
# (pretrain_videomae.py and train.py both gate on a config hash; mismatch aborts loudly).
# Re-run this script as many times as needed — it picks up where it left off.

set -e
set -u
set -o pipefail

cd /Data/challenge_sb

mkdir -p checkpoints submissions logs

PY="${PY:-/Data/challenge_sb/.venv/bin/python}"

# Fragmentation safety on A5000 — keeps activations + decoder memory predictable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Why: a wandb network blip can stall the training loop until python dies (saw it
# 2026-05-10 on the iBOT chain). Offline mode writes to wandb/offline-run-* and
# never touches the network during training; sync afterwards with
# `wandb sync wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"

STAGE1_CKPT="checkpoints/videomae_stage1.pt"
STAGE2A_CKPT="checkpoints/videomae_meanpool.pt"
STAGE2B_CKPT="checkpoints/videomae_transformer.pt"
STAGE2C_CKPT="checkpoints/videomae_perceiver.pt"
SUBMISSION_MEANPOOL="submissions/videomae_meanpool.csv"
SUBMISSION_STAGE2B="submissions/videomae_stage2b.csv"
SUBMISSION_PERCEIVER="submissions/videomae_perceiver.csv"
SUBMISSION_ENSEMBLE="submissions/videomae_ensemble.csv"

STAGE1_LOG="logs/stage1_videomae_pretrain.log"
PROBE_LOG="logs/videomae_probe_final.log"
STAGE2A_LOG="logs/stage2a_videomae_meanpool.log"
STAGE2B_LOG="logs/stage2b_videomae_transformer.log"
STAGE2C_LOG="logs/stage2c_videomae_perceiver.log"
SUBMIT_LOG="logs/stage_submit_videomae.log"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[chain] $(stamp) === Stage 1: VideoMAE pretrain ==="
"$PY" src/pretrain_videomae.py experiment=videomae_pretrain 2>&1 | tee -a "$STAGE1_LOG"

if [ ! -f "$STAGE1_CKPT" ]; then
  echo "[chain] $(stamp) FATAL: Stage 1 did not produce $STAGE1_CKPT — aborting"
  exit 1
fi

echo "[chain] $(stamp) === Stage 1 health probe (non-blocking) ==="
# Probe is diagnostic only — never abort the chain on probe failures.
"$PY" src/probe_stage1.py \
    --ckpt "$STAGE1_CKPT" \
    --encoder videomae \
    --n-train 5000 \
    --num-workers 4 2>&1 | tee "$PROBE_LOG" || \
    echo "[chain] $(stamp) WARN: probe failed; continuing to Stage 2"

echo "[chain] $(stamp) === Stage 2a: videomae_meanpool fine-tune ==="
if [ -f "$STAGE2A_CKPT" ]; then
  echo "[chain] $(stamp) Stage 2a ckpt already present at $STAGE2A_CKPT — skipping training"
else
  "$PY" src/train.py experiment=videomae_meanpool 2>&1 | tee -a "$STAGE2A_LOG"
  if [ ! -f "$STAGE2A_CKPT" ]; then
    echo "[chain] $(stamp) FATAL: Stage 2a did not produce $STAGE2A_CKPT — aborting"
    exit 1
  fi
fi

echo "[chain] $(stamp) === Stage 2b: videomae_transformer fine-tune ==="
"$PY" src/train.py experiment=videomae_transformer 2>&1 | tee -a "$STAGE2B_LOG"

if [ ! -f "$STAGE2B_CKPT" ]; then
  echo "[chain] $(stamp) FATAL: Stage 2b did not produce $STAGE2B_CKPT — aborting"
  exit 1
fi

echo "[chain] $(stamp) === Stage 2c: videomae_perceiver fine-tune ==="
"$PY" src/train.py experiment=videomae_perceiver 2>&1 | tee -a "$STAGE2C_LOG"

if [ ! -f "$STAGE2C_CKPT" ]; then
  echo "[chain] $(stamp) FATAL: Stage 2c did not produce $STAGE2C_CKPT — aborting"
  exit 1
fi

echo "[chain] $(stamp) === Per-head submissions ==="
"$PY" src/create_submission.py \
    training.checkpoint_path="$STAGE2A_CKPT" \
    dataset.submission_output="$SUBMISSION_MEANPOOL" 2>&1 | tee -a "$SUBMIT_LOG"

"$PY" src/create_submission.py \
    training.checkpoint_path="$STAGE2B_CKPT" \
    dataset.submission_output="$SUBMISSION_STAGE2B" 2>&1 | tee -a "$SUBMIT_LOG"

"$PY" src/create_submission.py \
    training.checkpoint_path="$STAGE2C_CKPT" \
    dataset.submission_output="$SUBMISSION_PERCEIVER" 2>&1 | tee -a "$SUBMIT_LOG"

echo "[chain] $(stamp) === 3-way ensemble submission (softmax-avg) ==="
"$PY" src/ensemble_submissions.py \
    --ckpts "$STAGE2A_CKPT" "$STAGE2B_CKPT" "$STAGE2C_CKPT" \
    --output "$SUBMISSION_ENSEMBLE" 2>&1 | tee -a "$SUBMIT_LOG"

echo "[chain] $(stamp) === Done — ensemble at $SUBMISSION_ENSEMBLE ==="
