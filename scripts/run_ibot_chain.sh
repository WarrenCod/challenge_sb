#!/usr/bin/env bash
# Stage 1 (iBOT pretrain v2) → Stage 2 (ibot_transf_4) → submission.
# Idempotent: every stage resumes from its `*_last.pt` snapshot if present
# (pretrain_ibot.py and train.py both gate on a config hash; mismatch aborts loudly).
# Re-run this script as many times as needed — it picks up where it left off.

set -e
set -u
set -o pipefail

cd /Data/challenge_sb

mkdir -p checkpoints submissions logs

PY="${PY:-/Data/challenge_sb/.venv/bin/python}"

# Fragmentation safety on A5000 — keeps multi-crop memory predictable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE1_CKPT="checkpoints/ibot_stage1.pt"
STAGE2_CKPT="checkpoints/ibot_transf_4.pt"
SUBMISSION_CSV="submissions/ibot_transf_4.csv"

STAGE1_LOG="logs/stage1_ibot_pretrain_v2.log"
STAGE2_LOG="logs/stage2_ibot_transf_4.log"
SUBMIT_LOG="logs/stage_submit.log"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[chain] $(stamp) === Stage 1: iBOT pretrain v2 ==="
"$PY" src/pretrain_ibot.py experiment=ibot_pretrain_v2 2>&1 | tee -a "$STAGE1_LOG"

if [ ! -f "$STAGE1_CKPT" ]; then
  echo "[chain] $(stamp) FATAL: Stage 1 did not produce $STAGE1_CKPT — aborting"
  exit 1
fi

echo "[chain] $(stamp) === Stage 2: ibot_transf_4 fine-tune ==="
"$PY" src/train.py experiment=ibot_transf_4 2>&1 | tee -a "$STAGE2_LOG"

if [ ! -f "$STAGE2_CKPT" ]; then
  echo "[chain] $(stamp) FATAL: Stage 2 did not produce $STAGE2_CKPT — aborting"
  exit 1
fi

echo "[chain] $(stamp) === Submission ==="
"$PY" src/create_submission.py \
    training.checkpoint_path="$STAGE2_CKPT" \
    dataset.submission_output="$SUBMISSION_CSV" 2>&1 | tee -a "$SUBMIT_LOG"

echo "[chain] $(stamp) === Done — submission at $SUBMISSION_CSV ==="
