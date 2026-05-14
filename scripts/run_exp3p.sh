#!/usr/bin/env bash
# Standalone exp3p runner: train → SWA-aware eval → submission → pseudo-label
# fine-tune → eval → submission → final ensemble.
#
# Idempotent: each stage gated on its output. Re-run safely after a crash.

set -e
set -u
set -o pipefail

cd /Data/challenge_sb
mkdir -p checkpoints submissions logs

PY="${PY:-/Data/challenge_sb/.venv/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

EXP3P_CKPT="checkpoints/videomae_exp3p.pt"
EXP3P_SWA_CKPT="checkpoints/videomae_exp3p_swa.pt"
EXP3P_PL_CKPT="checkpoints/videomae_exp3p_pl.pt"

SUB_EXP3P="submissions/videomae_exp3p.csv"
SUB_EXP3P_SWA="submissions/videomae_exp3p_swa.csv"
SUB_EXP3P_PL="submissions/videomae_exp3p_pl.csv"
SUB_FINAL_ENSEMBLE="submissions/videomae_exp3p_ensemble.csv"

TRAIN_LOG="logs/exp3p_train.log"
PL_LOG="logs/exp3p_pl.log"
SUBMIT_LOG="logs/exp3p_submit.log"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[exp3p] $(stamp) === Stage 2: videomae_exp3p main training ==="
if [ -f "$EXP3P_CKPT" ]; then
  echo "[exp3p] $(stamp) main ckpt already at $EXP3P_CKPT — skipping"
else
  "$PY" src/train.py experiment=videomae_exp3p 2>&1 | tee -a "$TRAIN_LOG"
  if [ ! -f "$EXP3P_CKPT" ]; then
    echo "[exp3p] $(stamp) FATAL: training did not produce $EXP3P_CKPT — aborting"
    exit 1
  fi
fi

echo "[exp3p] $(stamp) === Submission: main best.pt ==="
if [ ! -f "$SUB_EXP3P" ]; then
  "$PY" src/create_submission.py \
      training.checkpoint_path="$EXP3P_CKPT" \
      dataset.submission_output="$SUB_EXP3P" 2>&1 | tee -a "$SUBMIT_LOG"
fi

if [ -f "$EXP3P_SWA_CKPT" ] && [ ! -f "$SUB_EXP3P_SWA" ]; then
  echo "[exp3p] $(stamp) === Submission: SWA ckpt ==="
  "$PY" src/create_submission.py \
      training.checkpoint_path="$EXP3P_SWA_CKPT" \
      dataset.submission_output="$SUB_EXP3P_SWA" 2>&1 | tee -a "$SUBMIT_LOG"
fi

echo "[exp3p] $(stamp) === Pseudo-label fine-tune ==="
if [ -f "$EXP3P_PL_CKPT" ]; then
  echo "[exp3p] $(stamp) PL ckpt already at $EXP3P_PL_CKPT — skipping"
else
  # Prefer SWA ckpt as the teacher for pseudo-label generation if available.
  PL_SOURCE_CKPT="$EXP3P_CKPT"
  if [ -f "$EXP3P_SWA_CKPT" ]; then
    PL_SOURCE_CKPT="$EXP3P_SWA_CKPT"
  fi
  "$PY" src/pseudo_label_finetune.py \
      training.checkpoint_path="$PL_SOURCE_CKPT" \
      pseudo_label.output_ckpt="$EXP3P_PL_CKPT" \
      pseudo_label.threshold=0.85 \
      pseudo_label.epochs=10 \
      pseudo_label.lr=5.0e-5 \
      2>&1 | tee -a "$PL_LOG"
fi

if [ -f "$EXP3P_PL_CKPT" ] && [ ! -f "$SUB_EXP3P_PL" ]; then
  echo "[exp3p] $(stamp) === Submission: pseudo-label ckpt ==="
  "$PY" src/create_submission.py \
      training.checkpoint_path="$EXP3P_PL_CKPT" \
      dataset.submission_output="$SUB_EXP3P_PL" 2>&1 | tee -a "$SUBMIT_LOG"
fi

echo "[exp3p] $(stamp) === Final ensemble (best 3 of: meanpool, transformer, perceiver, exp3p, exp3p_swa, exp3p_pl) ==="
ENSEMBLE_CKPTS=()
for c in checkpoints/videomae_exp3p_pl.pt \
         checkpoints/videomae_exp3p_swa.pt \
         checkpoints/videomae_exp3p.pt \
         checkpoints/videomae_perceiver.pt \
         checkpoints/videomae_transformer.pt \
         checkpoints/videomae_meanpool.pt ; do
  if [ -f "$c" ] && [ "${#ENSEMBLE_CKPTS[@]}" -lt 5 ]; then
    ENSEMBLE_CKPTS+=("$c")
  fi
done
if [ "${#ENSEMBLE_CKPTS[@]}" -ge 2 ]; then
  "$PY" src/ensemble_submissions.py \
      --ckpts "${ENSEMBLE_CKPTS[@]}" \
      --output "$SUB_FINAL_ENSEMBLE" 2>&1 | tee -a "$SUBMIT_LOG"
fi

echo "[exp3p] $(stamp) === Done. Submissions: $SUB_EXP3P, ${SUB_EXP3P_SWA:-(no swa)}, ${SUB_EXP3P_PL:-(no pl)}, $SUB_FINAL_ENSEMBLE ==="
