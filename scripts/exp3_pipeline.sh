#!/usr/bin/env bash
# Chain: iBOT Stage 1 (pretrain) -> exp3 Stage 2 (classifier).
# Each stage is wrapped in train_robust.sh so transient crashes auto-resume.
# Designed to be re-launchable: each stage's resume picks up from <ckpt>_last.pt.
#
# Usage (always inside tmux):
#   tmux new -s exp3 'bash scripts/exp3_pipeline.sh 2>&1 | tee -a logs/exp3.log'
#
# Stop everything:
#   touch /Data/challenge_sb/STOP && tmux kill-session -t exp3
#
# Resume:
#   rm /Data/challenge_sb/STOP
#   tmux new -s exp3 'bash scripts/exp3_pipeline.sh 2>&1 | tee -a logs/exp3.log'

set -u
cd "$(dirname "$0")/.."

STOP_FILE="${STOP_FILE:-/Data/challenge_sb/STOP}"
mkdir -p logs checkpoints

stop_check() {
    if [ -e "$STOP_FILE" ]; then
        echo "[exp3-pipeline] STOP file present; exiting." >&2
        exit 0
    fi
}

echo "[exp3-pipeline] $(date -Iseconds) starting Stage 1 (iBOT pretrain)"
stop_check
# A5000 (24 GB) can't fit batch_size=256 (the config default) -> halve it.
# base_lr is scaled by batch_size/256 inside pretrain_ibot.py, so LR stays sane.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/train_robust.sh uv run python src/pretrain_ibot.py experiment=ibot_pretrain training.batch_size=128
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[exp3-pipeline] Stage 1 wrapper exited $rc; aborting." >&2
    exit "$rc"
fi
stop_check

if [ ! -f checkpoints/ibot_stage1.pt ]; then
    echo "[exp3-pipeline] Stage 1 finished but checkpoints/ibot_stage1.pt is missing; aborting." >&2
    exit 1
fi

echo "[exp3-pipeline] $(date -Iseconds) starting Stage 2 (exp3_ibot_transformer)"
bash scripts/train_robust.sh uv run python src/train.py experiment=exp3_ibot_transformer
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[exp3-pipeline] Stage 2 wrapper exited $rc." >&2
    exit "$rc"
fi

echo "[exp3-pipeline] $(date -Iseconds) pipeline complete."
