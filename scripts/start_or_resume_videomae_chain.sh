#!/usr/bin/env bash
# Idempotent launcher for the VideoMAE chain. Used both for manual launch and
# the reboot hook (cron @reboot). Re-running is always safe:
#   - if /Data/challenge_sb/STOP exists → do nothing (and don't restart)
#   - if the tmux session `videomae_chain` is alive → do nothing
#   - if the chain has already produced the ensemble CSV → do nothing
#   - otherwise (re)create the tmux session and let train_robust.sh +
#     run_videomae_chain.sh resume from *_last.pt

set -u
set -o pipefail

REPO=/Data/challenge_sb
SESSION=videomae_chain
SUBMISSION_CSV="$REPO/submissions/videomae_ensemble.csv"
STOP_FILE="${STOP_FILE:-$REPO/STOP}"
CHAIN_LOG="$REPO/logs/videomae_chain.log"

mkdir -p "$REPO/logs"

if [ -e "$STOP_FILE" ]; then
  echo "[start_or_resume] STOP file present at $STOP_FILE — not launching"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start_or_resume] tmux session '$SESSION' already running — nothing to do"
  exit 0
fi

if [ -f "$SUBMISSION_CSV" ]; then
  echo "[start_or_resume] submission already at $SUBMISSION_CSV — nothing to do"
  exit 0
fi

cd "$REPO"

echo "[start_or_resume] $(date '+%Y-%m-%d %H:%M:%S') launching tmux session '$SESSION'"
tmux new-session -d -s "$SESSION" \
  "bash scripts/train_robust.sh bash scripts/run_videomae_chain.sh 2>&1 | tee -a '$CHAIN_LOG'"

echo "[start_or_resume] tmux session created. Attach with: tmux attach -t $SESSION"
