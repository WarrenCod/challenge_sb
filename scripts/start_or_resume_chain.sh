#!/usr/bin/env bash
# Idempotent launcher for the iBOT chain. Used both for manual launch and for the
# reboot hook (systemd / cron @reboot). Re-running is always safe:
#   - if the tmux session `ibot_chain` is alive, do nothing
#   - if the chain has already produced the final submission CSV, do nothing
#   - otherwise (re)create the tmux session and let run_ibot_chain.sh resume

set -u
set -o pipefail

REPO=/Data/challenge_sb
SESSION=ibot_chain
SUBMISSION_CSV="$REPO/submissions/ibot_transf_4.csv"
CHAIN_LOG="$REPO/logs/chain.log"

mkdir -p "$REPO/logs"

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
  "bash scripts/run_ibot_chain.sh 2>&1 | tee -a '$CHAIN_LOG'"

echo "[start_or_resume] tmux session created. Attach with: tmux attach -t $SESSION"
