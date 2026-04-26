#!/usr/bin/env bash
# Wait for tmux session 'exp4_1' to end, then launch exp5 in tmux 'exp5'.
# Designed to be run inside its own tmux session so SSH drops can't kill it.

set -u
REPO=/Data/challenge_sb
LOG="$REPO/exp5_watcher.log"
TARGET=exp4_1
NEW=exp5
EXP=exp5_tsm_r50_transformer
RUN_LOG="$REPO/exp5.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher started; waiting for tmux session '$TARGET' to end"

# Poll every 30s. tmux has-session returns 0 while the session exists.
while tmux has-session -t "=$TARGET" 2>/dev/null; do
  sleep 30
done

log "$TARGET session ended; waiting 45s for GPU memory release"
sleep 45

# Refuse to clobber an in-flight exp5 session.
if tmux has-session -t "=$NEW" 2>/dev/null; then
  log "ERROR: tmux session '$NEW' already exists; aborting"
  exit 1
fi

# Refuse to launch if a stale resume file would block training (config hash mismatch).
LAST="$REPO/checkpoints/${EXP}_last.pt"
if [ -f "$LAST" ]; then
  log "WARNING: $LAST exists from a prior run; deleting so we start fresh"
  rm -f "$LAST" "$REPO/checkpoints/${EXP}.pt"
fi

cd "$REPO"
log "launching '$NEW' tmux session: experiment=$EXP"
tmux new -d -s "$NEW" "cd $REPO && python src/train.py experiment=$EXP 2>&1 | tee $RUN_LOG"

# Verify launch succeeded.
sleep 5
if tmux has-session -t "=$NEW" 2>/dev/null; then
  log "OK: '$NEW' tmux session is alive"
else
  log "ERROR: '$NEW' tmux session failed to start; check $RUN_LOG"
  exit 2
fi
