#!/usr/bin/env bash
# Wrap a training command so transient crashes auto-restart and resume from
# checkpoints/<run>_last.pt. Combine with @reboot crontab for full robustness.
#
# Usage (always inside tmux, so the loop survives SSH drops):
#   tmux new -s mae   'bash scripts/train_robust.sh python src/pretrain_mae.py experiment=mae_pretrain'
#   tmux new -s train 'bash scripts/train_robust.sh python src/train.py         experiment=baseline_pretrained'
#
# Stop everything (and prevent any restart, including @reboot):
#   touch /Data/challenge_sb/STOP        # blocks this loop and the @reboot job
#   tmux kill-session -t <name>          # kills the running python process
#
# Resume same run later (loads state from _last.pt automatically):
#   rm /Data/challenge_sb/STOP
#   tmux new -s <name> 'bash scripts/train_robust.sh <cmd>'
#
# Force a fresh run (three options, pick one):
#   rm checkpoints/<run>_last.pt                              # wipe resume file
#   ... <cmd> training.resume=false                            # one-shot CLI override
#   ... <cmd> training.checkpoint_path=checkpoints/<new>.pt   # new run name
#
# Behavior:
#   - exit 0  → loop breaks (training completed normally)
#   - exit ≠0 → sleep $SLEEP_AFTER_FAIL seconds, then re-run
#   - STOP file present → loop exits 0 immediately

set -u
STOP_FILE="${STOP_FILE:-/Data/challenge_sb/STOP}"
SLEEP_AFTER_FAIL="${SLEEP_AFTER_FAIL:-30}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <command...>" >&2
    exit 64
fi

while true; do
    if [ -e "$STOP_FILE" ]; then
        echo "[robust] STOP file present at $STOP_FILE; exiting." >&2
        exit 0
    fi
    echo "[robust] $(date -Iseconds) launching: $*"
    "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[robust] $(date -Iseconds) command exited 0; training complete."
        exit 0
    fi
    echo "[robust] $(date -Iseconds) command exited $rc; retrying in ${SLEEP_AFTER_FAIL}s (touch $STOP_FILE to abort)." >&2
    sleep "$SLEEP_AFTER_FAIL"
done
