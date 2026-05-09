#!/usr/bin/env bash
# Wrap a training command so transient crashes auto-restart and resume from
# <run>_last.pt. Combine with @reboot crontab for full robustness.
#
# Preferred launch (via scripts/launch.sh): nohup setsid daemon, survives SSH
# drop AND tmux-server death (the latter killed v4 on 2026-05-08):
#   bash scripts/launch.sh <experiment>
#
# Direct invocation (no launcher), if you need it:
#   nohup setsid bash scripts/train_robust.sh python src/train.py experiment=foo \
#       >> logs/foo.log 2>&1 < /dev/null & disown
#
# Stop everything (and prevent any restart, including @reboot):
#   touch /Data/challenge_sb/STOP        # blocks the next retry loop
#   pkill -f 'src/train.py.*experiment=foo'   # kill the running python child
#
# Resume same run later (loads state from _last.pt automatically):
#   rm /Data/challenge_sb/STOP
#   bash scripts/launch.sh <experiment>
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
