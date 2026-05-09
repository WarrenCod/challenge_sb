#!/usr/bin/env bash
# Robust launcher for exp11_flow_only. Used both for the initial launch and as
# an @reboot helper. Wraps src/train.py in train_robust.sh inside a *detached*
# tmux session, so:
#   - SSH drops / terminal closes cannot SIGHUP the run (tmux is detached).
#   - Python crashes / SIGKILL trigger a 30s back-off and resume from
#     checkpoints/exp11_flow_only_last.pt (training.resume=true is on).
#   - W&B id="exp11_flow_only" + resume="allow" appends to the existing run.
#
# Crontab entry (user crontab):
#   @reboot /Data/challenge_sb/scripts/exp11_reboot.sh >> /Data/challenge_sb/logs/exp11_reboot.log 2>&1
#
# Stop everything (kill switch shared with train_robust.sh and other helpers):
#   touch /Data/challenge_sb/STOP

set -u
REPO=/Data/challenge_sb
STOP_FILE=$REPO/STOP
SESSION=exp11
LOG=$REPO/exp11.log

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$REPO/logs"
cd "$REPO"

echo "[exp11-reboot] $(date -Iseconds) waking up"

if [ -e "$STOP_FILE" ]; then
    echo "[exp11-reboot] STOP file present at $STOP_FILE; refusing to relaunch."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[exp11-reboot] tmux session '$SESSION' already running; nothing to do."
    exit 0
fi

for _ in $(seq 1 12); do
    if nvidia-smi -L >/dev/null 2>&1; then break; fi
    sleep 5
done

tmux new -d -s "$SESSION" \
    "bash $REPO/scripts/train_robust.sh uv run python $REPO/src/train.py experiment=exp11_flow_only 2>&1 | tee -a $LOG"
echo "[exp11-reboot] launched detached tmux session '$SESSION'."
