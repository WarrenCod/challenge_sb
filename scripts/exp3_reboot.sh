#!/usr/bin/env bash
# @reboot helper: re-launch the exp3 pipeline inside tmux if not already running.
# Skips when /Data/challenge_sb/STOP exists (kill switch shared with train_robust.sh).
#
# Crontab entry (user crontab):
#   @reboot /Data/challenge_sb/repo/scripts/exp3_reboot.sh >> /Data/challenge_sb/repo/logs/exp3_reboot.log 2>&1

set -u
REPO=/Data/challenge_sb/repo
STOP_FILE=/Data/challenge_sb/STOP
SESSION=exp3

# Make sure user PATH has uv & tmux even under cron's minimal env.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$REPO/logs"
cd "$REPO"

echo "[exp3-reboot] $(date -Iseconds) waking up"

if [ -e "$STOP_FILE" ]; then
    echo "[exp3-reboot] STOP file present at $STOP_FILE; refusing to relaunch."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[exp3-reboot] tmux session '$SESSION' already running; nothing to do."
    exit 0
fi

# Wait briefly for the GPU/driver to settle after reboot.
for _ in $(seq 1 12); do
    if nvidia-smi -L >/dev/null 2>&1; then break; fi
    sleep 5
done

tmux new -d -s "$SESSION" "bash $REPO/scripts/exp3_pipeline.sh 2>&1 | tee -a $REPO/logs/exp3.log"
echo "[exp3-reboot] launched tmux session '$SESSION'."
