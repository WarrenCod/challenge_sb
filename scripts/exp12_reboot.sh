#!/usr/bin/env bash
# Robust launcher for exp12_spacetime. Used both for the initial launch and as
# an @reboot helper. Wraps src/train.py in train_robust.sh and detaches via
# nohup + setsid (NOT tmux — tmux server can die independently of our process,
# which has bitten us before; nohup+setsid keeps the loop alive past SSH drops,
# terminal closes, and tmux-server crashes).
#
#   - SSH drop / terminal close cannot SIGHUP the run (nohup + new session via setsid).
#   - Python crashes / SIGKILL trigger a 30s back-off and resume from
#     checkpoints/exp12_spacetime_last.pt (training.resume=true is on by default).
#   - W&B id="exp12_spacetime" + resume="allow" appends to the existing run.
#
# Crontab entry (user crontab):
#   @reboot /Data/challenge_sb/scripts/exp12_reboot.sh >> /Data/challenge_sb/logs/exp12_reboot.log 2>&1
#
# Stop everything (kill switch shared with train_robust.sh):
#   touch /Data/challenge_sb/STOP

set -u
REPO=/Data/challenge_sb
STOP_FILE=$REPO/STOP
RUN=exp12_spacetime
LOG=$REPO/exp12.log
PIDFILE=$REPO/logs/$RUN.pid

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$REPO/logs"
cd "$REPO"

echo "[$RUN-reboot] $(date -Iseconds) waking up"

if [ -e "$STOP_FILE" ]; then
    echo "[$RUN-reboot] STOP file present at $STOP_FILE; refusing to relaunch."
    exit 0
fi

# Already running? (train_robust.sh wrapper still alive)
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[$RUN-reboot] watchdog already running (pid=$(cat "$PIDFILE")); nothing to do."
    exit 0
fi
# Belt-and-suspenders: pgrep in case the pidfile was lost across a reboot.
if pgrep -f "train_robust.sh .*experiment=$RUN" >/dev/null 2>&1; then
    echo "[$RUN-reboot] another train_robust.sh for $RUN is already running; nothing to do."
    exit 0
fi

# Wait for the GPU driver to come up after a cold boot.
for _ in $(seq 1 12); do
    if nvidia-smi -L >/dev/null 2>&1; then break; fi
    sleep 5
done

# Detached launch: nohup + setsid puts the watchdog in its own session, so it
# survives SSH drop / terminal close / parent-shell exit. Output is appended to
# $LOG (which both train_robust.sh and the inner python process inherit).
nohup setsid bash "$REPO/scripts/train_robust.sh" \
    uv run python "$REPO/src/train.py" experiment="$RUN" \
    >>"$LOG" 2>&1 < /dev/null &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" > "$PIDFILE"
echo "[$RUN-reboot] launched detached watchdog pid=$WATCHDOG_PID, log=$LOG"
