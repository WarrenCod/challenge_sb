#!/usr/bin/env bash
# One-command launch for an experiment on a school machine.
#
# Usage (after SSH'ing into the box):
#   bash scripts/launch.sh <experiment_name> [<entry>]
#
# Examples:
#   bash scripts/launch.sh mae_pretrain         pretrain     # MAE Stage 1
#   bash scripts/launch.sh exp1_mae_meanpool                 # Stage 2 default = train
#   bash scripts/launch.sh exp2_mae_transformer
#   bash scripts/launch.sh exp3_ibot_transformer ibot        # iBOT Stage 1
#   bash scripts/launch.sh exp4_tsm_resnet18                 # train.py
#
# Behavior:
#   - git pull --rebase to sync latest configs/code.
#   - Picks the entry point based on the second arg (or autodetects from name).
#   - Wraps the run in scripts/train_robust.sh (auto-resume on crash, STOP file honored).
#   - Launches inside a detached tmux session named after the experiment.
#   - Pipes everything to logs/<experiment>.log via tee.

set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${1:-}"
ENTRY="${2:-}"
if [ -z "$EXP" ]; then
    echo "usage: $0 <experiment_name> [pretrain|ibot|train]" >&2
    exit 64
fi

if [ -z "$ENTRY" ]; then
    case "$EXP" in
        mae_pretrain*)   ENTRY="pretrain" ;;
        ibot_pretrain*)  ENTRY="ibot" ;;
        *)               ENTRY="train" ;;
    esac
fi

case "$ENTRY" in
    pretrain) SCRIPT="src/pretrain_mae.py" ;;
    ibot)     SCRIPT="src/pretrain_ibot.py" ;;
    train)    SCRIPT="src/train.py" ;;
    *) echo "unknown entry: $ENTRY (use pretrain|ibot|train)" >&2; exit 64 ;;
esac

# Resolve python explicitly to the project venv. Cron's PATH does NOT include
# the uv-managed venv, so without this, cron-spawned restarts fall back to
# system python and crash-loop with "No module named 'torch'" (happened
# 2026-05-10 04:46→10:25 — burned 5h30 of GPU time before being noticed).
PYTHON="${PYTHON:-/Data/challenge_sb/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "[launch] FATAL: venv python not found at $PYTHON; run 'uv sync' first." >&2
    exit 70
fi

git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)" || true
mkdir -p logs

PID_FILE="logs/${EXP}.pid"
LOG="logs/${EXP}.log"

# If a previous daemon is still running, refuse to start a second one.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[launch] '$EXP' watchdog already running (pid $(cat "$PID_FILE"))." >&2
    echo "[launch] follow:   tail -f $LOG" >&2
    echo "[launch] stop:     touch /Data/challenge_sb/STOP && kill \$(cat $PID_FILE) (or kill the python child)" >&2
    exit 1
fi

# Daemonize via `nohup setsid` so the watchdog survives SSH drop, parent-shell
# exit, AND tmux-server death (which killed v4 on 2026-05-08). The process
# becomes a session leader with no controlling terminal.
nohup setsid bash -c "exec bash scripts/train_robust.sh $PYTHON $SCRIPT experiment=$EXP" \
    >> "$LOG" 2>&1 < /dev/null &
DAEMON_PID=$!
disown "$DAEMON_PID" 2>/dev/null || true
echo "$DAEMON_PID" > "$PID_FILE"

echo "[launch] launched $SCRIPT (experiment=$EXP) as daemon pid $DAEMON_PID"
echo "[launch] follow:   tail -f $LOG"
echo "[launch] check:    ps -fp \$(cat $PID_FILE)   # watchdog"
echo "[launch] stop:     touch /Data/challenge_sb/STOP   # blocks the NEXT retry only; current python run keeps going"
echo "[launch] kill:     touch /Data/challenge_sb/STOP && pkill -f 'src/${SCRIPT##*/}.*experiment=$EXP'   # halt now"
