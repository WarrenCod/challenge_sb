#!/usr/bin/env bash
# Idempotent guard for long training runs (V-JEPA pretrain OR supervised Stage 2).
#
# Usage:   bash scripts/ensure_running.sh <experiment_name> [entrypoint]
# Examples:
#   bash scripts/ensure_running.sh vjepa_pretrain_v6                          # default: src/pretrain_vjepa.py
#   bash scripts/ensure_running.sh exp10_vjepa_v7_spacetime src/train.py      # Stage 2 supervised
#
# Behaviour:
#   - If /Data/challenge_sb/STOP exists, do nothing (and never relaunch).
#   - If a python process for this experiment+entrypoint is already alive,
#     do nothing (don't double-start).
#   - Otherwise: launch under `nohup setsid bash scripts/train_robust.sh ...`
#     so it survives SSH drops, tmux death, and parent-process kills.
#
# Pair with crontab:
#   */2 * * * *  /Data/challenge_sb/scripts/ensure_running.sh <exp> [entrypoint] >> /Data/challenge_sb/logs/cron_watchdog.log 2>&1
#   @reboot sleep 60 && /Data/challenge_sb/scripts/ensure_running.sh <exp> [entrypoint] >> /Data/challenge_sb/logs/cron_reboot.log 2>&1
#
# Halt path: `touch /Data/challenge_sb/STOP` (blocks every layer).

set -u

EXP="${1:-}"
ENTRY="${2:-src/pretrain_vjepa.py}"
if [ -z "$EXP" ]; then
    echo "[ensure] usage: $0 <experiment_name> [entrypoint]" >&2
    exit 64
fi

REPO_ROOT="/Data/challenge_sb"
STOP_FILE="$REPO_ROOT/STOP"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/${EXP}.log"
TS="$(date -Iseconds)"

# Pattern fragment to identify a live run: basename of the entry script (without
# .py) combined with the experiment= arg. Anchors on both so two different exps
# under the same entrypoint don't false-positive each other.
ENTRY_BASE="$(basename "$ENTRY" .py)"

mkdir -p "$LOG_DIR"

if [ -e "$STOP_FILE" ]; then
    echo "[ensure] $TS STOP file present; not launching $EXP."
    exit 0
fi

if pgrep -af "${ENTRY_BASE}.*experiment=${EXP}" >/dev/null; then
    echo "[ensure] $TS already running: $ENTRY_BASE experiment=$EXP"
    exit 0
fi

echo "[ensure] $TS launching $ENTRY_BASE experiment=$EXP -> $LOG_FILE"
cd "$REPO_ROOT" || exit 1
nohup setsid bash "$REPO_ROOT/scripts/train_robust.sh" \
    python "$REPO_ROOT/$ENTRY" "experiment=$EXP" \
    >> "$LOG_FILE" 2>&1 < /dev/null &
disown || true
echo "[ensure] $TS launched (pid=$!)"
