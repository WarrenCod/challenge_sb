#!/usr/bin/env bash
# Idempotent launcher for cron: relaunch <experiment> only if its watchdog
# is dead and no STOP file is present. Safe to call every minute.
#
# Usage (cron):
#   */2 * * * * /bin/bash /Data/challenge_sb/scripts/ensure_running.sh <exp> [<entry>]
#
# Behavior:
#   - STOP file present                  → exit 0 (do nothing)
#   - PID file present and process alive → exit 0 (do nothing)
#   - otherwise                          → call scripts/launch.sh (which
#                                          itself uses nohup setsid)
#
# Designed to coexist with the @reboot crontab entry: both call this same
# script, so there is exactly one source of truth for "is it supposed to
# be running?".

set -u
cd "$(dirname "$0")/.."

EXP="${1:-}"
ENTRY="${2:-}"
if [ -z "$EXP" ]; then
    echo "usage: $0 <experiment_name> [pretrain|ibot|train]" >&2
    exit 64
fi

STOP_FILE="${STOP_FILE:-/Data/challenge_sb/STOP}"
PID_FILE="logs/${EXP}.pid"

if [ -e "$STOP_FILE" ]; then
    # User asked us to stay down. Stay down.
    exit 0
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    # Watchdog already running. Nothing to do.
    exit 0
fi

echo "[ensure] $(date -Iseconds) watchdog for '$EXP' not running; relaunching."
exec /bin/bash scripts/launch.sh "$EXP" ${ENTRY:+"$ENTRY"}
