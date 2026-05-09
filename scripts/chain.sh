#!/usr/bin/env bash
# Wait for one experiment's tmux session to end, then launch the next.
#
# Usage:
#   bash scripts/chain.sh <wait_session> <next_experiment> [<entry>]
#
# Example (Phase A on Machine A):
#   bash scripts/chain.sh mae_pretrain exp1_mae_meanpool
#
# Run as a daemon so the chain survives SSH drops and tmux-server death:
#   nohup setsid bash scripts/chain.sh mae_pretrain exp1_mae_meanpool \
#       >> logs/chain_mae_to_exp1.log 2>&1 < /dev/null & disown
#
# Behavior:
#   - Polls every 30 s for the wait_session's watchdog daemon to exit
#     (PID from logs/<wait>.pid; falls back to tmux for legacy sessions).
#   - On exit, checks logs/<wait_session>.log for completion marker
#     ("Done." from the python script OR train_robust's "training complete").
#   - If completion confirmed AND no STOP file exists, runs scripts/launch.sh <next>.
#   - Otherwise exits 1 with a message (you'll have to re-launch by hand).
#
# launch.sh refuses to launch if the watchdog PID file shows a live process,
# so even if this script fires twice it can't double-spawn.

set -u
WAIT="${1:-}"
NEXT="${2:-}"
ENTRY="${3:-}"
if [ -z "$WAIT" ] || [ -z "$NEXT" ]; then
    echo "usage: $0 <wait_session> <next_experiment> [<entry>]" >&2
    exit 64
fi

cd "$(dirname "$0")/.."
LOG="logs/${WAIT}.log"
PID_FILE="logs/${WAIT}.pid"
STOP_FILE="${STOP_FILE:-/Data/challenge_sb/STOP}"

is_running() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        return 0
    fi
    if tmux has-session -t "$WAIT" 2>/dev/null; then
        return 0
    fi
    return 1
}

echo "[chain] $(date -Iseconds) waiting for '$WAIT' watchdog to exit..."
while is_running; do
    sleep 30
done
echo "[chain] $(date -Iseconds) '$WAIT' watchdog has exited."

if [ -e "$STOP_FILE" ]; then
    echo "[chain] STOP file present at $STOP_FILE; not launching '$NEXT'." >&2
    exit 0
fi

if ! grep -qE "command exited 0; training complete|^Done\. " "$LOG" 2>/dev/null; then
    echo "[chain] '$WAIT' did not finish cleanly (no completion marker in $LOG)." >&2
    echo "[chain] Inspect the log and re-launch '$NEXT' manually if appropriate." >&2
    exit 1
fi

echo "[chain] $(date -Iseconds) launching '$NEXT'..."
exec bash scripts/launch.sh "$NEXT" ${ENTRY:+"$ENTRY"}
