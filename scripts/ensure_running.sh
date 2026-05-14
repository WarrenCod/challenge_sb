#!/usr/bin/env bash
# ensure_running.sh <exp_id>
#
# Idempotent watchdog for a long training run. Designed to be invoked by:
#   */2 * * * * cd /Data/challenge_sb && bash scripts/ensure_running.sh <exp_id> >> logs/ensure_running.log 2>&1
#   @reboot   cd /Data/challenge_sb && sleep 30 && bash scripts/ensure_running.sh <exp_id> >> logs/ensure_running.log 2>&1
#
# Behavior on each tick:
#   - STOP file present                 → exit (no-op)
#   - <exp>.done marker present         → exit (no-op; training finished)
#   - python src/train.py experiment=<exp> already running → exit (no-op)
#   - _last.pt shows epoch >= training.epochs → touch .done, exit
#   - otherwise: launch under setsid + train_robust.sh (resume from _last.pt)
#
# Uses the venv python explicitly (cron's bare `python` resolves to /usr/bin/python).

set -u

EXP="${1:?usage: $0 <exp_id>}"
REPO="/Data/challenge_sb"
VENV_PY="$REPO/.venv/bin/python"
STOP_FILE="${STOP_FILE:-$REPO/STOP}"
DONE_FILE="$REPO/checkpoints/${EXP}.done"
LAST_CKPT="$REPO/checkpoints/${EXP}_last.pt"
LOG="$REPO/logs/${EXP}.log"

log() { echo "[$(date -Iseconds)] ensure_running[$EXP]: $*"; }

if [ -e "$STOP_FILE" ]; then
    exit 0
fi

if [ -e "$DONE_FILE" ]; then
    exit 0
fi

# Already running? Match the exact `experiment=<exp>` token on a python train.py command line.
if pgrep -af "[s]rc/train\.py.* experiment=${EXP}( |$)" > /dev/null; then
    exit 0
fi

# Completed already? (epoch >= total in _last.pt). Cheap-ish; only hit when no proc is running.
if [ -e "$LAST_CKPT" ]; then
    completed=$("$VENV_PY" - "$LAST_CKPT" <<'PY' 2>/dev/null || true
import sys, torch
try:
    s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    ep = int(s.get("epoch", 0))
    total = int(s.get("config", {}).get("training", {}).get("epochs", 0))
    print(1 if total > 0 and ep >= total else 0)
except Exception:
    print(0)
PY
)
    if [ "$completed" = "1" ]; then
        log "training complete (epoch >= total); writing $DONE_FILE"
        touch "$DONE_FILE"
        exit 0
    fi
fi

cd "$REPO" || exit 1
log "launching: $VENV_PY src/train.py experiment=$EXP"
setsid nohup bash "$REPO/scripts/train_robust.sh" \
    "$VENV_PY" src/train.py experiment="$EXP" \
    >> "$LOG" 2>&1 < /dev/null &
disown
