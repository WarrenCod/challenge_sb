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

git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)" || true
mkdir -p logs

if tmux has-session -t "$EXP" 2>/dev/null; then
    echo "[launch] tmux session '$EXP' already exists; attach with: tmux attach -t $EXP" >&2
    exit 1
fi

tmux new -d -s "$EXP" "bash scripts/train_robust.sh python $SCRIPT experiment=$EXP 2>&1 | tee -a logs/${EXP}.log"
echo "[launch] launched $SCRIPT (experiment=$EXP) in tmux session '$EXP'"
echo "[launch] watch:    tmux attach -t $EXP   (Ctrl-b d to detach)"
echo "[launch] logs:     tail -f logs/${EXP}.log"
echo "[launch] stop:     touch /Data/challenge_sb/STOP && tmux kill-session -t $EXP"
