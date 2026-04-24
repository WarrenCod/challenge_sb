#!/usr/bin/env bash
# Training progress for a train.py log:
#   - table of completed-epoch summaries (epoch, losses, accuracies)
#   - current tqdm batch line
#   - elapsed, min/epoch (fractional — uses mid-epoch tqdm fraction), ETA
#
# Usage:
#   bash scripts/progress.sh                           # newest log, target 20, one-shot
#   bash scripts/progress.sh <log> <target_epochs>
#   bash scripts/progress.sh ... --watch               # redraw every 10 s, Ctrl-C to exit
#   bash scripts/progress.sh ... --watch --interval 5  # custom refresh interval (s)

set -u

LOG=""; TARGET=""; WATCH=0; INTERVAL=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w) WATCH=1 ;;
        --interval) INTERVAL="$2"; shift ;;
        *) if [[ -z "$LOG" ]]; then LOG="$1"; elif [[ -z "$TARGET" ]]; then TARGET="$1"; fi ;;
    esac
    shift
done
LOG="${LOG:-$(ls -t /Data/challenge_sb/logs/*.log 2>/dev/null | head -1)}"
TARGET="${TARGET:-20}"

if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "No log file found (got: '${LOG}')."
    exit 1
fi

render() {
    local birth now elapsed n cur cur_frac frac per rem
    birth=$(stat -c '%W' "$LOG")
    if [[ "$birth" == "-" || "$birth" == "0" ]]; then
        birth=$(stat -c '%Y' "$LOG")
    fi
    now=$(date +%s)
    elapsed=$((now - birth))

    (( WATCH )) && printf '\033[2J\033[H'
    printf "log: %s\ntarget: %d epochs   elapsed: %d min\n\n" "$LOG" "$TARGET" "$((elapsed / 60))"

    printf "%-7s %-11s %-10s %-11s %-10s\n" "epoch" "train_loss" "train_acc" "val_loss" "val_acc"
    tr '\r' '\n' < "$LOG" | awk '
        match($0, /^Epoch ([0-9]+)\/[0-9]+ \| train loss ([0-9.]+) acc ([0-9.]+) \| val loss ([0-9.]+) acc ([0-9.]+)/, m) {
            printf "%-7s %-11s %-10s %-11s %-10s\n", m[1], m[2], m[3], m[4], m[5]
        }'
    echo

    n=$(tr '\r' '\n' < "$LOG" | grep -cE '^Epoch [0-9]+/')
    cur=$(tail -c 800 "$LOG" | tr '\r' '\n' | grep -E '^\[[0-9]+/[0-9]+\]' | tail -1)
    if [[ -n "$cur" ]]; then
        echo "current: $cur"
    fi

    cur_frac=0
    if [[ -n "$cur" ]]; then
        cur_frac=$(awk 'match($0, /([0-9]+)\/([0-9]+) /, m) { if (m[2]+0 > 0) print m[1]/m[2]; exit }' <<< "$cur")
        [[ -z "$cur_frac" ]] && cur_frac=0
    fi
    frac=$(awk -v n="$n" -v f="$cur_frac" -v t="$TARGET" 'BEGIN { v = n + f; if (v > t) v = t; print v }')

    if awk -v f="$frac" 'BEGIN { exit !(f > 0) }'; then
        per=$(awk -v e="$elapsed" -v f="$frac" 'BEGIN { print e / f }')
        rem=$(awk -v p="$per" -v t="$TARGET" -v f="$frac" 'BEGIN { v = p * (t - f); if (v < 0) v = 0; print v }')
        printf "\n%.2f/%d epochs done | %.1f min/epoch | ETA %.0f min (%.1f h)\n" \
            "$frac" "$TARGET" \
            "$(awk -v p="$per" 'BEGIN { print p / 60 }')" \
            "$(awk -v r="$rem" 'BEGIN { print r / 60 }')" \
            "$(awk -v r="$rem" 'BEGIN { print r / 3600 }')"
    else
        printf "\n0/%d epochs done | (ETA unavailable until first batch landed)\n" "$TARGET"
    fi
}

if (( WATCH )); then
    trap 'echo; exit 0' INT
    while true; do
        render
        sleep "$INTERVAL"
    done
else
    render
fi
