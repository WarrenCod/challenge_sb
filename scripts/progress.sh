#!/usr/bin/env bash
# Print a compact status for a running train.py log:
#   - all completed-epoch summary lines
#   - current tqdm line (batch-level progress)
#   - elapsed time, avg min/epoch, ETA
#
# Usage:
#   bash scripts/progress.sh <log_file> [target_epochs]
#
# Defaults: log = newest file under logs/, target = 20.

set -u

LOG="${1:-$(ls -t /Data/challenge_sb/logs/*.log 2>/dev/null | head -1)}"
TARGET="${2:-20}"

if [[ -z "${LOG}" || ! -f "${LOG}" ]]; then
    echo "No log file found (got: '${LOG}')."
    exit 1
fi

BIRTH=$(stat -c '%W' "${LOG}")
NOW=$(date +%s)
ELAPSED=$((NOW - BIRTH))

echo "log: ${LOG}"
echo

echo "Completed epochs:"
EPOCH_LINES=$(tr '\r' '\n' < "${LOG}" | grep -E '^Epoch [0-9]+/')
if [[ -n "${EPOCH_LINES}" ]]; then
    echo "${EPOCH_LINES}"
else
    echo "  (none yet)"
fi
echo

# Current batch-level progress (last tqdm redraw)
CUR=$(tail -c 400 "${LOG}" | tr '\r' '\n' | grep -E '^\[[0-9]+/[0-9]+\]' | tail -1)
if [[ -n "${CUR}" ]]; then
    echo "Current: ${CUR}"
    echo
fi

N=$(tr '\r' '\n' < "${LOG}" | grep -cE '^Epoch [0-9]+/')
ELAPSED_MIN=$((ELAPSED / 60))

if [[ "${N}" -gt 0 ]]; then
    PER=$((ELAPSED / N))
    REM=$((PER * (TARGET - N)))
    printf "%d/%d epochs done | %d min elapsed | %d min/epoch avg | ETA %d min (%.1f h remaining)\n" \
        "${N}" "${TARGET}" "${ELAPSED_MIN}" "$((PER / 60))" "$((REM / 60))" \
        "$(echo "scale=1; ${REM}/3600" | bc)"
else
    printf "0/%d epochs done | %d min elapsed | (ETA unavailable until epoch 1 completes)\n" \
        "${TARGET}" "${ELAPSED_MIN}"
fi
