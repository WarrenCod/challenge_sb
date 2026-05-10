#!/usr/bin/env bash
# Wait for the exp2m training process (PID 306112) to exit, then run
# create_submission.py against checkpoints/exp2m_st_perceiver.pt and write a
# brief sanity report (row count + class histogram) to the same log.

set -u
set -o pipefail

cd /Data/challenge_sb || exit 1

TRAIN_PID=306112
CKPT="checkpoints/exp2m_st_perceiver.pt"
OUT_CSV="processed_data/submission_exp2m_st_perceiver.csv"
LOG="logs/submit_exp2m.log"
NOTE="notes/exp2m_st_perceiver.md"

echo "[$(date -Is)] watcher up; waiting for PID ${TRAIN_PID} to exit" > "${LOG}"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 60
done

echo "[$(date -Is)] training PID exited; sleeping 30s before submission" >> "${LOG}"
sleep 30

echo "[$(date -Is)] launching create_submission.py" >> "${LOG}"
.venv/bin/python src/create_submission.py \
  training.checkpoint_path="${CKPT}" \
  dataset.submission_output="${OUT_CSV}" \
  >> "${LOG}" 2>&1
RC=$?

echo "[$(date -Is)] create_submission.py exit code: ${RC}" >> "${LOG}"

if [[ ${RC} -eq 0 && -f "${OUT_CSV}" ]]; then
  echo "[$(date -Is)] sanity check on ${OUT_CSV}" >> "${LOG}"
  ROWS=$(wc -l < "${OUT_CSV}")
  echo "  total lines (incl header): ${ROWS}" >> "${LOG}"
  echo "  class distribution (top 10 + bottom 5):" >> "${LOG}"
  tail -n +2 "${OUT_CSV}" | awk -F',' '{print $2}' | sort | uniq -c | sort -rn | head -10 >> "${LOG}"
  echo "  ..." >> "${LOG}"
  tail -n +2 "${OUT_CSV}" | awk -F',' '{print $2}' | sort | uniq -c | sort -rn | tail -5 >> "${LOG}"
  N_CLASSES=$(tail -n +2 "${OUT_CSV}" | awk -F',' '{print $2}' | sort -u | wc -l)
  echo "  unique classes predicted: ${N_CLASSES}" >> "${LOG}"

  {
    echo ""
    echo "## Submission ($(date -Is))"
    echo ""
    echo "- File: \`${OUT_CSV}\`"
    echo "- Rows (excl header): $((ROWS - 1))"
    echo "- Unique classes predicted: ${N_CLASSES}"
    echo "- Created from \`${CKPT}\` after PID ${TRAIN_PID} exit."
  } >> "${NOTE}"

  echo "[$(date -Is)] DONE — submission ready at ${OUT_CSV}" >> "${LOG}"
else
  echo "[$(date -Is)] ERROR — submission did not produce a CSV; check log above" >> "${LOG}"
fi
