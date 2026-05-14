#!/usr/bin/env bash
# Bootstrap a fresh Polytechnique A5000 box for exp3p (or any nandou run).
# Idempotent — re-run safely; each step is gated on its output existing.
#
# Prerequisites:
#   - Run from the repo root (/Data/challenge_sb).
#   - `uv` available on PATH (https://docs.astral.sh/uv/).
#   - Passwordless SSH to albatros.polytechnique.fr for the Stage-1 ckpt scp.
#     If you are not stanislas.mischler, edit STAGE1_REMOTE below or arrange
#     for the ckpt some other way (the file is too large to keep in git).
#
# What it does (in order):
#   1. uv sync                             — install pinned torch/etc into .venv/
#   2. scp checkpoints/videomae_stage1.pt  — from albatros (~88 MB)
#   3. gdown frames.zip                    — Google Drive ID below (~2.7 GiB)
#   4. unzip into processed_data/          — top dirs in zip are train/val/test
#   5. rm frames.zip                       — keep disk usage down
#
# After this finishes, smoke-test:
#   rm -fv checkpoints/videomae_exp3p*.pt
#   .venv/bin/python src/train.py experiment=videomae_exp3p \
#       dataset.max_samples=64 training.epochs=1 training.batch_size=4 \
#       training.freeze_backbone_epochs=0 training.warmup_epochs=0 \
#       training.swa.start_epoch=1 wandb.enabled=false
#
# Then launch:
#   rm -fv checkpoints/videomae_exp3p*.pt
#   tmux new -s exp3p "bash scripts/train_robust.sh bash scripts/run_exp3p.sh 2>&1 | tee -a logs/exp3p.log"

set -e
set -u
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/Data/challenge_sb}"
STAGE1_REMOTE="${STAGE1_REMOTE:-stanislas.mischler@albatros.polytechnique.fr:/Data/challenge_sb/checkpoints/videomae_stage1.pt}"
FRAMES_GDRIVE_ID="${FRAMES_GDRIVE_ID:-1SlRJBD6cyXMr5772kOKe5xXAU9Scu5vR}"

cd "$REPO_ROOT"
mkdir -p checkpoints submissions logs

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# 1. uv sync — pinned torch+cu128 wheels from pyproject.toml
if [ ! -x .venv/bin/python ]; then
  echo "[setup] $(stamp) uv sync (first run)…"
  uv sync
else
  echo "[setup] $(stamp) .venv/ already present — running uv sync to refresh lock state"
  uv sync
fi

.venv/bin/python -c "import torch; print(f'[setup] torch {torch.__version__} cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"}')"

# 2. Stage-1 ckpt
if [ -s checkpoints/videomae_stage1.pt ]; then
  echo "[setup] $(stamp) Stage-1 ckpt already present at checkpoints/videomae_stage1.pt — skipping scp"
else
  echo "[setup] $(stamp) scp Stage-1 ckpt from $STAGE1_REMOTE …"
  scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$STAGE1_REMOTE" checkpoints/
  ls -la checkpoints/videomae_stage1.pt
fi

# 3. Frames dataset
if [ -d processed_data/train ] && [ -d processed_data/val ] && [ -d processed_data/test ]; then
  echo "[setup] $(stamp) processed_data/{train,val,test} all present — skipping data download"
else
  if [ ! -s frames.zip ]; then
    echo "[setup] $(stamp) gdown frames.zip from Drive ID $FRAMES_GDRIVE_ID …"
    # NB: the gdown version uv pulls (0.x) does not accept --fuzzy. Pass the bare ID.
    uv run --with gdown gdown "$FRAMES_GDRIVE_ID" -O frames.zip
  fi
  echo "[setup] $(stamp) unzipping frames.zip → processed_data/ …"
  mkdir -p processed_data
  unzip -q frames.zip -d processed_data
  rm -f frames.zip
fi

echo "[setup] $(stamp) done. Counts:"
echo "  train videos: $(find processed_data/train -mindepth 2 -maxdepth 2 -type d | wc -l)"
echo "  val videos:   $(find processed_data/val   -mindepth 2 -maxdepth 2 -type d | wc -l)"
echo "  test videos:  $(find processed_data/test  -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "  stage1 ckpt:  $(stat -c '%s bytes' checkpoints/videomae_stage1.pt 2>/dev/null || echo missing)"
