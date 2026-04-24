# CLAUDE.md

## Project

CSC_43M04_EP (École Polytechnique Modal d'informatique, Deep Learning in Computer Vision) — the **"What Happens Next?"** video classification challenge.

- Task: classify each video into one of **33 action classes** (Something-Something v2-style data).
- Input: a folder of extracted JPG frames per video. Model sees a fixed number of frames per clip (default `num_frames=8`).
- Two tracks, both in scope:
  - **Track 1 — Closed World:** train from scratch, no ImageNet weights. Entry point: `experiment=baseline_from_scratch`.
  - **Track 2 — Open World:** ImageNet-pretrained backbones allowed. Entry point: `experiment=baseline_pretrained`.

## Stack

- Python 3.10+, **`uv`** for env (`uv sync`).
- PyTorch (CUDA 12.8 wheels via `pyproject.toml`), torchvision, Hydra, OmegaConf, Pillow.
- Hardware: **local CUDA GPU**. Always default `training.device=cuda`.

## Repo layout

- [src/train.py](src/train.py) — training loop; saves best-by-val-acc checkpoint with full merged Hydra config inside the `.pt`.
- [src/evaluate.py](src/evaluate.py) — rebuilds model from checkpoint config; reports top-1 / top-5 on the full `dataset.val_dir`.
- [src/create_submission.py](src/create_submission.py) — inference on test frames; writes `video_name,predicted_class` CSV.
- [src/dataset/video_dataset.py](src/dataset/video_dataset.py) — `VideoFrameDataset` + `collect_video_samples`; returns `(T, C, H, W)` tensors; class index parsed from `NNN_Name` folder prefix.
- [src/models/](src/models/) — each model maps `(B, T, C, H, W) → (B, num_classes)`. For the current set of registered models, read the `build_model` branches in [src/train.py](src/train.py) and list [src/configs/experiment/](src/configs/experiment/) — this doc does not maintain per-model inventories.
- [src/utils.py](src/utils.py) — seeds, transforms (ImageNet norm when pretrained), train/val split helper.
- [src/configs/](src/configs/) — Hydra: `config.yaml` composes `model/`, `data/`, `train/`, `experiment/`.
- [src/misc/](src/misc/) — one-off data prep scripts (`download_data.py`, `preprocess_ssv2.py`); not part of training.

## Commands

Run from the repo root (Hydra resolves `configs/` via `config_path="configs"` in the scripts, so `cd src` or `python src/train.py` both work):

```bash
# Data paths now live under processed_data/ at the repo root, so the Hydra default
# in src/configs/data/default.yaml resolves correctly when you run from /Data/challenge_sb.
# No DATA override needed unless you point at a different dataset.

# Smoke test FIRST — always — before any long run (short, stays in foreground)
python src/train.py experiment=baseline_from_scratch dataset.max_samples=64 training.epochs=1 training.batch_size=4

# Full training — ALWAYS run inside tmux so an SSH drop can't kill the job.
#   Detach from a live session:  Ctrl-b  then  d
#   Reattach later:               tmux attach -t train
#   List sessions:                tmux ls
#   Kill a stuck session:         tmux kill-session -t train
# `tee` mirrors tqdm/prints to train.log so you can inspect progress even without reattaching.
tmux new -s train "python src/train.py experiment=baseline_from_scratch 2>&1 | tee train.log"
tmux new -s train "python src/train.py experiment=baseline_pretrained   2>&1 | tee train.log"
tmux new -s train "python src/train.py experiment=cnn_lstm              training.epochs=10 training.batch_size=16 2>&1 | tee train.log"

# If tmux isn't available, the nohup fallback is:
#   nohup python src/train.py experiment=... > train.log 2>&1 &  disown
#   tail -f train.log   # to watch; Ctrl-c only stops tail, not the training

# Evaluate (short, keep in foreground)
python src/evaluate.py training.checkpoint_path=best_model.pt

# Submission (short, keep in foreground)
python src/create_submission.py training.checkpoint_path=best_model.pt
```

Best checkpoint path is `training.checkpoint_path` (default `best_model.pt` in cwd). The checkpoint embeds the merged Hydra config, so `evaluate.py` / `create_submission.py` reload the correct architecture automatically — **no duplicate model registry to update**.

## Adding a new model (the one pattern to follow)

1. `src/models/<name>.py` implementing `nn.Module` with `(B,T,C,H,W) → (B,num_classes)`.
2. Register one branch in [`build_model` in src/train.py](src/train.py) keyed on `cfg.model.name`.
3. `src/configs/model/<name>.yaml` with `# @package _global_` and `model.name: <name>`.
4. `src/configs/experiment/<name>.yaml` with `defaults: [- override /model: <name>]`.

`evaluate.py` and `create_submission.py` need no edits.

## Data layout

The dataset lives **at the repo root**, under `processed_data/`, with label CSVs in a sibling `labels/` folder:

```
/Data/challenge_sb/
├── src/
├── processed_data/
│   ├── train/   ← class subfolders (000_Closing_something, …)
│   ├── val/     ← class subfolders (same layout)
│   └── test/    ← video_<id> subfolders, no class grouping
├── labels/
│   ├── train_video_labels.csv   ← video_name,class_idx  (authoritative train labels)
│   └── val_video_labels.csv     ← video_name,class_idx  (authoritative val labels)
└── …
```

[src/configs/data/default.yaml](src/configs/data/default.yaml) uses `${hydra:runtime.cwd}/processed_data/{train,val,test}`. Run Hydra commands from `/Data/challenge_sb` and paths resolve correctly — no CLI overrides needed. Override only if you point at a different dataset.

**Label sources.** Two equivalent sources exist for train/val:
- Folder prefix: `processed_data/{train,val}/NNN_ClassName/video_<id>/` → class `NNN`. The current `src/dataset/video_dataset.py` parses labels this way.
- CSVs in `labels/`: `video_name,class_idx` — one row per video. Authoritative mapping; useful if the folder layout ever drifts, or if you want a different split keyed on video ID.

Class index space is `0..32` (33 classes), with **27 absent** from the on-disk folders and CSVs — no training example carries label 27. Keep `num_classes=33` in the model anyway (that output head is never exercised).

## Working preferences

- **Propose changes before editing.** Explain the approach + tradeoffs first; wait for a "go" before modifying files. This is the default mode for this repo.
- **Always smoke-test before long training.** Use `dataset.max_samples=64 training.epochs=1 training.batch_size=4` (or similar) to verify the pipeline end-to-end before committing GPU hours.
- **Every training run must survive SSH disconnection.** No exceptions — a full `python src/train.py …` call is never launched in the foreground. Wrap it in `tmux new -s <name> "<cmd> 2>&1 | tee <name>.log"` (preferred) or `nohup <cmd> > <name>.log 2>&1 & disown` (fallback). Smoke tests can stay in the foreground; anything over ~1 min cannot.
- **Current goal:** design a new architecture, or merge ideas from recent SOTA, to maximize accuracy on the challenge. Both tracks matter. Open to recent ideas — temporal transformers, 3D / (2+1)D CNNs, pretrained video backbones (VideoMAE, TimeSformer, X3D, V-JEPA, etc.), two-stream, masked modeling, distillation. Surface options with tradeoffs rather than picking unilaterally.
- User background: comfortable with PyTorch; frame explanations accordingly (no need to re-explain basics, but video-specific terminology is worth grounding briefly).
