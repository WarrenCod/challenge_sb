# CLAUDE.md

## Project

CSC_43M04_EP (École Polytechnique Modal d'informatique, Deep Learning in Computer Vision) — the **"What Happens Next?"** video classification challenge.

- Task: classify each video into one of **33 action classes** (Something-Something v2-style data).
- Input: a folder of extracted JPG frames per video. Model sees a fixed number of frames per clip — **structurally capped at 4** for this dataset (each video has only `frame_000…003.jpg`).
- **Working Track 1 — Closed World only.** No ImageNet weights, no externally-pretrained backbones. SSL pretraining on this challenge's own pixels (`processed_data/{train,val,test}`) is allowed because the supervision is internal. Entry point: `experiment=baseline_from_scratch`. Track 2 is out of scope; ignore `baseline_pretrained` unless explicitly told otherwise.

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
- **Current goal:** maximise Track 1 accuracy. Open to recent ideas constrained to *no external supervision* — TSM, R(2+1)D / 3D CNNs trained from scratch, temporal transformers, MAE / iBOT / V-JEPA SSL on this dataset's own pixels, two-stream, masked modeling, distillation between in-house models. Surface options with tradeoffs rather than picking unilaterally.
- **Direction-sensitive labels.** Never apply horizontal or vertical flip in any pipeline (training, SSL, TTA). SSv2 labels distinguish e.g. `Pulling left→right` vs `right→left`; flipping silently mislabels data.
- **No flip in SSL multi-crop either.** iBOT / DINO / SimCLR default recipes include `RandomHorizontalFlip` in their global and local crop pipelines. **Strip it from every view.** Direction-safe SSL augs only: `RandomResizedCrop`, `ColorJitter`, `GaussianBlur`, `Solarize`, `RandomGrayscale`, `RandomErasing`. Easy to forget when copying upstream code.
- User background: comfortable with PyTorch; frame explanations accordingly (no need to re-explain basics, but video-specific terminology is worth grounding briefly).

## Per-run protocol

Every training launch must follow this protocol. These rules are non-negotiable.

1. **Pre-run intent note (logged to W&B).** Before kicking off training, draft a short, concise note describing what's being tested and a tiny architecture sketch (ASCII shape diagram, one-line block sequence). Pass it to `wandb.init(notes=..., tags=[...])` so it lands on the run page. Style example:

   ```text
   exp7_tsm_r18: TSM(λ=0.25) on R18 from scratch, 4 frames, no flip.
   (B,4,3,224,224) → R18+TSM blocks → GAP_t → FC(33)
   ```

   Keep it under ~5 lines. The point is to make the W&B run self-explanatory months later.

2. **Robustness — every long run must survive reboot / `tmux kill` / SSH drop.**
   - Wrap the launch in `tmux new -s <run_name> "<cmd> 2>&1 | tee <run_name>.log"`.
   - Pass a stable `wandb.run_id` (e.g. `wandb.run_id=<run_name>_v1`) so a relaunch resumes the same W&B record instead of creating a duplicate.
   - Save **two** checkpoints periodically (every epoch or every N steps): `checkpoints/<run>/last.pt` (overwritten) and `checkpoints/<run>/best.pt` (best val). Training entry point must accept `training.resume_path=...` and pick up optimizer + scheduler + epoch + RNG state, not just weights.
   - On unexpected exit, the standard recovery is: `tmux new -s <run_name> "<cmd> training.resume_path=checkpoints/<run>/last.pt wandb.run_id=<same_id> 2>&1 | tee -a <run_name>.log"`.
   - If the current `train.py` doesn't yet support resume + W&B run_id, that gap is a blocker for any new long run — flag it and fix before launching.

3. **Monitor regularly.** During long runs, periodically (every 10–30 min for active sessions) check `tmux ls`, `nvidia-smi`, the live tail of `<run>.log`, and the W&B loss/val-acc curves. Don't fire-and-forget. If loss is flat/diverging or GPU util has dropped to 0, intervene rather than waiting for the run to finish.

4. **End-of-run interpretation.** When a run finishes (planned epochs, early stop, or aborted), write 2–3 concise lines in chat: did it learn, where it plateaued, what the next lever is. No standalone report files; this lives in the conversation and on the W&B run page (paste the same note as a final W&B comment if non-trivial).

5. **Always finish with a submission.** After a run yields a usable checkpoint, run `python src/create_submission.py training.checkpoint_path=checkpoints/<run>/best.pt dataset.submission_output=submissions/<run>.csv` and report the CSV path in chat. One submission per meaningful run, named after the run.
