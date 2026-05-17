# CLAUDE.md

## Project

CSC_43M04_EP (École Polytechnique Modal d'informatique, Deep Learning in Computer Vision) — the **"What Happens Next?"** video classification challenge.

- Task: classify each video into one of **33 action classes** (Something-Something v2-style data).
- Input: a folder of extracted JPG frames per video. Model sees a fixed number of frames per clip.
- Active focus: **Track 1 — Closed World** (train from scratch, no ImageNet weights).
- Track 2 — Open World (ImageNet-pretrained backbones) is parked unless explicitly revisited.

## Active architecture: V-JEPA v8 (Track 1)

We pretrain a ViT-S/16 video encoder with V-JEPA self-supervision (Stage 1), then fine-tune it with a 4-layer joint space-time transformer head on the labeled 33-class task (Stage 2). v8 is the current active recipe; v7 is the strongest baseline so far (Stage 1 probe acc 0.21, Stage 2 exp10 val acc 0.36 at ep 80, still climbing).

**v8 = v7 + two surgical upgrades**, both well-motivated by V-JEPA paper findings:
1. **Multi-mask M=3** (V-JEPA 1's main sample-efficiency lever). Single encoder forward, three independent tube-block masks per clip, predictor runs three times. ~3× more SSL signal per encoder step, expected +2–4 probe-pp at fixed compute.
2. **Hybrid cross-then-self predictor** (V-JEPA 2 cross-attention + DETR-style refine). Phase A: mask queries cross-attend to visible context only (no self-talk between mask positions → kills V-JEPA 1's known shortcut where mask tokens collude through self-attention). Phase B: short joint self-attention over `[visible; queries]` for global consistency. Forces more semantic work into the encoder.

```
clip (B,4,3,224,224)
   │   Conv3d-tubelet (tubelet=2) → 392 patch tokens
   ▼
 ┌──────────────┐                ┌──────────────┐
 │ ctx_enc      │── z_ctx ──┐    │ tgt_enc (EMA)│── z_tgt
 └──────────────┘           │    └──────────────┘
                            │       ▲ 3 independent
                            │       │ tube-block masks (90%)
                            ▼       │
        for m in 1..M:    visible = z_ctx[~mask_m]
                          queries = mask_tok + pos_embed[mask_m]
          Phase A:        queries  ←x-attend─ visible   (3 cross-attn blocks)
          Phase B:        [visible; queries] self-attn   (2 self-attn blocks)
                          z_pred_m  →  L1 vs LN(z_tgt)[mask_m]
                                       + 0.5 · MSE vs LN_patch(rgb)[mask_m]
        loss = mean over M
```

Phase 2 (exp11) reuses the v7 spacetime head verbatim, loading the v8 encoder; same recipe as exp10 except epochs 80→120 and warmup 5→8 (exp10's val acc was still climbing at ep 80).

### What changed vs v7

| Aspect          | v7                                  | v8                                                   |
| --------------- | ----------------------------------- | ---------------------------------------------------- |
| Masks per clip  | 1                                   | **3** (independent tube-block)                       |
| Predictor       | 6-block self-attention              | **3 cross-attn + 2 self-attn** (hybrid)              |
| Loss            | L1 feat + 0.5·MSE pixel (per-patch LN) | unchanged                                          |
| Stage-1 epochs  | 200                                 | **150** (M=3 carries ~3× signal per step)            |
| Stage-2 epochs  | 80 (exp10)                          | **120** (exp11)                                      |
| Wall-clock      | ~3.6 min/ep                         | **~4.7 min/ep** (~1.3× v7; ~12 h target for Stage 1) |

### Quickstart for v8

Run all commands from the repo root. Stage 1 produces `checkpoints/vjepa_stage1_v8.pt`, Stage 2 produces `checkpoints/exp11_vjepa_v8_spacetime.pt`.

```bash
# --- 0. Smoke test (stays in foreground, ~1 min) -------------------------
uv run python src/pretrain_vjepa.py experiment=vjepa_pretrain_v8 \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4

# --- 1. Stage 1: V-JEPA v8 SSL pretrain (~12h on a single A100) -----------
tmux new -s vjepa_v8 "uv run python src/pretrain_vjepa.py experiment=vjepa_pretrain_v8 2>&1 | tee vjepa_v8.log"

# --- 2. Probe gate (verify Stage 1 is on track) ---------------------------
#   ep  30 ≥ 0.15 real val   |   ep 80 ≥ 0.25   |   ep 150 ≥ 0.30
#   If a gate misses badly, kill and revert to v7 baseline.
uv run python src/evaluate.py training.checkpoint_path=checkpoints/vjepa_stage1_v8.pt

# --- 3. Stage 2: spacetime head fine-tune (~6h) ---------------------------
tmux new -s exp11 "uv run python src/train.py experiment=exp11_vjepa_v8_spacetime 2>&1 | tee exp11.log"

# --- 4. Evaluate + submit -------------------------------------------------
uv run python src/evaluate.py training.checkpoint_path=checkpoints/exp11_vjepa_v8_spacetime.pt
uv run python src/create_submission.py training.checkpoint_path=checkpoints/exp11_vjepa_v8_spacetime.pt
```

**Target metrics:** Stage 1 probe ≥ 0.30 (v7 was 0.21). Stage 2 real val ≥ 0.40 (vs exp10's 0.36); stretch ≥ 0.45.

### Low-risk follow-on tweaks (in priority order)

Once v8 lands and is interpretable, the following are safe single-knob ablations:

1. **Bigger M (M=4 or M=6)** — paper shows monotone gain to ~M=8; we cap at M=3 for wall-clock but if Stage 1 finishes early, push it.
2. **Longer Stage-2 warmup (8 → 12 ep)** — exp10 had a small early loss spike; longer warmup damps it.
3. **Heavier Stage-2 mixup (0.2 → 0.4)** — once Stage 1 is stronger, the head can tolerate more regularization.
4. **Stage-2 grad accumulation (effective batch 32)** — only if val acc plateaus before ep 80.

## Stack

- Python 3.10+, **`uv`** for env management (`uv sync` reproduces the lockfile).
- PyTorch (CUDA 12.8 wheels via `pyproject.toml`), torchvision, timm, Hydra, OmegaConf, Pillow, wandb.
- Hardware: **single CUDA GPU**. All configs default `training.device=cuda`.

## Setting up a fresh clone

```bash
git clone <repo-url> challenge_sb && cd challenge_sb
uv sync                                  # installs deps from pyproject.toml + uv.lock

# Data + labels are NOT in the repo. Link them in:
ln -s /path/to/your/processed_data processed_data    # train/val/test frame folders
ln -s /path/to/your/labels         labels            # train_video_labels.csv, val_video_labels.csv
mkdir -p checkpoints                                  # Stage 1 + Stage 2 .pt files land here

# W&B (optional but recommended — see "W&B setup" below)
wandb login <your-api-key>

# Smoke test (verifies env + data wiring in ~1 min)
uv run python src/pretrain_vjepa.py experiment=vjepa_pretrain_v8 \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4
```

If you don't have access to the original dataset, `processed_data/` must contain `train/NNN_ClassName/video_<id>/{0001.jpg,...}` and the same for `val/`. See "Data layout" below.

## W&B setup

All v8 configs have `wandb.enabled: true` by default and include a meaningful `run_name`, `tags`, and `notes` block. Activate W&B once per machine:

```bash
wandb login <api-key>                # paste the key once; cached in ~/.netrc
# OR set env var instead:
export WANDB_API_KEY=<api-key>
```

To disable for a single run: `wandb.enabled=false`. To override the entity/project, edit `src/configs/wandb/default.yaml` or pass `wandb.project=<name>`.

## Repo layout

- [src/train.py](src/train.py) — supervised training loop (Stage 2 + baselines); saves best-by-val-acc checkpoint with full merged Hydra config inside the `.pt`.
- [src/pretrain_vjepa.py](src/pretrain_vjepa.py) — V-JEPA SSL pretraining loop (Stage 1); writes `vjepa_stage1_*.pt`.
- [src/evaluate.py](src/evaluate.py) — rebuilds model from checkpoint config; reports top-1 / top-5 on `dataset.val_dir`.
- [src/create_submission.py](src/create_submission.py) — inference on test frames; writes `video_name,predicted_class` CSV.
- [src/dataset/video_dataset.py](src/dataset/video_dataset.py) — `VideoFrameDataset` + `collect_video_samples`; returns `(T, C, H, W)` tensors; class index from `NNN_Name` folder prefix.
- [src/models/](src/models/) — each classifier maps `(B, T, C, H, W) → (B, num_classes)`. Registered models live in the `build_model` branches in [src/train.py](src/train.py).
  - [src/models/vjepa.py](src/models/vjepa.py) — Stage 1 V-JEPA (encoder + EMA teacher + hybrid predictor + mask samplers).
  - [src/models/_vit_utils.py](src/models/_vit_utils.py) — shared ViT helpers (`PatchEmbed`, sincos 2D/3D pos-embed).
  - [src/models/spatial/vit_mae.py](src/models/spatial/vit_mae.py) — Stage-2 spatial encoder (loads V-JEPA Stage-1 weights).
  - [src/models/temporal/spacetime.py](src/models/temporal/spacetime.py) — joint space-time transformer head.
- [src/utils.py](src/utils.py) — seeds, transforms, train/val split helper.
- [src/configs/](src/configs/) — Hydra: `config.yaml` composes `model/`, `data/`, `train/`, `wandb/`, `experiment/`.

## Commands (general)

```bash
# Smoke test FIRST — always — before any long run (foreground)
uv run python src/train.py experiment=baseline_from_scratch \
  dataset.max_samples=64 training.epochs=1 training.batch_size=4

# Full training — ALWAYS run inside tmux so an SSH drop can't kill the job.
#   Detach from a live session:  Ctrl-b  then  d
#   Reattach later:               tmux attach -t <name>
#   List sessions:                tmux ls
#   Kill a stuck session:         tmux kill-session -t <name>
tmux new -s train "uv run python src/train.py experiment=<name> 2>&1 | tee <name>.log"

# nohup fallback if tmux is unavailable:
#   nohup uv run python src/train.py experiment=... > train.log 2>&1 &  disown
#   tail -f train.log

# Evaluate + submit (short, foreground)
uv run python src/evaluate.py training.checkpoint_path=<path>.pt
uv run python src/create_submission.py training.checkpoint_path=<path>.pt
```

Best checkpoint path is `training.checkpoint_path`. The checkpoint embeds the merged Hydra config, so `evaluate.py` / `create_submission.py` reload the correct architecture automatically — **no duplicate model registry to update**.

## Adding a new model (the one pattern to follow)

1. `src/models/<name>.py` implementing `nn.Module` with `(B,T,C,H,W) → (B,num_classes)`.
2. Register one branch in [`build_model` in src/train.py](src/train.py) keyed on `cfg.model.name`.
3. `src/configs/model/<name>.yaml` with `# @package _global_` and `model.name: <name>`.
4. `src/configs/experiment/<name>.yaml` with `defaults: [- override /model: <name>]`.

`evaluate.py` and `create_submission.py` need no edits.

## Data layout

```
challenge_sb/
├── src/
├── processed_data/        ← symlink or real dir
│   ├── train/  NNN_ClassName/video_<id>/0001.jpg…
│   ├── val/    NNN_ClassName/video_<id>/0001.jpg…
│   └── test/   video_<id>/                ← no class grouping
├── labels/                ← symlink or real dir
│   ├── train_video_labels.csv   ← video_name,class_idx
│   └── val_video_labels.csv     ← video_name,class_idx
├── checkpoints/           ← .pt outputs land here
└── …
```

[src/configs/data/default.yaml](src/configs/data/default.yaml) uses `${hydra:runtime.cwd}/processed_data/{train,val,test}`. Run Hydra commands from the repo root and paths resolve correctly — no CLI overrides needed.

**Label sources.** Two equivalent sources exist for train/val:
- Folder prefix: `processed_data/{train,val}/NNN_ClassName/video_<id>/` → class `NNN`. Current `src/dataset/video_dataset.py` parses labels this way.
- CSVs in `labels/`: `video_name,class_idx` — authoritative mapping; useful if the folder layout drifts.

Class index space is `0..32` (33 classes), with **27 absent** from on-disk folders and CSVs — no training example carries label 27. Keep `num_classes=33` (that output head is never exercised).

## Working preferences

- **Propose changes before editing.** Explain the approach + tradeoffs first; wait for a "go" before modifying files. This is the default mode for this repo.
- **Always smoke-test before long training.** Use `dataset.max_samples=64 training.epochs=1 training.batch_size=4` to verify the pipeline end-to-end before committing GPU hours.
- **Every training run must survive SSH disconnection, host reboot, and accidental tmux kills.** No exceptions — a full training call is never launched in the foreground. Wrap it in `tmux new -s <name> "<cmd> 2>&1 | tee <name>.log"` (preferred) or `nohup` (fallback). Beyond tmux, every long run should: (a) checkpoint frequently so a crash loses at most one epoch, (b) be relaunchable from the latest checkpoint without manual surgery, (c) tee a persistent log file.
- **Each run gets a short experiment note on W&B.** 1–3 lines + a tiny ASCII sketch of the architecture or the key change vs. the previous run. Keep it terse — no prose paragraphs.
- **Monitor training on a regular cadence.** Don't fire-and-forget: periodically re-check tmux session health, GPU utilization (`nvidia-smi`), and the loss/accuracy trajectory in the log or on W&B. Catch divergence, NaNs, or stalls early.
- **End every run with a concise interpretation.** Once training finishes (or is stopped), write a 2–4 line takeaway: what the run achieved (best val acc), what worked / didn't, what it implies for the next iteration. No essays.
- **Always create a submission at the end of a successful run.** Run `uv run python src/create_submission.py training.checkpoint_path=<best>.pt` and surface the resulting CSV path. Don't leave a finished run without a submission file.
- **No horizontal/vertical flip augmentation.** SSv2-style labels are direction-sensitive ("pushing left" vs "pushing right") — flips destroy the signal. Tubelet=2, num_frames=4 are hard caps for this challenge.
- **Cleanup scope is restricted.** MAE/iBOT models + their pretrain scripts were removed during the v8 refactor; `cnn_baseline`, `cnn_lstm`, `cmt`, all ResNet/TSM/ViT spatial encoders, and all legacy temporal heads are kept as base repo.
- **Current goal:** maximize Track 1 accuracy on the challenge via V-JEPA v8. Open to recent SOTA ideas (temporal transformers, 3D / (2+1)D CNNs, masked modeling, V-JEPA SSL, distillation, two-stream) — but constrained to from-scratch training. Surface options with tradeoffs rather than picking unilaterally.
- User background: comfortable with PyTorch; frame explanations accordingly (no need to re-explain basics, but video-specific terminology is worth grounding briefly).
