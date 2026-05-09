# Next experiment — Stage 2 improvement candidates

## Diagnosis (from exp2h)

- Best single: exp2h = 0.391 / 0.387 (TTA=1 / TTA=3) real val. Train 0.42 / val 0.39 → **narrow gap → underfitting the task, not overfitting**.
- 6-model ensemble lifts to 0.400 (+1.3 pp). The optimizer assigns non-trivial weight to weaker models (exp2g 0.159, exp2f 0.152, exp2c 0.110), proving **the ensemble has signal exp2h doesn't** — there is genuine recipe-axis diversity to compress into a single student.
- Recipe axes tried: distill (single + 4-way ensemble), born-again, seed/temporal jitter, longer schedule + heavier reg, difference tokens + 3-layer head. Regularization-axis variations have plateaued.
- Constraints: 4 frames hard cap, no hflip, no external pretraining, frozen `mae_stage1.pt` as the SSL artifact. SSv2-style task is direction-sensitive.

## Candidates ranked by upside / effort

### 1. Ensemble-distillation born-again (exp2i) — lead recommendation

Use the 6-model NLL-optimal ensemble's softmax probs *on the training set*
as soft targets for a fresh student, on top of the same exp2h recipe (3-layer
relational temporal transformer, diff tokens, single CE teacher → swap to
ensemble teacher).

- **Why:** the ensemble *is* a strictly stronger teacher than any single model
  we have (it's literally the val-NLL-minimizing combination). Born-again
  on this dataset has a track record: exp2c→2d (+1.0 pp), exp2d→2e (+0.6 pp).
  With a stronger teacher than 2e/2d/2f had, expected lift > those.
- **Cost:** ~30 min to dump train-set softmax for all 6 models, run optimizer
  to produce per-train-clip soft targets, save .npy. Then ~5 h overnight
  training. Code change: small — extend `train.py`'s distill path to load
  precomputed (N_train, K) soft labels by video index.
- **Risk:** low. Worst case: another exp2h-equivalent. Best case: matches or
  beats the ensemble itself in a single model (~0.40, with TTA potentially
  ~0.41).

### 2. FixMatch / Mean-Teacher on the unlabeled test set (exp2j)

Use the 6,913 unlabeled test videos as a semi-supervised signal: weak-aug →
EMA teacher → confidence threshold τ → CE on strong-aug student.

- **Why:** the test distribution is sitting unused. ~7k unlabeled videos at
  the eval distribution is non-trivial. Standard FixMatch/SVFormer recipe.
- **Cost:** code change is medium (unlabeled loader, weak/strong aug split,
  consistency loss with threshold). 1 day to implement + train. Already
  have EMA infra.
- **Risk:** moderate. Pseudo-labels noisy if teacher is weak; τ tuning
  matters. Could pair with #1 (use ensemble as the pseudo-labeler).

### 3. TimeSformer-style divided space-time attention in last K backbone blocks (exp2k)

In the last 4 ViT blocks, insert a temporal-attention pass over same-position
tokens across frames before each spatial pass. Early blocks keep MAE init.

- **Why:** currently the backbone sees one frame at a time; all temporal
  reasoning lives in the head. SSv2 SOTA at our scale (TimeSformer / MViTv2 /
  Uniformer) all do some variant of this. Most "real" architectural
  improvement.
- **Cost:** significant code change (modify ViT block class). ~half-day +
  training. Will need a smoke-test that the spatial path still loads MAE
  weights correctly.
- **Risk:** highest. MAE pretraining isn't space-time aware, so repurposing
  late blocks may disrupt the transfer. With T=4 the temporal axis is small.

### 4. Auxiliary "predict next CLS" head (exp2l)

Tiny MLP head: given frames 1–3 spatial CLS, predict frame 4's spatial CLS
(cosine loss, alongside CE).

- **Why:** the challenge *is* "what happens next." Direct task-matched
  inductive bias.
- **Cost:** small (~50 LOC). Single training run.
- **Risk:** low. Worst case it's a mild regularizer. Upside also bounded.
- Cheap **adjunct** to #1 or #2 (can be added as a second loss term in the
  same run).

## Recommendation

Run **#1 (exp2i, ensemble-distillation born-again)** next. It directly
operationalizes the ensemble lift we just measured. If/when it plateaus,
**#2 (FixMatch)** or **#3 (TimeSformer)** is the next axis. **#4** can be
folded into either #1 or #2 cheaply.

If the user wants more "research-y" ambition, lead with #3 instead — but
expect higher implementation effort and uncertainty.
