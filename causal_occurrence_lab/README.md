# Causal occurrence lab for Sim-DETR

This directory is an isolated experiment harness for testing whether
candidate-specific occurrence binding is learned in Sim-DETR.  It does not
modify `sim_detr/`, the production DQ-CGP implementation, or the existing
training results.

The central analysis is:

```text
output Hungarian assignment
        !=
candidate-to-occurrence evidence binding
```

The harness captures the native decoder cross-attention at all four decoder
layers, evaluates each layer with both its own Hungarian assignment and the
final D4 assignment, and uses Sim-DETR's official ranking score:

```text
foreground probability * sigmoid(IoU score)
```

For DQ checkpoints it supports `active`, `beta_zero`, and `stripped` modes.
The stripped mode builds vanilla Sim-DETR and removes only `query_cgp.*`
parameters from the checkpoint.  `verify_strip_equivalence.py` checks the
numerical equivalence before any mechanism claim is made.

## Quick start

Run commands from the Sim-DETR repository root:

```bash
python causal_occurrence_lab/inspect_dataset.py \
  --data data/highlight_test_with_gt.jsonl \
  --output causal_occurrence_lab/outputs/test_census.json

python causal_occurrence_lab/analyze_checkpoints.py \
  --checkpoint /path/to/dq/model_best.ckpt \
  --mode dq_active \
  --split test \
  --data data/highlight_test_with_gt.jsonl \
  --output causal_occurrence_lab/outputs/dq_active_test
```

`--mode` is one of `baseline`, `dq_active`, `dq_beta_zero`, or
`dq_stripped`.  The checkpoint's saved `opt` is used to reconstruct the
architecture and feature paths; command-line dataset/device overrides are
available when a checkpoint was moved.

## Completed Native Binding pair

The completed pure Native Binding experiment is documented in
[`NATIVE_BINDING_RESULTS.md`](NATIVE_BINDING_RESULTS.md). It includes only the
fully trained Sim-DETR baseline and NativeBind `lambda=0.5`; incomplete
coefficient runs are excluded.

Launch the exact seed-2017 Baseline vs NativeBind pair on two GPUs:

```bash
bash causal_occurrence_lab/scripts/run_native_binding_pair.sh
```

The launcher shares one argument list across both jobs. The only experimental
difference is `--variant native_bind --query_cgp_binding_loss_coef 0.5` for the
second job. For sequential execution on one GPU:

```bash
CAUSAL_PARALLEL=0 BASELINE_GPU_ID=0 NATIVE_GPU_ID=0 \
bash causal_occurrence_lab/scripts/run_native_binding_pair.sh
```

If the fixed baseline is already available, train only NativeBind with:

```bash
bash causal_occurrence_lab/scripts/run_native_binding_lambda_0p5.sh
```

After validation-based checkpoint selection, evaluate both `model_best.ckpt`
files with the same test protocol:

```bash
CAUSAL_BASELINE_CHECKPOINT=/path/to/baseline/model_best.ckpt \
NATIVE_BIND_CHECKPOINT=/path/to/native_bind/model_best.ckpt \
bash causal_occurrence_lab/scripts/eval_native_binding_completed.sh
```

The evaluation launcher reconstructs both checkpoints with `--mode baseline`,
runs the formal D1-D4 and occurrence-subset metrics, and writes a compact
`summary.json`. The published copy is
[`results/native_binding/summary.json`](results/native_binding/summary.json).

The first causal training round is intentionally explicit:

```bash
bash causal_occurrence_lab/scripts/run_first_round.sh
```

It runs `full_repro` followed by the planned `no_bind`, `supervision_only`,
and `union_bind` variants.  Add `--debug` or set `CAUSAL_EPOCHS` before using
the script for a smoke run.  Formal runs should keep the release protocol
(seed 2017, four decoder layers, 200 epochs, validation `MR-full-mAP`
checkpoint selection).  The coefficients and target can also be overridden
with `--query_cgp_binding_loss_coef`, `--query_cgp_route_loss_coef`,
`--query_cgp_inject`/`--no-query_cgp_inject`, and
`--query_cgp_binding_target`.

When two GPUs are available, the four first-round jobs can be launched
concurrently (two jobs per GPU):

```bash
bash causal_occurrence_lab/scripts/run_parallel_first_round.sh
```

The parallel launcher writes to `outputs/causal_training_parallel/` by
default.  Use `tmux ls` and the per-variant `*_tmux.log` files to monitor it.
After all four checkpoints finish, evaluate them with:

```bash
CAUSAL_RUN_ROOT="$PWD/causal_occurrence_lab/outputs/causal_training_parallel" \
bash causal_occurrence_lab/scripts/run_eval_variants.sh
```

`scripts/finalize_first_round.sh` can be left running in a fifth tmux session;
it waits for all four fixed-seed jobs and then performs the evaluation and
compact artifact collection automatically.

## Checkpoint conventions

The repository currently contains several DQ-CGP result families.  The
analysis scripts never silently choose a checkpoint when an explicit path is
provided.  Without a path, the default resolver prefers the V3 configuration
with `beta=0.05`, `binding=0.20`, and `route=0.01`; inspect the printed path
and record it in the experiment manifest.  A matching seed-2017 four-layer
baseline is available at
`test_sim_detr/hl-video_tef-exp-2026_08_20_16_41_03/model_best.ckpt`; the
`run_phase1.sh` helper uses it by default.  Pass `--checkpoint` explicitly to
the Python analyzer when comparing another baseline.

## Experiment order

1. Dataset census and existing-checkpoint D1--D4 analysis.
2. DQ beta-zero versus stripped equivalence.
3. `no_bind`, `supervision_only`, and `union_bind` at seed 2017.
4. Only if warranted: `no_route`, `architecture_only`, `native_bind`, and
   `wrong_bind`.
5. If needed later, trajectory snapshots, similarity strata, and
   activity-matched secondary analyses.  This delivery deliberately does not
   run multi-seed experiments; all formal commands are fixed to seed 2017.

All causal loss controls use the production routing objective
`H(W|C) - H(W)` and aggregate matched routes across the whole batch.

## Current execution status

The full Phase-1 analysis has been run on the 1,542-query test split.  The
formal first-round jobs use only seed 2017 and are launched concurrently by
`run_parallel_first_round.sh`; their checkpoints and logs are kept under
`outputs/causal_training_parallel/`.  Multi-seed training is intentionally
not run.  Generated checkpoints, logs, and JSON outputs are ignored by the
local `.gitignore`; compact publishable summaries are copied into
`results/` after evaluation.

The dedicated Baseline vs NativeBind `lambda=0.5` pair is also complete. Its
validation, test, occurrence-subset, and D1/D4 mechanism results are published
in `NATIVE_BINDING_RESULTS.md`; runs that did not reach 200 epochs are not
reported.
