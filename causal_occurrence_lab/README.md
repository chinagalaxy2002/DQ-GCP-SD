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
causal training variants have only received a short four-batch seed-2017
forward/backward smoke test in this delivery; no formal 200-epoch causal
training claim is recorded here.  Multi-seed training is intentionally not
run.  Generated checkpoints, logs, and JSON outputs are ignored by the local
`.gitignore`; use the scripts to regenerate them in the working tree.
