# Sim-DETR + Parameter-Tied Dual-Stage DQ-CGP

This directory is an isolated experiment derived from the faithful Sim-DETR
DQ-CGP V3 migration. It tests one controlled change: the **same** DQ-CGP
module is applied twice, after decoder layer 1 and after decoder layer 3.

```text
D1 -> shared DQ-CGP -> D2 -> D3 -> the same DQ-CGP -> D4
```

There is only one registered `query_cgp` module. The second call reuses its
bases, router, temporal-binding projections, FRF and fixed beta. No
`query_cgp_late.*` parameters are created, so the trainable parameter count and
state-dict tensor keys are identical to the single-stage V3 model. Compute is
higher because the module executes twice.

## What stays unchanged

- Official Sim-DETR data, SlowFast/CLIP/TEF video features and R2-Tuning CLIP
  text features.
- Four decoder layers, DAB reference refinement, query competition, matcher,
  prediction heads and evaluation protocol.
- Native reference refinement and auxiliary supervision occur before each
  adapter call. Therefore the early call changes D2-D4 content refinement; the
  late call changes only the content passed from D3 to D4.
- DQ-CGP settings: 16 bases, prompt length 6 and fixed `beta=0.05`.
- Global DQ loss weights: binding `0.2`, routing `0.01`.

The early and late diagnostics are snapshotted immediately after their
respective calls. Binding and routing are calculated separately at the two
depths and then averaged:

```text
L_bind  = (L_bind_early  + L_bind_late)  / 2
L_route = (L_route_early + L_route_late) / 2
```

This keeps the total auxiliary-loss budget equal to single-stage V3. Routing
is not calculated on concatenated cross-stage routes, which would introduce a
different diversity objective.

## Canonical from-scratch training

Run from the repository root:

```bash
cd /home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR
CUDA_VISIBLE_DEVICES=0 \
TIED_DQCGP_GPU_ID=0 \
bash sim_detr/dq_cgp/tied_dual_stage/scripts/train_tied_dual_stage_scratch.sh
```

This canonical script deliberately rejects command-line arguments and has no
`--init_from`, `--resume`, `--resume_all` or nonzero `--start_epoch` path. It
trains the tied model from random initialization for 200 epochs using the same
non-DQ settings as the single-stage V3 scratch run.

Operational overrides are provided only through:

```text
PYTHON_BIN
TIED_DQCGP_GPU_ID
TIED_DQCGP_NUM_WORKERS
TIED_DQCGP_RESULTS_ROOT
TIED_DQCGP_EXP_ID
TIED_DQCGP_SEED
```

Changing the seed is a scientific change: compare it with baseline and
single-stage runs using the same seed.

Default output root:

```text
results_dq_cgp/tied_dual_stage_scratch/runs/
```

Background execution:

```bash
CUDA_VISIBLE_DEVICES=0 TIED_DQCGP_GPU_ID=0 \
nohup bash sim_detr/dq_cgp/tied_dual_stage/scripts/train_tied_dual_stage_scratch.sh \
  > tied_dual_stage_dq_cgp_seed2017.log 2>&1 &
```

The flexible script is reserved for smoke tests, explicit warm-start studies
or resuming a tied checkpoint:

```bash
bash sim_detr/dq_cgp/tied_dual_stage/scripts/train_tied_dual_stage.sh [extra args]
```

Do not use the flexible script for the canonical scratch comparison.

## Checkpoints and inference

Tied checkpoints contain:

```text
architecture_id = sim_detr_tied_dual_dq_cgp_v1
```

Training resume and inference require this identifier, an exact match on the
saved DQ-CGP/insertion configuration, and strict state-dict loading. This is
necessary because parameter tying makes the model state keys identical to
single-stage V3; strict loading alone cannot detect a wrong single-stage
checkpoint or a different tied insertion policy.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash sim_detr/dq_cgp/tied_dual_stage/scripts/inference_tied_dual_stage.sh \
  /path/to/model_best.ckpt val
```

The two calls can be disabled independently for causal evaluation without
changing the shared beta buffer:

```bash
# Keep early, disable late.
bash sim_detr/dq_cgp/tied_dual_stage/scripts/inference_tied_dual_stage.sh \
  /path/to/model_best.ckpt val --eval_disable_query_cgp_late

# Disable both calls.
bash sim_detr/dq_cgp/tied_dual_stage/scripts/inference_tied_dual_stage.sh \
  /path/to/model_best.ckpt val \
  --eval_disable_query_cgp_early --eval_disable_query_cgp_late
```

These are inference-time ablations of one jointly trained checkpoint, not
substitutes for independently trained baseline or single-stage experiments.
When either call is disabled, use the retrieval metrics for the ablation; the
combined two-stage DQ eval-loss entry is reported as zero because one required
diagnostic branch is intentionally absent.

## Diagnostics and verification

The model exposes two distinct snapshots despite sharing one module:

```text
query_cgp_temporal_attention
query_cgp_basis_weights
query_cgp_late_temporal_attention
query_cgp_late_basis_weights
```

If `pytest` is installed, run:

```bash
/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python -m pytest -q \
  sim_detr/dq_cgp/tied_dual_stage/tests
bash -n sim_detr/dq_cgp/tied_dual_stage/scripts/train_tied_dual_stage_scratch.sh
bash -n sim_detr/dq_cgp/tied_dual_stage/scripts/train_tied_dual_stage.sh
bash -n sim_detr/dq_cgp/tied_dual_stage/scripts/inference_tied_dual_stage.sh
```

The important regression conditions are: one shared object called twice;
early and late snapshots both retain gradients; no `query_cgp_late.*` state
keys; equal parameter count to single-stage V3; and strict tied-checkpoint
round-trip.
