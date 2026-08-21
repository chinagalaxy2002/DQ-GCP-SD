# Sim-DETR + Dual-Stage DQ-CGP

This directory is a complete, isolated copy of the single-stage V3 experiment.
It changes one method variable: a second, independently parameterized DQ-CGP
adapter is inserted between decoder layers 3 and 4.

```text
D1 -> DQ-early -> D2 -> D3 -> DQ-late -> D4
```

Both adapters preserve native Sim-DETR reference refinement and auxiliary
supervision: each decoder layer first emits its native reference and auxiliary
state, and only then is its content state adapted for the next layer.

## Controlled comparison

The canonical run starts from random initialization. It does not load the
baseline or the single-stage checkpoint. All data, features, seed, optimizer,
schedule, decoder depth and original losses match the single-stage V3 run.

The early and late adapters use independent bases, routers and FRF parameters,
with identical settings: 16 bases, prompt length 6 and fixed beta 0.05. Each
stage receives its own binding and routing loss. The two stage losses are
averaged before applying the original global weights 0.2 and 0.01, so stacking
does not double the auxiliary-loss budget.

## Canonical scratch training

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
DUAL_DQCGP_GPU_ID=0 \
bash sim_detr/dq_cgp/dual_stage/scripts/train_dual_stage_scratch.sh
```

The canonical script rejects all command-line arguments, including checkpoint
initialization and resume flags. Operational overrides use environment
variables:

```text
PYTHON_BIN
DUAL_DQCGP_GPU_ID
DUAL_DQCGP_NUM_WORKERS
DUAL_DQCGP_RESULTS_ROOT
DUAL_DQCGP_EXP_ID
DUAL_DQCGP_SEED
```

Changing the seed is a scientific setting and requires a same-seed baseline
and single-stage comparison.

Default outputs are isolated under:

```text
results_dq_cgp/dual_stage_scratch/runs/
```

For background execution:

```bash
CUDA_VISIBLE_DEVICES=0 DUAL_DQCGP_GPU_ID=0 \
nohup bash sim_detr/dq_cgp/dual_stage/scripts/train_dual_stage_scratch.sh \
  > dual_stage_dq_cgp_seed2017.log 2>&1 &
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
bash sim_detr/dq_cgp/dual_stage/scripts/inference_dual_stage.sh \
  /path/to/model_best.ckpt val
```

The inference entry rebuilds the dual architecture from the checkpoint's
`opt.json` and uses strict state-dict loading.

For causal adapter ablations, beta overrides are applied only after strict
checkpoint loading, so checkpoint buffers cannot silently undo them:

```bash
# Disable late only (tests whether stacking contributes at inference).
bash sim_detr/dq_cgp/dual_stage/scripts/inference_dual_stage.sh \
  /path/to/model_best.ckpt val --eval_query_cgp_late_beta 0

# Disable both adapters.
bash sim_detr/dq_cgp/dual_stage/scripts/inference_dual_stage.sh \
  /path/to/model_best.ckpt val \
  --eval_query_cgp_early_beta 0 --eval_query_cgp_late_beta 0
```

## Verification

```bash
python -m pytest -q sim_detr/dq_cgp/dual_stage/tests
bash -n sim_detr/dq_cgp/dual_stage/scripts/train_dual_stage_scratch.sh
bash -n sim_detr/dq_cgp/dual_stage/scripts/train_dual_stage.sh
bash -n sim_detr/dq_cgp/dual_stage/scripts/inference_dual_stage.sh
```

Key diagnostics are kept separate:

```text
query_cgp_temporal_attention
query_cgp_basis_weights
query_cgp_late_temporal_attention
query_cgp_late_basis_weights
```

This experiment must be reported as a parameter-increased stacking ablation,
not as an equal-parameter comparison.
