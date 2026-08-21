# Sim-DETR + Parameter-Tied All-Interlayer DQ-CGP

This isolated experiment applies one shared DQ-CGP module in every gap of the
official four-layer Sim-DETR decoder:

```text
D1 -> shared DQ-CGP -> D2 -> same DQ-CGP -> D3 -> same DQ-CGP -> D4
```

Only one `query_cgp` module is registered. Its temporal binding projections,
prompt bases, router, FRF and fixed beta are reused three times. Therefore the
trainable parameter count and state-dict tensor keys remain identical to the
single-stage V3 and tied-dual-stage models; only computation increases.

## Controlled design

- Insertion layers are fixed to zero-indexed layers `0, 1, 2`.
- Native reference refinement and auxiliary supervision run before every
  adapter call.
- Early, middle and late diagnostics are snapshotted separately so later calls
  cannot overwrite earlier autograd graphs.
- Each stage uses the final Hungarian assignment to compute its own binding
  and routing loss.
- The three stage losses are averaged before applying the original global
  weights:

```text
L_bind  = (L_early_bind  + L_middle_bind  + L_late_bind)  / 3
L_route = (L_early_route + L_middle_route + L_late_route) / 3
L_total = L_SimDETR + 0.2 * L_bind + 0.01 * L_route
```

This prevents the DQ auxiliary-loss budget from becoming three times larger.
All non-DQ data, feature, optimizer, schedule, seed and Sim-DETR settings match
the existing scratch experiments.

## Canonical from-scratch training

From the repository root:

```bash
cd /home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR
CUDA_VISIBLE_DEVICES=0 \
TIED_ALL_DQCGP_GPU_ID=0 \
bash sim_detr/dq_cgp/tied_all_stage/scripts/train_tied_all_stage_scratch.sh
```

The canonical script trains for 200 epochs and intentionally rejects all CLI
arguments. It has no `--init_from`, `--resume`, `--resume_all` or altered
`--start_epoch` path.

Operational environment variables:

```text
PYTHON_BIN
TIED_ALL_DQCGP_GPU_ID
TIED_ALL_DQCGP_NUM_WORKERS
TIED_ALL_DQCGP_RESULTS_ROOT
TIED_ALL_DQCGP_EXP_ID
TIED_ALL_DQCGP_SEED
```

Default result root:

```text
results_dq_cgp/tied_all_stage_scratch/runs/
```

Background execution:

```bash
CUDA_VISIBLE_DEVICES=0 TIED_ALL_DQCGP_GPU_ID=0 \
nohup bash sim_detr/dq_cgp/tied_all_stage/scripts/train_tied_all_stage_scratch.sh \
  > tied_all_stage_dq_cgp_seed2017.log 2>&1 &
```

If physical GPU 1 is selected using `CUDA_VISIBLE_DEVICES=1`, the process sees
that card as logical GPU 0, so keep `TIED_ALL_DQCGP_GPU_ID=0`.

## Checkpoint and inference

Checkpoints carry the architecture contract:

```text
architecture_id = sim_detr_tied_all_stage_dq_cgp_v1
```

Resume and inference validate this identifier, the saved DQ/insertion config,
and the complete state dict.

```bash
bash sim_detr/dq_cgp/tied_all_stage/scripts/inference_tied_all_stage.sh \
  /path/to/model_best.ckpt val
```

The three calls can be disabled independently during causal evaluation:

```text
--eval_disable_query_cgp_early
--eval_disable_query_cgp_middle
--eval_disable_query_cgp_late
```

These switches evaluate one jointly trained checkpoint; they do not replace
independently trained ablations.

## Diagnostics

```text
query_cgp_temporal_attention
query_cgp_basis_weights
query_cgp_middle_temporal_attention
query_cgp_middle_basis_weights
query_cgp_late_temporal_attention
query_cgp_late_basis_weights
```

Shell checks:

```bash
bash -n sim_detr/dq_cgp/tied_all_stage/scripts/train_tied_all_stage_scratch.sh
bash -n sim_detr/dq_cgp/tied_all_stage/scripts/train_tied_all_stage.sh
bash -n sim_detr/dq_cgp/tied_all_stage/scripts/inference_tied_all_stage.sh
```
