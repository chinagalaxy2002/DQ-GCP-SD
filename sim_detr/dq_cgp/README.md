# Sim-DETR + DQ-CGP: training and inference

## Released V3 (D1) checkpoint and results

The published checkpoint is selected at epoch 103 using validation
`MR-full-mAP`. Download `model_best.ckpt` and its matching `opt.json` from the
[GitHub release](https://github.com/chinagalaxy2002/DQ-GCP-SD/releases/tag/v3-d1-qvhighlights).

| Method | test R1@0.5 | test R1@0.7 | test mAP@0.5 | test mAP@0.75 | test mAP Avg. | val R1@0.5 | val R1@0.7 | val mAP@0.5 | val mAP@0.75 | val mAP Avg. |
| ------ | ----------: | ----------: | -----------: | ------------: | ------------: | ---------: | ---------: | ----------: | -----------: | -----------: |
| Sim-DETR baseline | 66.93 | **51.56** | 67.75 | 48.89 | 47.60 | **68.32** | 53.81 | **69.03** | 50.77 | 49.14 |
| **DQ-CGP V3 (D1)** | **67.96** | 51.36 | **68.94** | **49.01** | **48.06** | 67.81 | **54.06** | 68.81 | **51.01** | **49.66** |
| Improvement | +1.03 | -0.20 | +1.19 | +0.12 | **+0.46** | -0.51 | +0.25 | -0.22 | +0.24 | **+0.52** |

The baseline and DQ-CGP use the same QVHighlights splits, offline features,
seed, optimizer, and 200-epoch training budget. Validation is used for
checkpoint selection; `highlight_test_with_gt.jsonl` is evaluated only after
selection.

Run all commands from the repository root. The scripts resolve the repository
and data paths automatically. Use `PYTHON_BIN` when the desired Python is not
the active environment:

```bash
export PYTHON_BIN=/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python
```

## Main effectiveness experiment: train V3 from scratch

```bash
CUDA_VISIBLE_DEVICES=0 \
bash sim_detr/dq_cgp/scripts/train_dq_cgp_v3_scratch.sh
```

This is the canonical experiment for testing DQ-CGP effectiveness. It never
loads the baseline checkpoint. The repository's original Sim-DETR checkpoint
is used only as the metric reference. The script rejects all command-line
arguments so that `--init_from` or `--resume` cannot be introduced by mistake.

The experiment exactly preserves the local baseline protocol: four decoder
layers, two encoder layers, SlowFast + CLIP video features, R2-Tuning CLIP
text features, batch size 32, seed 2017, and a 200-epoch budget. DQ-CGP is
inserted once after decoder layer 1. Outputs are isolated under
`results_dq_cgp/v3_scratch/runs/`. Compare the two final best checkpoints only
after both runs have completed the same 200-epoch protocol.

Runtime/output settings are configurable through environment variables.
`DQCGP_SEED` is a scientific setting: when it is changed, compare against an
original Sim-DETR baseline trained with the same seed.

```bash
PYTHON_BIN=/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python \
DQCGP_GPU_ID=0 \
DQCGP_NUM_WORKERS=4 \
DQCGP_RESULTS_ROOT=/path/to/results_dq_cgp \
DQCGP_EXP_ID=sim_detr_dq_cgp_v3_scratch_seed2017 \
bash sim_detr/dq_cgp/scripts/train_dq_cgp_v3_scratch.sh
```

When `CUDA_VISIBLE_DEVICES` exposes a single physical GPU, keep
`DQCGP_GPU_ID=0` because it is the process-local GPU index.

For a background run:

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
bash sim_detr/dq_cgp/scripts/train_dq_cgp_v3_scratch.sh \
> dq_cgp_v3_scratch_seed2017.log 2>&1 &
```

## Optional fine-tuning mode—not the main effectiveness comparison

```bash
bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh \
  --init_from /path/to/sim_detr_baseline_best.ckpt
```

`--init_from` loads all baseline tensors and leaves only `query_cgp.*` at
their DQ-CGP initial values. Here, “baseline” means the repository's original
`sim_detr.model.SimDETR`, trained without DQ-CGP; it is not a separately
defined baseline. The loader rejects checkpoints that already contain
`query_cgp.*`. `--init_from` is mutually exclusive with `--resume`.

Do not use this mode for the primary claim that DQ-CGP outperforms the
original from-scratch Sim-DETR. A warm-start experiment needs an additional
control that continues training the baseline for the same number of epochs.

## Resume DQ-CGP training

Weights only:

```bash
bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh \
  --resume /path/to/model_latest.ckpt
```

Exact continuation, including optimizer, scheduler, epoch, best metric and
early-stopping counter:

```bash
bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh \
  --resume /path/to/model_latest.ckpt \
  --resume_all
```

Training validates every epoch and saves `model_best.ckpt` and
`model_latest.ckpt`. Periodic epoch checkpoints follow the official Sim-DETR
schedule.

## Evaluate a trained DQ-CGP checkpoint

Validation split:

```bash
bash sim_detr/dq_cgp/scripts/inference_dq_cgp.sh \
  /path/to/model_best.ckpt val
```

Explicit annotation file:

```bash
bash sim_detr/dq_cgp/scripts/inference_dq_cgp.sh \
  /path/to/model_best.ckpt val data/highlight_val_release.jsonl
```

The inference entry point reloads the architecture from the `opt.json` stored
beside the checkpoint and uses strict checkpoint loading. A checkpoint copied
elsewhere must be copied together with its `opt.json`.

## Reproduce the reported V3 (D1) result

This is the selected single-insertion DQ-CGP V3 setting: DQ-CGP is inserted
after decoder layer 1, with `beta=0.05`, binding-loss weight `0.20`, and
routing-loss weight `0.01`.  It is trained from scratch for the same
200-epoch budget as Sim-DETR, using seed 2017.

```bash
cd /path/to/Sim-DETR
export PYTHON_BIN=/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python

# When only one physical GPU is visible, use its process-local index 0.
CUDA_VISIBLE_DEVICES=0 DQCGP_GPU_ID=0 \
  bash sim_detr/dq_cgp/scripts/train_dq_cgp_v3_scratch.sh
```

The released checkpoint is `model_best.ckpt`; retain its adjacent `opt.json`.
For the run reported in this repository it is selected at epoch 103 by
validation MR-full-mAP.  Its validation metrics are R1@0.5=67.81,
R1@0.7=54.06, mAP@0.5=68.81, mAP@0.75=51.01, and mAP Avg.=49.66.

Evaluate that checkpoint on the validation annotations:

```bash
CKPT=results_dq_cgp/v3_scratch/runs/\
hl-video_tef-sim_detr_dq_cgp_v3_scratch_seed2017-2026_08_20_18_12_29/\
model_best.ckpt

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sim_detr.dq_cgp.inference_dq_cgp \
  --resume "$CKPT" \
  --eval_split_name val \
  --eval_path data/highlight_val_release.jsonl \
  --eval_results_dir results_dq_cgp/reproduce_v3_d1/val \
  --gpu_id 0 --device 0
```

The local GT test file is not used for model selection.  After choosing the
checkpoint on validation, run this one-off test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m sim_detr.dq_cgp.inference_dq_cgp \
  --resume "$CKPT" \
  --eval_split_name test_with_gt \
  --eval_path data/highlight_test_with_gt.jsonl \
  --eval_results_dir results_dq_cgp/reproduce_v3_d1/test_with_gt \
  --gpu_id 0 --device 0
```

Expected test metrics are R1@0.5=67.96, R1@0.7=51.36, mAP@0.5=68.94,
mAP@0.75=49.01, and mAP Avg.=48.06.  The evaluator writes the full metrics to
`hl_test_with_gt_submission_metrics.json` in `--eval_results_dir`.

## Verify the migration

```bash
PYTHONPATH=. python -m pytest sim_detr/dq_cgp/tests -q
bash -n sim_detr/dq_cgp/scripts/train_dq_cgp_v3_scratch.sh
bash -n sim_detr/dq_cgp/scripts/train_dq_cgp.sh
bash -n sim_detr/dq_cgp/scripts/inference_dq_cgp.sh
```

The tests cover the V3 tensor path, masked temporal attention, fixed-β
residual, Hungarian-matched binding/routing losses, the D1→DQ→D2 insertion
order, native reference preservation, β=0 baseline identity, and checkpoint
compatibility.

## Flexible utility for smoke tests and non-primary runs

```bash
# Select a physical GPU when the process can see multiple GPUs.
bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh --gpu_id 1

# Or expose one GPU and keep the internal id at zero.
CUDA_VISIBLE_DEVICES=1 \
  bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh --gpu_id 0

# Smoke test: four batches, one validation pass, isolated debug directory.
bash sim_detr/dq_cgp/scripts/train_dq_cgp.sh --debug --num_workers 0
```
