#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-2023}"
PYTHON_BIN="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
EXP_ID="${EXP_ID:-ls_dq_cgp_d2_seed${SEED}}"

exec "${PYTHON_BIN}" -m sim_detr.soccer_gmr_ls_dq_cgp.train \
  --exp_id "${EXP_ID}" \
  --gpu_id "${GPU_ID}" \
  --seed "${SEED}" \
  "${@:3}"
