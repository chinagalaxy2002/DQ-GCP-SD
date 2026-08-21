#!/usr/bin/env bash
# Run the nine grid points as two persistent GPU worker queues.
set -euo pipefail

# This script lives in sim_detr/dq_cgp/select_v3_parameter_dir.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
grid_root="${repo_root}/sim_detr/dq_cgp/select_v3_parameter_dir"
results_root="${repo_root}/results_dq_cgp/select_v3_parameter"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"

variants=(
  beta_0p025_bind_0p10_route_0p005
  beta_0p025_bind_0p10_route_0p020
  beta_0p025_bind_0p40_route_0p005
  beta_0p025_bind_0p40_route_0p020
  beta_0p100_bind_0p10_route_0p005
  beta_0p100_bind_0p10_route_0p020
  beta_0p100_bind_0p40_route_0p005
  beta_0p100_bind_0p40_route_0p020
  beta_0p050_bind_0p20_route_0p010_center
)

run_worker() {
  local physical_gpu="$1"
  local start_index="$2"
  local index variant
  for ((index=start_index; index<${#variants[@]}; index+=2)); do
    variant="${variants[index]}"
    mkdir -p "${results_root}/${variant}"
    echo "[$(date '+%F %T')] GPU ${physical_gpu}: starting ${variant}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" PYTHON_BIN="${python_bin}" V3_GRID_GPU_ID=0 \
      bash "${grid_root}/${variant}/train.sh" \
      > "${results_root}/${variant}/launcher.log.txt" 2>&1
    echo "[$(date '+%F %T')] GPU ${physical_gpu}: completed ${variant}"
  done
}

run_worker 0 0 &
worker_zero_pid=$!
run_worker 1 1 &
worker_one_pid=$!
wait "${worker_zero_pid}"
wait "${worker_one_pid}"
