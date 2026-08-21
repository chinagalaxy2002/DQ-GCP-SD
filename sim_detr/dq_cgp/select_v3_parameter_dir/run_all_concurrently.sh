#!/usr/bin/env bash
# Durable all-at-once launcher for the nine V3 screening-grid configurations.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
grid_root="${repo_root}/sim_detr/dq_cgp/select_v3_parameter_dir"
results_root="${repo_root}/results_dq_cgp/select_v3_parameter"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"

launch() {
  local physical_gpu="$1"
  local variant="$2"
  mkdir -p "${results_root}/${variant}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" PYTHON_BIN="${python_bin}" \
    V3_GRID_GPU_ID=0 V3_GRID_FORCE_RUN=1 \
    bash "${grid_root}/${variant}/train.sh" \
    > "${results_root}/${variant}/background_launcher.log.txt" 2>&1 &
}

launch 0 beta_0p025_bind_0p10_route_0p005
launch 0 beta_0p025_bind_0p40_route_0p005
launch 0 beta_0p100_bind_0p10_route_0p005
launch 0 beta_0p100_bind_0p40_route_0p005
launch 0 beta_0p050_bind_0p20_route_0p010_center
launch 1 beta_0p025_bind_0p10_route_0p020
launch 1 beta_0p025_bind_0p40_route_0p020
launch 1 beta_0p100_bind_0p10_route_0p020
launch 1 beta_0p100_bind_0p40_route_0p020

wait
