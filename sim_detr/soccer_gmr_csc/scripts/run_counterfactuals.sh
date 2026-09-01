#!/usr/bin/env bash
set -euo pipefail

checkpoint="$1"
gpu_id="${2:-0}"
split="${3:-val}"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_root="$(dirname "${checkpoint}")/counterfactuals_${split}"

for spec in \
  'aligned full aligned' \
  'roll-1 full roll-1' \
  'roll-2 full roll-2' \
  'roll-3 full roll-3' \
  'random-derangement full random-derangement' \
  'farthest-context full farthest-context' \
  'uniform full uniform' \
  'static static aligned' \
  'native native aligned'; do
  read -r name semantic context <<<"${spec}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${PYTHONPATH:-}:${repo_root}" \
  "${python_bin}" -m sim_detr.soccer_gmr_csc.inference \
    --checkpoint "${checkpoint}" --split "${split}" --gpu_id 0 \
    --semantic_variant "${semantic}" --context_variant "${context}" \
    --output_dir "${output_root}/${name}"
done
