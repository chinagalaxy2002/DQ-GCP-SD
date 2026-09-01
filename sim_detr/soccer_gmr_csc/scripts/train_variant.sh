#!/usr/bin/env bash
set -euo pipefail

variant="$1"
gpu_id="${2:-0}"
seed="${3:-2023}"
run_tag="${SOCCER_GMR_RUN_TAG:-nullaware_masklogits}"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gmr_root="${SOCCER_GMR_ROOT:-/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval}"

CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${PYTHONPATH:-}:${repo_root}" \
"${python_bin}" -m sim_detr.soccer_gmr_csc.train \
  --semantic_variant "${variant}" \
  --exp_id "${variant}_${run_tag}_seed${seed}" \
  --seed "${seed}" \
  --gpu_id 0 \
  --gmr_root "${gmr_root}" \
  "${@:4}"
