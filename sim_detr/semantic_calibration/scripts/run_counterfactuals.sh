#!/usr/bin/env bash
set -euo pipefail

checkpoint="$1"
gpu_id="${2:-0}"
shift 2
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" \
CUDA_VISIBLE_DEVICES="${gpu_id}" \
"${python_bin}" -m sim_detr.semantic_calibration.inference_semantic \
  --resume "${checkpoint}" \
  --eval_split_name val \
  --eval_path data/highlight_val_release.jsonl \
  --eval_results_dir "$(dirname "${checkpoint}")/counterfactuals" \
  "$@"
