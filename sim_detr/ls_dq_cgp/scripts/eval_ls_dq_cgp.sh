#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
checkpoint="${LS_DQ_CGP_CKPT:?Set LS_DQ_CGP_CKPT to an LS-DQ-CGP checkpoint}"
gpu_id="${LS_DQ_CGP_GPU_ID:-0}"
eval_path="${LS_DQ_CGP_EVAL_PATH:-${repo_root}/data/highlight_val_release.jsonl}"
split="${LS_DQ_CGP_SPLIT:-val}"
output_root="${LS_DQ_CGP_EVAL_ROOT:-$(dirname "${checkpoint}")/causal_eval}"

for mode in active static_bypass context_roll; do
  mode_flag=()
  [[ "${mode}" == static_bypass ]] && mode_flag=(--ls_static_bypass)
  [[ "${mode}" == context_roll ]] && mode_flag=(--ls_context_roll)
  "${python_bin}" -m sim_detr.ls_dq_cgp.inference_ls_dq_cgp \
    --resume "${checkpoint}" \
    --eval_split_name "${split}" \
    --eval_path "${eval_path}" \
    --eval_results_dir "${output_root}/${mode}" \
    --device 0 \
    --gpu_id "${gpu_id}" \
    "${mode_flag[@]}"
done
