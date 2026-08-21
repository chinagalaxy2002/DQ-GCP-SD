#!/usr/bin/env bash
# Strict inference for a Sim-DETR + DQ-CGP checkpoint.
#
# Usage:
#   bash sim_detr/dq_cgp/scripts/inference_dq_cgp.sh CHECKPOINT SPLIT [JSONL] [extra args]
# Example:
#   bash sim_detr/dq_cgp/scripts/inference_dq_cgp.sh \
#       results_dq_cgp/.../model_best.ckpt val

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 CHECKPOINT SPLIT [JSONL] [extra args]" >&2
  exit 2
fi

checkpoint=$1
eval_split_name=$2
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python}"

if (( $# > 0 )) && [[ $1 != --* ]]; then
  eval_path=$1
  shift
else
  eval_path="data/highlight_${eval_split_name}_release.jsonl"
fi

"${python_bin}" -m sim_detr.dq_cgp.inference_dq_cgp \
  --resume "${checkpoint}" \
  --eval_split_name "${eval_split_name}" \
  --eval_path "${eval_path}" \
  "$@"
