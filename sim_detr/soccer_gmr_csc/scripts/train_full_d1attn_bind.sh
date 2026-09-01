#!/usr/bin/env bash
# Train Full variant + D1 Attention evidence pooling + Hungarian Binding Loss on Soccer-GMR
set -euo pipefail

gpu_id="${1:-1}"
seed="${2:-2023}"
bind_coef="${3:-0.2}"
dec_layers="${4:-2}"
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gmr_root="${SOCCER_GMR_ROOT:-/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval}"
exp_id="full_d1attn_bind${bind_coef}_bsz8_dec${dec_layers}_seed${seed}"

echo "[Launch] Running Full + D1 Attention + Binding Loss (coef=${bind_coef}) on GPU ${gpu_id}..."

CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${PYTHONPATH:-}:${repo_root}" "${python_bin}" -m sim_detr.soccer_gmr_csc.train   --semantic_variant full   --semantic_evidence_source d1_attention   --binding_loss_coef "${bind_coef}"   --dec_layers "${dec_layers}"   --exp_id "${exp_id}"   --seed "${seed}"   --gpu_id 0   --gmr_root "${gmr_root}"   "${@:5}"
