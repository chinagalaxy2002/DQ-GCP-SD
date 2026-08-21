#!/usr/bin/env bash
# Canonical effectiveness experiment:
#   repository Sim-DETR trained from scratch  vs.  Sim-DETR + DQ-CGP V3
#
# This script deliberately has no --init_from/--resume path.  The original
# baseline checkpoint is an evaluation reference only; it is never loaded
# into the DQ-CGP run.

set -euo pipefail

if (( $# != 0 )); then
  echo "This canonical scratch script accepts no CLI arguments." >&2
  echo "Use DQCGP_* environment variables documented in dq_cgp/README.md." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
seed="${DQCGP_SEED:-2017}"
gpu_id="${DQCGP_GPU_ID:-0}"
num_workers="${DQCGP_NUM_WORKERS:-4}"
results_root="${DQCGP_RESULTS_ROOT:-${repo_root}/results_dq_cgp/v3_scratch/runs}"
exp_id="${DQCGP_EXP_ID:-sim_detr_dq_cgp_v3_scratch_seed${seed}}"

if [[ "${python_bin}" == */* ]]; then
  [[ -x "${python_bin}" ]] || {
    echo "PYTHON_BIN is not executable: ${python_bin}" >&2
    exit 2
  }
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

for value_name in seed gpu_id num_workers; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "${value_name} must be a non-negative integer; got ${value}" >&2
    exit 2
  }
done

[[ "${exp_id}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "DQCGP_EXP_ID may contain only letters, numbers, '.', '_' and '-'." >&2
  exit 2
}

train_path="${repo_root}/data/highlight_train_release.jsonl"
eval_path="${repo_root}/data/highlight_val_release.jsonl"
slowfast_dir="${repo_root}/data/slowfast_features"
clip_video_dir="${repo_root}/data/clip_b32_vid_k4"
clip_text_dir="${repo_root}/data/clip_b32_txt_k4"

for required_file in "${train_path}" "${eval_path}"; do
  [[ -f "${required_file}" ]] || {
    echo "Required annotation file is missing: ${required_file}" >&2
    exit 2
  }
done
for required_dir in "${slowfast_dir}" "${clip_video_dir}" "${clip_text_dir}"; do
  [[ -d "${required_dir}" ]] || {
    echo "Required feature directory is missing: ${required_dir}" >&2
    exit 2
  }
done

echo "Starting Sim-DETR + DQ-CGP V3 from scratch"
echo "seed=${seed} gpu_id=${gpu_id} exp_id=${exp_id}"
echo "results_root=${results_root}"

# Every Sim-DETR argument below matches the locally trained repository
# baseline, including its 200-epoch training budget.  The only additions are
# the DQ-CGP module and its two V3 auxiliary losses.
exec "${python_bin}" -m sim_detr.dq_cgp.train_dq_cgp \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path "${train_path}" \
  --eval_path "${eval_path}" \
  --eval_split_name val \
  --v_feat_dirs "${slowfast_dir}" "${clip_video_dir}" \
  --v_feat_dim 5376 \
  --t_feat_dir "${clip_text_dir}" \
  --t_feat_dim 2048 \
  --results_root "${results_root}" \
  --exp_id "${exp_id}" \
  --seed "${seed}" \
  --device 0 \
  --gpu_id "${gpu_id}" \
  --num_workers "${num_workers}" \
  --data_ratio 1.0 \
  --bsz 32 \
  --eval_bsz 100 \
  --lr 0.0001 \
  --lr_drop 100 \
  --wd 0.0001 \
  --n_epoch 200 \
  --max_es_cnt 200 \
  --grad_clip 0.1 \
  --max_q_l 32 \
  --max_v_l 75 \
  --clip_length 2 \
  --max_windows 10 \
  --position_embedding sine \
  --enc_layers 2 \
  --dec_layers 4 \
  --dim_feedforward 1024 \
  --hidden_dim 256 \
  --input_dropout 0.5 \
  --dropout 0.1 \
  --txt_drop_ratio 0 \
  --nheads 8 \
  --num_queries 10 \
  --n_input_proj 2 \
  --temperature 0.07 \
  --lw_saliency 1.0 \
  --saliency_margin 0.2 \
  --span_loss_type l1 \
  --set_cost_span 10 \
  --set_cost_giou 1 \
  --set_cost_class 4 \
  --set_cost_mask 6 \
  --span_loss_coef 10 \
  --giou_loss_coef 1 \
  --label_loss_coef 4 \
  --eos_coef 0.1 \
  --contrastive_align_loss_coef 0.0 \
  --mask_loss_coef 6 \
  --iou_scores_loss_coef 2 \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  --use_query_cgp \
  --query_cgp_num_basis 16 \
  --query_cgp_prompt_length 6 \
  --query_cgp_router_hidden_dim 256 \
  --query_cgp_frf_hidden_dim 512 \
  --query_cgp_temperature 1.0 \
  --query_cgp_beta 0.05 \
  --query_cgp_after_layer 0 \
  --query_cgp_binding_loss_coef 0.2 \
  --query_cgp_route_loss_coef 0.01
