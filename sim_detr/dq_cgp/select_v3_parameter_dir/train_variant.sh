#!/usr/bin/env bash
# Launch one isolated DQ-CGP V3 grid point from scratch.
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 VARIANT_DIRECTORY" >&2
  exit 2
fi

variant_dir="$(cd "$1" && pwd)"
repo_root="$(cd "${variant_dir}/../../../.." && pwd)"
variant_name="$(basename "${variant_dir}")"
source "${variant_dir}/variant.env"

python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
gpu_id="${V3_GRID_GPU_ID:-0}"
num_workers="${V3_GRID_NUM_WORKERS:-4}"
seed="${V3_GRID_SEED:-2017}"
n_epoch="${V3_GRID_EPOCHS:-200}"
results_root="${repo_root}/results_dq_cgp/select_v3_parameter/${variant_name}/runs"
exp_id="sim_detr_dq_cgp_v3_grid_${variant_name}_seed${seed}"
module="sim_detr.dq_cgp.select_v3_parameter_dir.${variant_name}.train_dq_cgp"
resume_args=()
if [[ -n "${V3_GRID_RESUME:-}" ]]; then
  [[ -f "${V3_GRID_RESUME}" ]] || { echo "V3_GRID_RESUME does not exist: ${V3_GRID_RESUME}" >&2; exit 2; }
  resume_args+=(--resume "${V3_GRID_RESUME}")
  if [[ "${V3_GRID_RESUME_ALL:-1}" == "1" ]]; then
    resume_args+=(--resume_all)
  fi
fi

[[ -x "${python_bin}" ]] || { echo "PYTHON_BIN is not executable: ${python_bin}" >&2; exit 2; }
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "V3_GRID_GPU_ID must be non-negative" >&2; exit 2; }
[[ "${num_workers}" =~ ^[0-9]+$ ]] || { echo "V3_GRID_NUM_WORKERS must be non-negative" >&2; exit 2; }
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "V3_GRID_SEED must be non-negative" >&2; exit 2; }
[[ "${n_epoch}" =~ ^[0-9]+$ ]] || { echo "V3_GRID_EPOCHS must be non-negative" >&2; exit 2; }

mkdir -p "${results_root}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting ${variant_name}: beta=${V3_GRID_BETA}, bind=${V3_GRID_BIND}, route=${V3_GRID_ROUTE}"
exec "${python_bin}" -m "${module}" \
  --dset_name hl --ctx_mode video_tef \
  --train_path "${repo_root}/data/highlight_train_release.jsonl" \
  --eval_path "${repo_root}/data/highlight_val_release.jsonl" --eval_split_name val \
  --v_feat_dirs "${repo_root}/data/slowfast_features" "${repo_root}/data/clip_b32_vid_k4" \
  --v_feat_dim 5376 --t_feat_dir "${repo_root}/data/clip_b32_txt_k4" --t_feat_dim 2048 \
  --results_root "${results_root}" --exp_id "${exp_id}" --seed "${seed}" \
  --device 0 --gpu_id "${gpu_id}" --num_workers "${num_workers}" \
  --data_ratio 1.0 --bsz 32 --eval_bsz 100 --lr 0.0001 --lr_drop 100 --wd 0.0001 \
  --n_epoch "${n_epoch}" --max_es_cnt 200 --grad_clip 0.1 --max_q_l 32 --max_v_l 75 \
  --clip_length 2 --max_windows 10 --position_embedding sine --enc_layers 2 --dec_layers 4 \
  --dim_feedforward 1024 --hidden_dim 256 --input_dropout 0.5 --dropout 0.1 \
  --txt_drop_ratio 0 --nheads 8 --num_queries 10 --n_input_proj 2 --temperature 0.07 \
  --lw_saliency 1.0 --saliency_margin 0.2 --span_loss_type l1 --set_cost_span 10 \
  --set_cost_giou 1 --set_cost_class 4 --set_cost_mask 6 --span_loss_coef 10 --giou_loss_coef 1 \
  --label_loss_coef 4 --eos_coef 0.1 --contrastive_align_loss_coef 0.0 --mask_loss_coef 6 \
  --iou_scores_loss_coef 2 --VTC_loss_coef 0.3 --CTC_loss_coef 0.5 \
  --use_query_cgp --query_cgp_num_basis 16 --query_cgp_prompt_length 6 \
  --query_cgp_router_hidden_dim 256 --query_cgp_frf_hidden_dim 512 --query_cgp_temperature 1.0 \
  --query_cgp_beta "${V3_GRID_BETA}" --query_cgp_after_layer 0 \
  --query_cgp_binding_loss_coef "${V3_GRID_BIND}" --query_cgp_route_loss_coef "${V3_GRID_ROUTE}" \
  "${resume_args[@]}"
