#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
seed="${LS_DQ_CGP_SEED:-2017}"
gpu_id="${LS_DQ_CGP_GPU_ID:-0}"
workers="${LS_DQ_CGP_NUM_WORKERS:-4}"
results_root="${LS_DQ_CGP_RESULTS_ROOT:-${repo_root}/results_ls_dq_cgp/runs}"
exp_id="${LS_DQ_CGP_EXP_ID:-sim_detr_ls_dq_cgp_d2_seed${seed}}"

exec "${python_bin}" -m sim_detr.ls_dq_cgp.train_ls_dq_cgp \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path "${repo_root}/data/highlight_train_release.jsonl" \
  --eval_path "${repo_root}/data/highlight_val_release.jsonl" \
  --eval_split_name val \
  --v_feat_dirs "${repo_root}/data/slowfast_features" "${repo_root}/data/clip_b32_vid_k4" \
  --v_feat_dim 5376 \
  --t_feat_dir "${repo_root}/data/clip_b32_txt_k4" \
  --t_feat_dim 2048 \
  --results_root "${results_root}" \
  --exp_id "${exp_id}" \
  --seed "${seed}" \
  --device 0 \
  --gpu_id "${gpu_id}" \
  --num_workers "${workers}" \
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
  --dec_layers 2 \
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
  --mask_loss_coef 6 \
  --iou_scores_loss_coef 2 \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  --ls_num_basis 16 \
  --ls_prompt_length 6 \
  --ls_router_hidden_dim 256 \
  --ls_frf_hidden_dim 512 \
  --ls_temperature 1.0 \
  --ls_binding_loss_coef 0.2
