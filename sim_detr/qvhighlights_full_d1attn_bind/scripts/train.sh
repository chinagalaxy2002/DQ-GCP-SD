#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-0}"
seed="${2:-2023}"
bind_coef="${3:-0.2}"
dec_layers="${4:-2}"
shift $(( $# >= 4 ? 4 : $# ))
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exp_id="full_d1attn_bind${bind_coef}_bsz8_dec${dec_layers}_seed${seed}"

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${PYTHONPATH:-}:${repo_root}" \
"${python_bin}" -m sim_detr.qvhighlights_full_d1attn_bind.train \
  --seed "${seed}" \
  --gpu_id 0 \
  --device 0 \
  --binding_loss_coef "${bind_coef}" \
  --dec_layers "${dec_layers}" \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/highlight_train_release.jsonl \
  --eval_path data/highlight_val_release.jsonl \
  --eval_split_name val \
  --v_feat_dirs data/slowfast_features data/clip_b32_vid_k4 \
  --v_feat_dim 5376 \
  --t_feat_dir data/clip_b32_txt_k4 \
  --t_feat_dim 2048 \
  --bsz 8 \
  --eval_bsz 8 \
  --max_windows 8 \
  --results_root results_semantic_calibration \
  --exp_id "${exp_id}" \
  --lr 5e-5 \
  --n_epoch 400 \
  --max_es_cnt 50 \
  --lr_drop 400 \
  --lw_saliency 1.0 \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  --label_loss_coef 4 \
  --enc_layers 2 \
  "${@}"
