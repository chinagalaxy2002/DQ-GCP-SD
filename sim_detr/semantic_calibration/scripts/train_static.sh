#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-1}"
seed="${2:-2017}"
shift $(( $# >= 2 ? 2 : $# ))
python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python}"
PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" \
CUDA_VISIBLE_DEVICES="${gpu_id}" \
"${python_bin}" -m sim_detr.semantic_calibration.train_semantic \
  --seed "${seed}" \
  --semantic_variant static \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/highlight_train_release.jsonl \
  --eval_path data/highlight_val_release.jsonl \
  --eval_split_name val \
  --v_feat_dirs data/slowfast_features data/clip_b32_vid_k4 \
  --v_feat_dim 5376 \
  --t_feat_dir data/clip_b32_txt_k4 \
  --t_feat_dim 2048 \
  --bsz 32 \
  --results_root results_semantic_calibration \
  --exp_id static_seed${seed} \
  --lr 0.0001 \
  --n_epoch 200 \
  --lw_saliency 1.0 \
  --lr_drop 100 \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  --label_loss_coef 4 \
  --dec_layers 4 \
  --enc_layers 2 \
  "$@"
