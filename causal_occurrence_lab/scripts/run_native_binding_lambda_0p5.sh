#!/usr/bin/env bash
set -euo pipefail

# Train the completed pure NativeBind configuration.  This builds vanilla
# Sim-DETR and adds only the training-time D1 native-attention loss.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CAUSAL_DATA_ROOT:-$ROOT/data}"
OUT_DIR="${NATIVE_BIND_OUTPUT_DIR:-$ROOT/causal_occurrence_lab/outputs/native_bind_lambda_0p5_seed2017}"
DEVICE_ID="${CAUSAL_DEVICE_ID:-0}"
GPU_ID="${CAUSAL_GPU_ID:-0}"

python causal_occurrence_lab/train_causal.py \
  --variant native_bind \
  --query_cgp_binding_loss_coef 0.5 \
  --output-dir "$OUT_DIR" \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path "$DATA_ROOT/highlight_train_release.jsonl" \
  --eval_path "$DATA_ROOT/highlight_val_release.jsonl" \
  --eval_split_name val \
  --v_feat_dirs "$DATA_ROOT/slowfast_features" "$DATA_ROOT/clip_b32_vid_k4" \
  --v_feat_dim 5376 \
  --t_feat_dir "$DATA_ROOT/clip_b32_txt_k4" \
  --t_feat_dim 2048 \
  --causal-dec-layers 4 \
  --bsz 32 \
  --eval_bsz 100 \
  --lr 0.0001 \
  --lr_drop 100 \
  --n_epoch 200 \
  --seed 2017 \
  --gpu_id "$GPU_ID" \
  --device "$DEVICE_ID" \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  "$@"
