#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${LCB_DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${LCB_OUTPUT_ROOT:-$ROOT/Layeerconsistentbinding/outputs}"
DEVICE="${LCB_DEVICE_ID:-0}"
GPU_ID="${LCB_GPU_ID:-0}"
EPOCHS="${LCB_EPOCHS:-200}"

COMMON_ARGS=(
  --dset_name hl
  --ctx_mode video_tef
  --train_path "$DATA_ROOT/highlight_train_release.jsonl"
  --eval_path "$DATA_ROOT/highlight_val_release.jsonl"
  --eval_split_name val
  --v_feat_dirs "$DATA_ROOT/slowfast_features" "$DATA_ROOT/clip_b32_vid_k4"
  --v_feat_dim 5376
  --t_feat_dir "$DATA_ROOT/clip_b32_txt_k4"
  --t_feat_dim 2048
  --dec_layers 4
  --enc_layers 2
  --bsz 32
  --eval_bsz 32
  --lr 0.0001
  --lr_drop 100
  --n_epoch "$EPOCHS"
  --seed 2017
  --gpu_id "$GPU_ID"
  --device "$DEVICE"
  --VTC_loss_coef 0.3
  --CTC_loss_coef 0.5
  --lcb-d1-bind-coef 0.5
  --lcb-late-bind-coef 0.1
  --lcb-owner-cons-coef 0.1
  --lcb-drop-coef 0.1
  --lcb-drop-margin 0.05
)

echo "=== Running Layer-Consistent Binding (LCB Acquire -> Preserve) Training ==="
python Layeerconsistentbinding/train_lcb.py \
  --variant lcb_full \
  --output-dir "$OUT_ROOT/lcb_full_seed2017" \
  "${COMMON_ARGS[@]}" \
  "$@"
