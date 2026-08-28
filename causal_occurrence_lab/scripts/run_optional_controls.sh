#!/usr/bin/env bash
set -euo pipefail

# Optional single-seed controls.  This script is not part of the first-round
# execution; it is provided for the follow-up decision after the four primary
# runs have been inspected.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CAUSAL_DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_training_optional}"
DEVICE="${CAUSAL_DEVICE_ID:-0}"
GPU_ID="${CAUSAL_GPU_ID:-0}"

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
  --causal-dec-layers 4
  --bsz 32
  --eval_bsz 100
  --lr 0.0001
  --lr_drop 100
  --n_epoch "${CAUSAL_EPOCHS:-200}"
  --seed 2017
  --gpu_id "$GPU_ID"
  --device "$DEVICE"
  --VTC_loss_coef 0.3
  --CTC_loss_coef 0.5
)

for VARIANT in no_route architecture_only wrong_bind native_bind; do
  python causal_occurrence_lab/train_causal.py \
    --variant "$VARIANT" \
    --output-dir "$OUT_ROOT/${VARIANT}_seed2017" \
    "${COMMON_ARGS[@]}" \
    "$@"
done
