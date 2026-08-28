#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CAUSAL_DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_occurrence_lab/outputs/three_seeds}"
DEVICE="${CAUSAL_DEVICE_ID:-0}"
GPU_ID="${CAUSAL_GPU_ID:-0}"

for SEED in 2023 2024 2025; do
  for VARIANT in baseline full no_bind supervision_only union_bind; do
    python causal_occurrence_lab/train_causal.py \
      --variant "$VARIANT" \
      --output-dir "$OUT_ROOT/${VARIANT}_seed${SEED}" \
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
      --seed "$SEED" \
      --device "$DEVICE" \
      --gpu_id "$GPU_ID" \
      --VTC_loss_coef 0.3 \
      --CTC_loss_coef 0.5 \
      --lr 0.0001 \
      --lr_drop 100 \
      --n_epoch "${CAUSAL_EPOCHS:-200}" \
      "$@"
  done
done
