#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CAUSAL_DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training}"
DEVICE="${CAUSAL_DEVICE_ID:-0}"
GPU_ID="${CAUSAL_GPU_ID:-0}"
EPOCHS="${CAUSAL_EPOCHS:-200}"
BASELINE_INIT="${CAUSAL_BASELINE_CHECKPOINT:-}"

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
  --eval_bsz 100
  --lr 0.0001
  --lr_drop 100
  --n_epoch "$EPOCHS"
  --seed 2017
  --gpu_id "$GPU_ID"
  --device "$DEVICE"
  --VTC_loss_coef 0.3
  --CTC_loss_coef 0.5
)

VARIANT_LIST="${CAUSAL_VARIANTS:-full no_bind supervision_only union_bind}"
read -r -a VARIANTS <<< "$VARIANT_LIST"

for VARIANT in "${VARIANTS[@]}"; do
  RUN_NAME="${VARIANT}_seed2017"
  if [[ "$VARIANT" == "full" ]]; then
    RUN_NAME="full_repro_seed2017"
  fi
  INIT_ARGS=()
  if [[ -n "$BASELINE_INIT" ]]; then
    INIT_ARGS+=(--causal-init-from "$BASELINE_INIT")
  fi
  python causal_occurrence_lab/train_causal.py \
    --variant "$VARIANT" \
    --output-dir "$OUT_ROOT/$RUN_NAME" \
    "${COMMON_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    "$@"
done
