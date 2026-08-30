#!/usr/bin/env bash
set -euo pipefail

# Reproduce the exact Baseline vs NativeBind lambda=0.5 training pair.
# Default: run concurrently on physical GPUs 0 and 1.
# Single-GPU serial mode: CAUSAL_PARALLEL=0 NATIVE_GPU_ID=0 bash <this-script>.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CAUSAL_DATA_ROOT:-$ROOT/data}"
OUT_ROOT="${NATIVE_BIND_PAIR_ROOT:-$ROOT/causal_occurrence_lab/outputs/native_binding_pair_seed2017}"
BASELINE_GPU_ID="${BASELINE_GPU_ID:-0}"
NATIVE_GPU_ID="${NATIVE_GPU_ID:-1}"
PARALLEL="${CAUSAL_PARALLEL:-1}"

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
  --n_epoch 200
  --seed 2017
  --VTC_loss_coef 0.3
  --CTC_loss_coef 0.5
  --num_workers 0
)

mkdir -p "$OUT_ROOT/baseline_seed2017" "$OUT_ROOT/native_bind_lambda_0p5_seed2017"

run_baseline() {
  python causal_occurrence_lab/train_causal.py \
    --variant baseline \
    --output-dir "$OUT_ROOT/baseline_seed2017" \
    --gpu_id "$BASELINE_GPU_ID" \
    --device "$BASELINE_GPU_ID" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_native() {
  python causal_occurrence_lab/train_causal.py \
    --variant native_bind \
    --query_cgp_binding_loss_coef 0.5 \
    --output-dir "$OUT_ROOT/native_bind_lambda_0p5_seed2017" \
    --gpu_id "$NATIVE_GPU_ID" \
    --device "$NATIVE_GPU_ID" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

if [[ "$PARALLEL" == "1" ]]; then
  run_baseline "$@" >"$OUT_ROOT/baseline_seed2017/console.log" 2>&1 &
  baseline_pid=$!
  run_native "$@" >"$OUT_ROOT/native_bind_lambda_0p5_seed2017/console.log" 2>&1 &
  native_pid=$!

  cleanup() {
    kill "$baseline_pid" "$native_pid" 2>/dev/null || true
  }
  trap cleanup INT TERM

  baseline_status=0
  native_status=0
  wait "$baseline_pid" || baseline_status=$?
  wait "$native_pid" || native_status=$?
  trap - INT TERM
  if (( baseline_status != 0 || native_status != 0 )); then
    echo "Training failed: baseline=$baseline_status native_bind=$native_status" >&2
    exit 1
  fi
else
  run_baseline "$@" 2>&1 | tee "$OUT_ROOT/baseline_seed2017/console.log"
  run_native "$@" 2>&1 | tee "$OUT_ROOT/native_bind_lambda_0p5_seed2017/console.log"
fi

echo "Completed Baseline and NativeBind lambda=0.5 training under: $OUT_ROOT"
