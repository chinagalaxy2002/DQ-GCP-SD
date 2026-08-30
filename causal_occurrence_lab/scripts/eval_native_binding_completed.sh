#!/usr/bin/env bash
set -euo pipefail

# Evaluate only the two completed runs: the fixed Sim-DETR baseline and
# NativeBind lambda=0.5.  Both checkpoints are reconstructed in baseline mode.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA="${CAUSAL_DATA:-$ROOT/data/highlight_test_with_gt.jsonl}"
DEVICE="${CAUSAL_DEVICE:-cuda:0}"
BASELINE_CHECKPOINT="${CAUSAL_BASELINE_CHECKPOINT:-$ROOT/test_sim_detr/hl-video_tef-exp-2026_08_20_16_41_03/model_best.ckpt}"
NATIVE_CHECKPOINT="${NATIVE_BIND_CHECKPOINT:-$ROOT/causal_occurrence_lab/outputs/native_bind_lambda_0p5_seed2017/model_best.ckpt}"
BASELINE_VAL_METRICS="${CAUSAL_BASELINE_VAL_METRICS:-$(dirname "$BASELINE_CHECKPOINT")/best_hl_val_preds_metrics.json}"
NATIVE_VAL_METRICS="${NATIVE_BIND_VAL_METRICS:-$(dirname "$NATIVE_CHECKPOINT")/best_hl_val_preds_metrics.json}"
OUT_ROOT="${NATIVE_BIND_EVAL_ROOT:-$ROOT/causal_occurrence_lab/outputs/native_binding_completed_test}"

for path in \
  "$DATA" \
  "$BASELINE_CHECKPOINT" \
  "$NATIVE_CHECKPOINT" \
  "$BASELINE_VAL_METRICS" \
  "$NATIVE_VAL_METRICS"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python causal_occurrence_lab/analyze_checkpoints.py \
  --checkpoint "$BASELINE_CHECKPOINT" \
  --mode baseline \
  --split test \
  --data "$DATA" \
  --device "$DEVICE" \
  --eval-bsz 32 \
  --output "$OUT_ROOT/baseline"

python causal_occurrence_lab/analyze_checkpoints.py \
  --checkpoint "$NATIVE_CHECKPOINT" \
  --mode baseline \
  --split test \
  --data "$DATA" \
  --device "$DEVICE" \
  --eval-bsz 32 \
  --output "$OUT_ROOT/native_bind_lambda_0p5"

python causal_occurrence_lab/summarize_native_binding.py \
  --baseline-analysis "$OUT_ROOT/baseline/analysis.json" \
  --native-analysis "$OUT_ROOT/native_bind_lambda_0p5/analysis.json" \
  --baseline-val-metrics "$BASELINE_VAL_METRICS" \
  --native-val-metrics "$NATIVE_VAL_METRICS" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --native-checkpoint "$NATIVE_CHECKPOINT" \
  --output "$OUT_ROOT/summary.json"
