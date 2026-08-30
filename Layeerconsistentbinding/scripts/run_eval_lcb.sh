#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-$ROOT/Layeerconsistentbinding/outputs/lcb_full_seed2017/model_best.ckpt}"
DATA="${2:-$ROOT/data/highlight_test_with_gt.jsonl}"
OUT_DIR="${3:-$ROOT/Layeerconsistentbinding/outputs/lcb_full_seed2017/eval_test}"
DEVICE="${LCB_DEVICE:-cuda:0}"

echo "=== Running Layer-Consistent Binding (LCB-Full) Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA"
echo "Output:     $OUT_DIR"

python Layeerconsistentbinding/evaluate_lcb.py \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --split test \
  --output "$OUT_DIR" \
  --device "$DEVICE" \
  "${@:4}"
