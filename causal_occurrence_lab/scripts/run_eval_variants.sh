#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA="${CAUSAL_DATA:-$ROOT/data/highlight_test_with_gt.jsonl}"
DEVICE="${CAUSAL_DEVICE:-cuda:0}"
RUN_ROOT="${CAUSAL_RUN_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training}"
OUT_ROOT="${CAUSAL_EVAL_ROOT:-$ROOT/causal_occurrence_lab/outputs/evaluated_variants}"

for VARIANT in full_repro no_bind supervision_only union_bind; do
  RUN_NAME="${VARIANT}_seed2017"
  CHECKPOINT="$RUN_ROOT/$RUN_NAME/model_best.ckpt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
  python causal_occurrence_lab/analyze_checkpoints.py \
    --checkpoint "$CHECKPOINT" \
    --mode dq_active \
    --data "$DATA" \
    --device "$DEVICE" \
    --output "$OUT_ROOT/$RUN_NAME" \
    $([[ "$VARIANT" == "supervision_only" ]] && echo --no-injection || true)
done
