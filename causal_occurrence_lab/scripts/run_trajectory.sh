#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CHECKPOINT_DIR="${CAUSAL_CHECKPOINT_DIR:?Set CAUSAL_CHECKPOINT_DIR to a directory of snapshots}"
DATA="${CAUSAL_DATA:-$ROOT/data/highlight_val_release.jsonl}"
OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_occurrence_lab/outputs/trajectory}"

python causal_occurrence_lab/analyze_trajectory.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --mode "${CAUSAL_MODE:-dq_active}" \
  --data "$DATA" \
  --device "${CAUSAL_DEVICE:-cuda:0}" \
  --output-dir "$OUT_ROOT" \
  "$@"

