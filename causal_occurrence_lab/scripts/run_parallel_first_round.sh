#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training_parallel}"
EPOCHS="${CAUSAL_EPOCHS:-200}"
SESSION_PREFIX="${CAUSAL_SESSION_PREFIX:-dqgcp}"
mkdir -p "$OUT_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for parallel training" >&2
  exit 1
fi

launch_variant() {
  local session="$1"
  local variant="$2"
  local gpu="$3"
  local log="$OUT_ROOT/${variant}_tmux.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Keeping existing tmux session: $session"
    return
  fi

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && CAUSAL_EPOCHS='$EPOCHS' CAUSAL_VARIANTS='$variant' CAUSAL_OUTPUT_ROOT='$OUT_ROOT' CAUSAL_DEVICE_ID='$gpu' CAUSAL_GPU_ID='$gpu' bash '$ROOT/causal_occurrence_lab/scripts/run_first_round.sh' > '$log' 2>&1"
  echo "Started $variant on GPU $gpu in tmux session $session"
}

# Two jobs per GPU are intentional: the Sim-DETR jobs fit comfortably within
# the available memory, while keeping all four first-round controls aligned
# to the same fixed seed and protocol.
launch_variant "${SESSION_PREFIX}_full" full 0
launch_variant "${SESSION_PREFIX}_no_bind" no_bind 1
launch_variant "${SESSION_PREFIX}_supervision" supervision_only 0
launch_variant "${SESSION_PREFIX}_union" union_bind 1

tmux list-sessions | rg "${SESSION_PREFIX}_(full|no_bind|supervision|union)" || true
