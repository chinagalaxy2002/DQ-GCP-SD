#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRAIN_ROOT="${CAUSAL_RUN_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training_parallel}"
EVAL_ROOT="${CAUSAL_EVAL_ROOT:-$ROOT/causal_occurrence_lab/outputs/evaluated_variants_parallel}"
SESSION_PREFIX="${CAUSAL_SESSION_PREFIX:-dqgcp}"
VARIANTS=(full_repro_seed2017 no_bind_seed2017 supervision_only_seed2017 union_bind_seed2017)
SESSIONS=(
  "${SESSION_PREFIX}_full"
  "${SESSION_PREFIX}_no_bind"
  "${SESSION_PREFIX}_supervision"
  "${SESSION_PREFIX}_union"
)

echo "Waiting for fixed-seed-2017 first-round jobs under $TRAIN_ROOT"
while true; do
  all_finished=true
  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$index]}"
    log="$TRAIN_ROOT/$variant/causal_train.log"
    if [[ ! -f "$log" ]] || ! rg -q "Finished variant" "$log"; then
      if ! tmux has-session -t "${SESSIONS[$index]}" 2>/dev/null; then
        echo "Training session ${SESSIONS[$index]} exited before completion" >&2
        exit 1
      fi
      all_finished=false
      break
    fi
  done

  if [[ "$all_finished" == true ]]; then
    any_session=false
    for session in "${SESSIONS[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        any_session=true
        break
      fi
    done
    if [[ "$any_session" == false ]]; then
      break
    fi
  fi
  sleep 30
done

echo "All four training jobs finished; running variant evaluation"
CAUSAL_RUN_ROOT="$TRAIN_ROOT" \
CAUSAL_EVAL_ROOT="$EVAL_ROOT" \
bash causal_occurrence_lab/scripts/run_eval_variants.sh

python causal_occurrence_lab/summarize_causal_training.py \
  --root "$EVAL_ROOT" \
  --train-root "$TRAIN_ROOT" \
  --output "$EVAL_ROOT/causal_training_summary.json"

CAUSAL_RUN_ROOT="$TRAIN_ROOT" \
CAUSAL_EVAL_ROOT="$EVAL_ROOT" \
bash causal_occurrence_lab/scripts/collect_publishable_results.sh

echo "First-round evaluation and compact collection completed"
