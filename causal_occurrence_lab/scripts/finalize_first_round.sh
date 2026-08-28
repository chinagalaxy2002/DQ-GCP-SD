#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRAIN_ROOT="${CAUSAL_RUN_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training_parallel}"
EVAL_ROOT="${CAUSAL_EVAL_ROOT:-$ROOT/causal_occurrence_lab/outputs/evaluated_variants_parallel}"
VARIANTS=(full_repro_seed2017 no_bind_seed2017 supervision_only_seed2017 union_bind_seed2017)
SESSIONS=(dqgcp_full dqgcp_no_bind dqgcp_supervision dqgcp_union)

echo "Waiting for fixed-seed-2017 first-round jobs under $TRAIN_ROOT"
while true; do
  all_finished=true
  for variant in "${VARIANTS[@]}"; do
    log="$TRAIN_ROOT/$variant/causal_train.log"
    if [[ ! -f "$log" ]] || ! rg -q "Finished variant" "$log"; then
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
