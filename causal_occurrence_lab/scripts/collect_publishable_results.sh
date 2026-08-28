#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PHASE_ROOT="${CAUSAL_PHASE1_ROOT:-$ROOT/causal_occurrence_lab/outputs/phase1}"
EVAL_ROOT="${CAUSAL_EVAL_ROOT:-$ROOT/causal_occurrence_lab/outputs/evaluated_variants}"
TRAIN_ROOT="${CAUSAL_RUN_ROOT:-$ROOT/causal_occurrence_lab/outputs/causal_training}"
DEST="${CAUSAL_RESULTS_ROOT:-$ROOT/causal_occurrence_lab/results}"

mkdir -p "$DEST/phase1/manifests" "$DEST/causal_training"

for name in \
  dataset_census.json \
  phase1_summary.json \
  dq_vs_baseline_bootstrap.json \
  dq_active_vs_beta_zero.json \
  dq_beta_zero_vs_stripped.json \
  similarity_baseline.json \
  similarity_dq_active.json; do
  if [[ -f "$PHASE_ROOT/$name" ]]; then
    cp "$PHASE_ROOT/$name" "$DEST/phase1/$name"
  fi
done

if [[ -f "$ROOT/causal_occurrence_lab/outputs/dq_strip_equivalence_seed2017.json" ]]; then
  cp "$ROOT/causal_occurrence_lab/outputs/dq_strip_equivalence_seed2017.json" \
    "$DEST/phase1/dq_strip_equivalence_seed2017.json"
fi

for mode in baseline dq_active dq_beta_zero dq_stripped; do
  if [[ -f "$PHASE_ROOT/$mode/manifest.json" ]]; then
    cp "$PHASE_ROOT/$mode/manifest.json" "$DEST/phase1/manifests/${mode}_manifest.json"
  fi
done

if [[ -f "$EVAL_ROOT/causal_training_summary.json" ]]; then
  cp "$EVAL_ROOT/causal_training_summary.json" "$DEST/causal_training/summary.json"
fi

for variant in full_repro_seed2017 no_bind_seed2017 supervision_only_seed2017 union_bind_seed2017; do
  source_dir="$TRAIN_ROOT/$variant"
  publish_dir="$DEST/causal_training/$variant"
  if [[ ! -d "$source_dir" ]]; then
    continue
  fi
  mkdir -p "$publish_dir"
  for name in variant.json causal_train.log train.log.txt eval.log.txt; do
    if [[ -f "$source_dir/$name" ]]; then
      cp "$source_dir/$name" "$publish_dir/$name"
    fi
  done
done

echo "Collected compact artifacts under $DEST"
