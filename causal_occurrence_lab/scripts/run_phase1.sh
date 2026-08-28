#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA="${CAUSAL_DATA:-$ROOT/data/highlight_test_with_gt.jsonl}"
DQ_CHECKPOINT="${CAUSAL_DQ_CHECKPOINT:-$ROOT/results_dq_cgp/select_v3_parameter/beta_0p050_bind_0p20_route_0p010_center/runs/hl-video_tef-sim_detr_dq_cgp_v3_grid_beta_0p050_bind_0p20_route_0p010_center_seed2017-2026_08_21_08_50_54/model_best.ckpt}"
BASELINE_CHECKPOINT="${CAUSAL_BASELINE_CHECKPOINT:-$ROOT/test_sim_detr/hl-video_tef-exp-2026_08_20_16_41_03/model_best.ckpt}"
DEVICE="${CAUSAL_DEVICE:-cuda:0}"
OUT_ROOT="${CAUSAL_OUTPUT_ROOT:-$ROOT/causal_occurrence_lab/outputs/phase1}"
ANALYSIS_FLAGS=()
if [[ "${CAUSAL_SKIP_FORMAL_METRICS:-0}" == "1" ]]; then
  ANALYSIS_FLAGS+=(--skip-formal-metrics)
fi

python causal_occurrence_lab/inspect_dataset.py \
  --data "$DATA" \
  --output "$OUT_ROOT/dataset_census.json"

for MODE in dq_active dq_beta_zero dq_stripped; do
  python causal_occurrence_lab/analyze_checkpoints.py \
    --checkpoint "$DQ_CHECKPOINT" \
    --mode "$MODE" \
    --data "$DATA" \
    --device "$DEVICE" \
    --output "$OUT_ROOT/$MODE" \
    "${ANALYSIS_FLAGS[@]}"
done

python causal_occurrence_lab/analyze_checkpoints.py \
  --checkpoint "$BASELINE_CHECKPOINT" \
  --mode baseline \
  --data "$DATA" \
  --device "$DEVICE" \
  --output "$OUT_ROOT/baseline" \
  "${ANALYSIS_FLAGS[@]}"

python causal_occurrence_lab/compare_runs.py \
  --left "$OUT_ROOT/baseline/analysis.json" \
  --right "$OUT_ROOT/dq_active/analysis.json" \
  --subset multi \
  --activity-matched \
  --output "$OUT_ROOT/dq_vs_baseline_bootstrap.json"

python causal_occurrence_lab/compare_prediction_modes.py \
  --left "$OUT_ROOT/dq_active/analysis.json" \
  --right "$OUT_ROOT/dq_beta_zero/analysis.json" \
  --output "$OUT_ROOT/dq_active_vs_beta_zero.json"

python causal_occurrence_lab/compare_prediction_modes.py \
  --left "$OUT_ROOT/dq_beta_zero/analysis.json" \
  --right "$OUT_ROOT/dq_stripped/analysis.json" \
  --output "$OUT_ROOT/dq_beta_zero_vs_stripped.json"

python causal_occurrence_lab/summarize_phase1.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/phase1_summary.json"
