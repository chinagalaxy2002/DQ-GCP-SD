#!/usr/bin/env bash
set -euo pipefail

session="${1:-simdetr_semantic_pilot}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "${root}/results_semantic_calibration"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 1
fi

tmux new-session -d -s "${session}" -n static
tmux send-keys -t "${session}:static" "cd '${root}' && bash sim_detr/semantic_calibration/scripts/train_static.sh 1 2017 2>&1 | tee results_semantic_calibration/static_seed2017.tmux.log" C-m
tmux new-window -t "${session}" -n full
tmux send-keys -t "${session}:full" "cd '${root}' && bash sim_detr/semantic_calibration/scripts/train_full.sh 0 2017 2>&1 | tee results_semantic_calibration/full_seed2017.tmux.log" C-m
tmux select-window -t "${session}:static"
echo "Started tmux session ${session} with Static on GPU 1 and Full on GPU 0."
echo "Attach with: tmux attach -t ${session}"
