#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2023}"
run_tag="${SOCCER_GMR_RUN_TAG:-nullaware_masklogits}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
log_dir="${repo_root}/results_soccer_gmr_csc/tmux_logs"
mkdir -p "${log_dir}"

launch() {
  local session="$1"
  local variant="$2"
  local gpu="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${repo_root}' && SOCCER_GMR_RUN_TAG='${run_tag}' bash '${script_dir}/train_variant.sh' '${variant}' '${gpu}' '${seed}' 2>&1 | tee '${log_dir}/${variant}_${run_tag}_seed${seed}.log'"
}

launch "soccer_csc_na_native_${seed}" native 0
launch "soccer_csc_na_static_${seed}" static 1
launch "soccer_csc_na_full_${seed}" full 1

tmux list-sessions -F '#{session_name} #{session_windows} #{session_attached}' | grep "soccer_csc_na_.*_${seed}"
