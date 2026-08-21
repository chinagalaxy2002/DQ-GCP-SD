#!/usr/bin/env bash
set -euo pipefail

variant_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The persistent two-GPU dispatcher skips jobs launched by the concurrent
# launcher below.  V3_GRID_FORCE_RUN=1 is reserved for that launcher.
if [[ -d "${variant_dir}/.manual_launch" && "${V3_GRID_FORCE_RUN:-0}" != "1" ]]; then
  echo "Skipping externally launched variant $(basename "${variant_dir}")"
  exit 0
fi

exec "${variant_dir}/../train_variant.sh" "${variant_dir}"
