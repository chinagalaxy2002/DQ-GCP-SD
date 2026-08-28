"""Aggregate mechanism summaries across variant/seed analysis JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json


def _mean(records, getter):
    values = [float(getter(row)) for row in records]
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)} if values else {"mean": 0.0, "std": 0.0, "n": 0}


def summarize(paths):
    analyses = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        analyses.append(data)
    result: Dict[str, Any] = {"analyses": [str(Path(path).resolve()) for path in paths], "variants": {}}
    for data in analyses:
        variant = data.get("manifest", {}).get("options", {}).get("variant") or data.get("mode", "unknown")
        rows = data.get("records", [])
        multi = [row for row in rows if row.get("num_gt", 0) >= 2]
        clean = [row for row in multi if row.get("clean_multi", False)]
        result["variants"].setdefault(variant, []).append({
            "checkpoint": data.get("manifest", {}).get("checkpoint"),
            "epoch": data.get("manifest", {}).get("epoch"),
            "num_records": len(rows),
            "multi": len(multi),
            "clean_multi": len(clean),
            "coverage@5_50": _mean(multi, lambda r: r["coverage"]["d4"]["coverage@5_50"]),
            "aec_d1_final": _mean(multi, lambda r: r["d1_final"]["aec"]),
            "aec_d4_final": _mean(multi, lambda r: r["d4_final"]["aec"]),
            "ecr_d1_final": _mean(multi, lambda r: r["d1_final"]["ecr"]),
            "ecr_d4_final": _mean(multi, lambda r: r["d4_final"]["ecr"]),
            "clean_coverage@5_50": _mean(clean, lambda r: r["coverage"]["d4"]["coverage@5_50"]),
            "clean_aec_d4_final": _mean(clean, lambda r: r["d4_final"]["aec"]),
            "clean_ecr_d4_final": _mean(clean, lambda r: r["d4_final"]["ecr"]),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(args.analysis)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
