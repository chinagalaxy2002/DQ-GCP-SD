"""Create a compact table from the four Phase-1 analysis directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json


MODES = ("baseline", "dq_active", "dq_beta_zero", "dq_stripped")


def _mean(rows, getter):
    values = [float(getter(row)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def summarize(root: str | Path) -> dict:
    root = Path(root)
    result = {"root": str(root.resolve()), "modes": {}}
    for mode in MODES:
        path = root / mode / "analysis.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("records", [])
        multi = [row for row in rows if row.get("num_gt", 0) >= 2]
        clean = [row for row in multi if row.get("clean_multi", False)]
        result["modes"][mode] = {
            "records": len(rows),
            "multi": len(multi),
            "clean_multi": len(clean),
            "d4_mr_full_map": (
                data.get("formal_metrics", {})
                .get("d4", {})
                .get("brief", {})
                .get("MR-full-mAP")
            ),
            "multi_coverage@5_50": _mean(
                multi, lambda row: row["coverage"]["d4"]["coverage@5_50"]
            ),
            "multi_d1_aec": _mean(multi, lambda row: row["d1_final"]["aec"]),
            "multi_d4_aec": _mean(multi, lambda row: row["d4_final"]["aec"]),
            "multi_d1_ecr": _mean(multi, lambda row: row["d1_final"]["ecr"]),
            "multi_d4_ecr": _mean(multi, lambda row: row["d4_final"]["ecr"]),
            "clean_d4_aec": _mean(clean, lambda row: row["d4_final"]["aec"]),
            "clean_d4_ecr": _mean(clean, lambda row: row["d4_final"]["ecr"]),
        }
    for name in ("dq_active_vs_beta_zero", "dq_beta_zero_vs_stripped"):
        path = root / f"{name}.json"
        if path.is_file():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(args.root)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
