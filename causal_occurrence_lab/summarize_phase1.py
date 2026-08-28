"""Create compact Phase-1 tables from saved occurrence analyses.

The summary keeps the original headline keys for backwards compatibility and
adds count-aware macro/micro binding statistics, margin/enrichment summaries,
and official D4 MR metrics split by occurrence count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json


MODES = ("baseline", "dq_active", "dq_beta_zero", "dq_stripped")
BINDING_SUBSETS = ("all", "multi", "clean_multi", "single", "two", "three_or_more")
MR_SUBSETS = ("single", "multi", "two", "three_or_more")
MR_BRIEF_KEYS = (
    "MR-full-R1@0.5",
    "MR-full-R1@0.7",
    "MR-full-mAP@0.5",
    "MR-full-mAP@0.75",
    "MR-full-mAP",
)


def _mean(rows, getter):
    values = [float(getter(row)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _rows_for_subset(rows, subset):
    if subset == "all":
        return list(rows)
    if subset == "multi":
        return [row for row in rows if row.get("num_gt", 0) >= 2]
    if subset == "clean_multi":
        return [
            row for row in rows
            if row.get("num_gt", 0) >= 2 and row.get("clean_multi", False)
        ]
    if subset == "single":
        return [row for row in rows if row.get("num_gt", 0) <= 1]
    if subset == "two":
        return [row for row in rows if row.get("num_gt", 0) == 2]
    if subset == "three_or_more":
        return [row for row in rows if row.get("num_gt", 0) >= 3]
    raise ValueError(f"Unknown binding subset {subset!r}")


def _number(row, key, default=0.0):
    value = row.get(key, default)
    return float(value) if value is not None else float(default)


def _binding_aggregate(rows, stage):
    """Aggregate one stage with both qid-macro and count-weighted metrics."""

    values = [row.get(stage) for row in rows if row.get(stage) is not None]
    matched = sum(int(_number(value, "num_matched")) for value in values)
    correct = sum(int(_number(value, "num_correct")) for value in values)
    correct_norm = sum(int(_number(value, "num_correct_norm")) for value in values)
    valid_pairs = sum(int(_number(value, "num_valid_pairs")) for value in values)
    collisions = sum(int(_number(value, "num_collisions")) for value in values)
    collisions_norm = sum(
        int(_number(value, "num_collisions_norm")) for value in values
    )

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    result = {
        "num_qids": len(values),
        "num_matched": matched,
        "num_correct": correct,
        "num_correct_norm": correct_norm,
        "num_valid_pairs": valid_pairs,
        "num_collisions": collisions,
        "num_collisions_norm": collisions_norm,
        "aec_macro": _mean(values, lambda value: _number(value, "aec")),
        "aec_norm_macro": _mean(values, lambda value: _number(value, "aec_norm")),
        "aec_micro": ratio(correct, matched),
        "aec_norm_micro": ratio(correct_norm, matched),
        "bm_macro": _mean(
            values, lambda value: _number(value, "binding_margin")
        ),
        "bm_norm_macro": _mean(
            values, lambda value: _number(value, "binding_margin_norm")
        ),
        "own_mass_macro": _mean(values, lambda value: _number(value, "own_mass")),
        "own_enrichment_macro": _mean(
            values, lambda value: _number(value, "own_enrichment")
        ),
        "ecr_macro": _mean(values, lambda value: _number(value, "ecr")),
        "ecr_norm_macro": _mean(values, lambda value: _number(value, "ecr_norm")),
        "ecr_micro": ratio(collisions, valid_pairs),
        "ecr_norm_micro": ratio(collisions_norm, valid_pairs),
    }
    result.update({
        "binding_margin_macro": result["bm_macro"],
        "binding_margin_norm_macro": result["bm_norm_macro"],
        "own_mass": result["own_mass_macro"],
        "own_enrichment": result["own_enrichment_macro"],
    })
    return result


def _formal_brief(metric):
    if not metric:
        return None
    brief = metric.get("brief", {})
    return {key: brief.get(key) for key in MR_BRIEF_KEYS if key in brief}


def _mode_summary(data):
    rows = data.get("records", [])
    multi = _rows_for_subset(rows, "multi")
    clean = _rows_for_subset(rows, "clean_multi")
    d4_brief = _formal_brief(
        (data.get("formal_metrics") or {}).get("d4")
    ) or {}
    binding = {}
    for subset in BINDING_SUBSETS:
        subset_rows = _rows_for_subset(rows, subset)
        binding[subset] = {}
        for layer in range(1, int(data.get("decoder_layers", 4)) + 1):
            for assignment in ("own", "final"):
                stage = f"d{layer}_{assignment}"
                binding[subset][stage] = _binding_aggregate(subset_rows, stage)
        binding[subset]["dq_private"] = _binding_aggregate(
            subset_rows, "dq_private"
        )

    formal_subsets = (data.get("formal_metrics") or {}).get("subsets") or {}
    mr_subsets = {
        subset: {
            "num_records": formal_subsets.get(subset, {}).get("num_records", 0),
            "d4": _formal_brief(formal_subsets.get(subset, {}).get("d4")),
        }
        for subset in MR_SUBSETS
        if subset in formal_subsets
    }
    return {
        "records": len(rows),
        "multi": len(multi),
        "clean_multi": len(clean),
        "d4_mr_full_map": d4_brief.get("MR-full-mAP"),
        "d4_mr": d4_brief,
        "mr_subsets": mr_subsets,
        "binding": binding,
        "multi_coverage@5_50": _mean(
            multi, lambda row: row["coverage"]["d4"]["coverage@5_50"]
        ),
        # Backwards-compatible headline fields from the original compact file.
        "multi_d1_aec": _mean(multi, lambda row: row["d1_final"]["aec"]),
        "multi_d4_aec": _mean(multi, lambda row: row["d4_final"]["aec"]),
        "multi_d1_ecr": _mean(multi, lambda row: row["d1_final"]["ecr"]),
        "multi_d4_ecr": _mean(multi, lambda row: row["d4_final"]["ecr"]),
        "clean_d4_aec": _mean(clean, lambda row: row["d4_final"]["aec"]),
        "clean_d4_ecr": _mean(clean, lambda row: row["d4_final"]["ecr"]),
    }


def summarize(root: str | Path) -> dict:
    root = Path(root)
    result = {"root": str(root.resolve()), "modes": {}}
    for mode in MODES:
        path = root / mode / "analysis.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result["modes"][mode] = _mode_summary(data)
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
