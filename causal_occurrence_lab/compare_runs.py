"""Paired qid-level comparisons for saved occurrence analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json


def load_analysis(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(record: Mapping[str, Any], name: str) -> float:
    """Resolve compact names such as ``d1_final.aec`` or ``d4.coverage@5_50``."""

    if name.startswith("coverage@") or name.startswith("duplicate_rate@") or name.startswith("dar@"):
        return float(record["coverage"]["d4"].get(name, 0.0))
    if name.startswith("d") and "." in name:
        stage, field = name.split(".", 1)
        if stage.endswith("_final") or stage.endswith("_own"):
            return float(record[stage][field])
        return float(record[f"{stage}_final"][field])
    if name.startswith("d") and name.endswith("_aec"):
        stage = name[:-4]
        return float(record[f"{stage}_final"]["aec"])
    if name.startswith("d") and name.endswith("_ecr"):
        stage = name[:-4]
        return float(record[f"{stage}_final"]["ecr"])
    if name in record:
        value = record[name]
        return float(value) if np.isscalar(value) else float(np.mean(value))
    raise KeyError(f"Unknown metric {name!r}")


def _select_records(data: Mapping[str, Any], subset: str) -> Dict[str, Mapping[str, Any]]:
    rows = data.get("records", [])
    if subset == "all":
        selected = rows
    elif subset == "multi":
        selected = [row for row in rows if row.get("num_gt", 0) >= 2]
    elif subset == "clean_multi":
        selected = [row for row in rows if row.get("num_gt", 0) >= 2 and row.get("clean_multi", False)]
    elif subset == "two":
        selected = [row for row in rows if row.get("num_gt", 0) == 2]
    elif subset == "three_or_more":
        selected = [row for row in rows if row.get("num_gt", 0) >= 3]
    else:
        raise ValueError(f"Unknown subset {subset}")
    return {str(row["qid"]): row for row in selected}


def paired_bootstrap(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    samples: int = 10000,
    seed: int = 2023,
) -> Dict[str, Any]:
    qids = sorted(set(left).intersection(right))
    if not qids:
        return {"metric": metric, "n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    deltas = np.asarray([
        _metric(right[qid], metric) - _metric(left[qid], metric)
        for qid in qids
    ], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(deltas), size=(int(samples), len(deltas)))
    distribution = deltas[sampled].mean(axis=1)
    return {
        "metric": metric,
        "n": len(qids),
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "positive_fraction": float(np.mean(deltas > 0)),
    }


def activity_matched(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    metric: str,
    *,
    threshold: str = "0.5",
    max_count_delta: int = 1,
    samples: int = 10000,
    seed: int = 2023,
) -> Dict[str, Any]:
    common = sorted(set(left).intersection(right))
    qids = [
        qid for qid in common
        if abs(
            int(left[qid].get("active_query_count", {}).get(threshold, 0))
            - int(right[qid].get("active_query_count", {}).get(threshold, 0))
        ) <= max_count_delta
    ]
    return paired_bootstrap(
        {qid: left[qid] for qid in qids},
        {qid: right[qid] for qid in qids},
        metric,
        samples=samples,
        seed=seed,
    ) | {"activity_threshold": threshold, "max_count_delta": max_count_delta}


def compare(
    left_path: str,
    right_path: str,
    *,
    metrics: Sequence[str],
    subset: str = "multi",
    samples: int = 10000,
    seed: int = 2023,
    activity_match: bool = False,
) -> Dict[str, Any]:
    left_data, right_data = load_analysis(left_path), load_analysis(right_path)
    left, right = _select_records(left_data, subset), _select_records(right_data, subset)
    result = {
        "left": str(Path(left_path).resolve()),
        "right": str(Path(right_path).resolve()),
        "subset": subset,
        "bootstrap_samples": int(samples),
        "comparisons": [
            paired_bootstrap(left, right, metric, samples=samples, seed=seed + i)
            for i, metric in enumerate(metrics)
        ],
    }
    if activity_match:
        result["activity_matched"] = [
            activity_matched(left, right, metric, samples=samples, seed=seed + 100 + i)
            for i, metric in enumerate(metrics)
        ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="analysis.json for baseline/control")
    parser.add_argument("--right", required=True, help="analysis.json for comparison")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["coverage@5_50", "d1_final.aec", "d4_final.aec", "d1_final.ecr", "d4_final.ecr"],
    )
    parser.add_argument("--subset", choices=["all", "multi", "clean_multi", "two", "three_or_more"], default="multi")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--activity-matched", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare(
        args.left, args.right, metrics=args.metrics, subset=args.subset,
        samples=args.samples, seed=args.seed, activity_match=args.activity_matched,
    )
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

