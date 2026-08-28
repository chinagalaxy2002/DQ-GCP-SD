"""Compare query-level perturbations between two saved analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json


def _records(path: str | Path) -> dict[str, Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(row["qid"]): row for row in data.get("records", [])}


def _mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "n": int(values.size),
    }


def _top5_jaccard(left_order, right_order) -> float:
    left = set(int(value) for value in left_order[:5])
    right = set(int(value) for value in right_order[:5])
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def compare(left_path: str, right_path: str) -> dict[str, Any]:
    left, right = _records(left_path), _records(right_path)
    qids = sorted(set(left).intersection(right))
    score_delta = []
    foreground_delta = []
    iou_delta = []
    span_delta = []
    ranking_jaccard = []
    relative_update = []
    for qid in qids:
        lrow, rrow = left[qid], right[qid]
        lscore = np.asarray(lrow.get("d4_query_scores", []), dtype=float)
        rscore = np.asarray(rrow.get("d4_query_scores", []), dtype=float)
        lfg = np.asarray(lrow.get("d4_query_fg_probs", []), dtype=float)
        rfg = np.asarray(rrow.get("d4_query_fg_probs", []), dtype=float)
        liou = np.asarray(lrow.get("d4_query_iou_scores", []), dtype=float)
        riou = np.asarray(rrow.get("d4_query_iou_scores", []), dtype=float)
        lspan = np.asarray(lrow.get("d4_query_spans_seconds", []), dtype=float)
        rspan = np.asarray(rrow.get("d4_query_spans_seconds", []), dtype=float)
        if lscore.shape == rscore.shape and lscore.size:
            score_delta.extend(np.abs(lscore - rscore).tolist())
        if lfg.shape == rfg.shape and lfg.size:
            foreground_delta.extend(np.abs(lfg - rfg).tolist())
        if liou.shape == riou.shape and liou.size:
            iou_delta.extend(np.abs(liou - riou).tolist())
        if lspan.shape == rspan.shape and lspan.ndim == 2 and lspan.size:
            span_delta.extend(np.abs(lspan - rspan).max(axis=-1).tolist())
        if lrow.get("d4_query_order") and rrow.get("d4_query_order"):
            ranking_jaccard.append(
                _top5_jaccard(lrow["d4_query_order"], rrow["d4_query_order"])
            )
        if lrow.get("relative_update_mean") is not None:
            relative_update.append(float(lrow["relative_update_mean"]))
    return {
        "left": str(Path(left_path).resolve()),
        "right": str(Path(right_path).resolve()),
        "num_paired_qids": len(qids),
        "active_vs_zero": {
            "mean_abs_ranking_score_delta": _mean(score_delta),
            "mean_abs_foreground_probability_delta": _mean(foreground_delta),
            "mean_abs_iou_score_delta": _mean(iou_delta),
            "mean_abs_span_delta_seconds": _mean(span_delta),
            "top5_query_ranking_jaccard": _mean(ranking_jaccard),
            "relative_update_mean": _mean(relative_update),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="active/DQ analysis")
    parser.add_argument("--right", required=True, help="beta-zero/stripped analysis")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare(args.left, args.right)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
