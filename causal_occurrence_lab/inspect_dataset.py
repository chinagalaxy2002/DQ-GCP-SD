"""QVHighlights multi-occurrence census for the Sim-DETR experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow ``python causal_occurrence_lab/inspect_dataset.py`` from the
# repository root as well as ``python -m causal_occurrence_lab...``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import default_split_path, save_json
from causal_occurrence_lab.metrics import is_clean_multi, pairwise_iou


def load_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def census(rows, *, max_windows: int = 10, clean_iou: float = 0.1):
    counts = Counter()
    overlap_hist = Counter()
    num_gt_values = []
    durations = []
    for row in rows:
        windows = list(row.get("relevant_windows", []))[:max_windows]
        n = len(windows)
        num_gt_values.append(n)
        if row.get("duration") is not None:
            durations.append(float(row["duration"]))
        counts["total"] += 1
        counts["single"] += n <= 1
        counts["multi"] += n >= 2
        counts["two"] += n == 2
        counts["three_or_more"] += n >= 3
        clean = is_clean_multi(windows, clean_iou)
        counts["clean_multi"] += clean
        counts["clean_two"] += clean and n == 2
        counts["clean_three_or_more"] += clean and n >= 3
        for value in pairwise_iou(windows):
            if value < 0.1:
                key = "[0,0.1)"
            elif value < 0.3:
                key = "[0.1,0.3)"
            elif value < 0.5:
                key = "[0.3,0.5)"
            elif value < 0.7:
                key = "[0.5,0.7)"
            else:
                key = "[0.7,1]"
            overlap_hist[key] += 1
    return {
        "counts": dict(counts),
        "max_windows": int(max_windows),
        "clean_iou_threshold": float(clean_iou),
        "mean_num_gt": (sum(num_gt_values) / len(num_gt_values)) if num_gt_values else 0.0,
        "mean_duration": (sum(durations) / len(durations)) if durations else 0.0,
        "gt_overlap_histogram": dict(sorted(overlap_hist.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="QVHighlights JSONL")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--clean-iou", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data_path = Path(args.data) if args.data else default_split_path(args.split)
    result = census(load_jsonl(data_path), max_windows=args.max_windows, clean_iou=args.clean_iou)
    result["data"] = str(data_path.resolve())
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
