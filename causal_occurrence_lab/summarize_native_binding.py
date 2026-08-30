"""Summarize the completed Baseline versus NativeBind experiment.

Only two completed, validation-selected runs belong in this report.  The
script reads the full checkpoint analyses and writes a compact JSON artifact
containing validation, test, occurrence-subset, and binding-mechanism metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json
from causal_occurrence_lab.summarize_phase1 import _mode_summary


VAL_KEYS = (
    "MR-full-R1@0.5",
    "MR-full-R1@0.7",
    "MR-full-mAP@0.5",
    "MR-full-mAP@0.75",
    "MR-full-mAP",
    "HL-min-Fair-mAP",
    "HL-min-Good-mAP",
    "HL-min-VeryGood-mAP",
)

TEST_HL_KEYS = (
    "HL-min-Fair-mAP",
    "HL-min-Good-mAP",
    "HL-min-VeryGood-mAP",
)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    return {
        key: round(float(right[key]) - float(left[key]), 6)
        for key in left
        if key in right
        and isinstance(left[key], (int, float))
        and isinstance(right[key], (int, float))
    }


def _validation(path: str | Path) -> dict:
    brief = _load(path)["brief"]
    return {key: brief[key] for key in VAL_KEYS}


def _mechanism(summary: Mapping[str, Any]) -> dict:
    return {
        "multi_coverage@5_50": summary["multi_coverage@5_50"],
        "d1_final_aec": summary["multi_d1_aec"],
        "d1_final_ecr": summary["multi_d1_ecr"],
        "d4_final_aec": summary["multi_d4_aec"],
        "d4_final_ecr": summary["multi_d4_ecr"],
    }


def _test_highlight(analysis: Mapping[str, Any]) -> dict:
    brief = analysis["formal_metrics"]["d4"]["brief"]
    return {key: brief[key] for key in TEST_HL_KEYS}


def summarize(args: argparse.Namespace) -> dict:
    baseline_analysis = _load(args.baseline_analysis)
    native_analysis = _load(args.native_analysis)
    baseline = _mode_summary(baseline_analysis)
    native = _mode_summary(native_analysis)
    baseline_val = _validation(args.baseline_val_metrics)
    native_val = _validation(args.native_val_metrics)
    baseline_mechanism = _mechanism(baseline)
    native_mechanism = _mechanism(native)
    baseline_highlight = _test_highlight(baseline_analysis)
    native_highlight = _test_highlight(native_analysis)

    baseline_subsets = baseline["mr_subsets"]
    native_subsets = native["mr_subsets"]
    subset_delta = {
        name: _difference(baseline_subsets[name]["d4"], native_subsets[name]["d4"])
        for name in baseline_subsets
        if name in native_subsets
    }

    return {
        "experiment": "Vanilla Sim-DETR vs Vanilla Sim-DETR + Native Hungarian Binding",
        "selection": "best checkpoint by validation MR-full-mAP",
        "included_runs": {
            "baseline": {
                "completed": True,
                "epoch": baseline_analysis["manifest"].get("epoch"),
                "sha256": _sha256(args.baseline_checkpoint),
            },
            "native_bind_lambda_0p5": {
                "completed": True,
                "epoch": native_analysis["manifest"].get("epoch"),
                "lambda_native_bind": 0.5,
                "sha256": _sha256(args.native_checkpoint),
            },
        },
        "checkpoint_contract": {
            "architecture": "Vanilla Sim-DETR",
            "inference_mode": "baseline",
            "added_trainable_parameters": 0,
            "native_bind_is_training_only": True,
        },
        "validation": {
            "baseline": baseline_val,
            "native_bind_lambda_0p5": native_val,
            "delta_native_minus_baseline": _difference(baseline_val, native_val),
        },
        "test": {
            "num_records": baseline["records"],
            "d4": {
                "baseline": baseline["d4_mr"],
                "native_bind_lambda_0p5": native["d4_mr"],
                "delta_native_minus_baseline": _difference(
                    baseline["d4_mr"], native["d4_mr"]
                ),
            },
            "highlight": {
                "baseline": baseline_highlight,
                "native_bind_lambda_0p5": native_highlight,
                "delta_native_minus_baseline": _difference(
                    baseline_highlight, native_highlight
                ),
            },
            "occurrence_subsets": {
                "baseline": baseline_subsets,
                "native_bind_lambda_0p5": native_subsets,
                "delta_native_minus_baseline": subset_delta,
            },
            "mechanism_on_511_multi_occurrence_queries": {
                "assignment": "final D4 Hungarian assignment traced to native attention",
                "baseline": baseline_mechanism,
                "native_bind_lambda_0p5": native_mechanism,
                "delta_native_minus_baseline": _difference(
                    baseline_mechanism, native_mechanism
                ),
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-analysis", required=True)
    parser.add_argument("--native-analysis", required=True)
    parser.add_argument("--baseline-val-metrics", required=True)
    parser.add_argument("--native-val-metrics", required=True)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--native-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(args)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
