"""Create a compact report for the fixed-seed causal training variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json
from causal_occurrence_lab.summarize_phase1 import _mode_summary


VARIANTS = (
    "full_repro_seed2017",
    "no_bind_seed2017",
    "supervision_only_seed2017",
    "union_bind_seed2017",
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(root: str | Path) -> dict:
    root = Path(root)
    result = {
        "root": str(root.resolve()),
        "seed_policy": "single fixed seed 2017; no multi-seed runs",
        "variants": {},
    }
    for variant in VARIANTS:
        path = root / variant / "analysis.json"
        if not path.is_file():
            continue
        data = _read_json(path)
        compact = _mode_summary(data)
        multi = compact["binding"]["multi"]["d4_final"]
        clean = compact["binding"]["clean_multi"]["d4_final"]
        result["variants"][variant] = {
            "checkpoint": data.get("manifest", {}).get("checkpoint"),
            "checkpoint_epoch": data.get("manifest", {}).get("epoch"),
            "checkpoint_metric": data.get("manifest", {}).get("metric"),
            "records": compact["records"],
            "multi": compact["multi"],
            "clean_multi": compact["clean_multi"],
            "d4_mr": compact["d4_mr"],
            "mr_subsets": compact["mr_subsets"],
            "multi_d4_binding": multi,
            "clean_multi_d4_binding": clean,
        }
        config_path = root.parent / "causal_training" / variant / "variant.json"
        if config_path.is_file():
            result["variants"][variant]["variant_config"] = _read_json(config_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="evaluated_variants root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(args.root)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
