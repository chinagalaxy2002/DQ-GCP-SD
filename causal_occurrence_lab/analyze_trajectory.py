"""Run the same mechanism analysis over saved training snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.analyze_checkpoints import analyze
from causal_occurrence_lab.common import default_split_path, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--mode", choices=["baseline", "dq_active", "dq_beta_zero", "dq_stripped"], required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-bsz", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-formal-metrics", action="store_true")
    args = parser.parse_args(argv)
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files under {args.checkpoint_dir}")
    data_path = args.data or str(default_split_path(args.split))
    manifest = []
    for checkpoint in checkpoints:
        tag = checkpoint.stem
        output = Path(args.output_dir) / tag
        result = analyze(
            str(checkpoint), args.mode, data_path=data_path, device=args.device,
            output_dir=str(output), num_workers=args.num_workers, eval_bsz=args.eval_bsz,
            max_batches=args.max_batches, skip_formal_metrics=args.skip_formal_metrics,
        )
        manifest.append({
            "checkpoint": str(checkpoint.resolve()),
            "output": str(output.resolve()),
            "epoch": result["manifest"].get("epoch"),
            "formal_metrics": result.get("formal_metrics"),
        })
    save_json(Path(args.output_dir) / "trajectory_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
