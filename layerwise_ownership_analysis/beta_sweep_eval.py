"""Evaluate one trained DQ-CGP checkpoint with inference-time beta overrides.

This file is intentionally isolated from the original Sim-DETR sources.  It
reuses the repository's strict DQ-CGP loader, dataset, inference, and official
evaluation code; only ``model.query_cgp.beta`` is changed between evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from sim_detr.dq_cgp.inference_dq_cgp import (
    build_eval_dataset,
    parse_inference_options,
    setup_model_dq_cgp_inference,
)
from sim_detr.inference import eval_epoch


def beta_tag(beta: float) -> str:
    return f"beta_{beta:.3f}".replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-path")
    parser.add_argument("--eval-split-name", default="val")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inference_argv = [
        "--resume", str(Path(args.checkpoint).resolve()),
        "--eval_split_name", args.eval_split_name,
        "--eval_results_dir", str(output_root / "initialization"),
        "--gpu_id", str(args.gpu_id),
        "--device", str(args.device),
    ]
    if args.eval_path:
        inference_argv.extend(["--eval_path", str(Path(args.eval_path).resolve())])

    opt = parse_inference_options(inference_argv)
    cudnn.benchmark = True
    cudnn.deterministic = False
    dataset = build_eval_dataset(opt)
    model, criterion = setup_model_dq_cgp_inference(opt)

    adapter = getattr(model, "query_cgp", None)
    if adapter is None or not hasattr(adapter, "set_beta"):
        raise RuntimeError("Loaded model has no settable model.query_cgp adapter")

    records = []
    for beta in args.betas:
        adapter.set_beta(beta)
        run_dir = output_root / beta_tag(beta)
        run_dir.mkdir(parents=True, exist_ok=True)
        opt.results_dir = str(run_dir)
        opt.eval_results_dir = str(run_dir)
        submission_name = f"hl_{opt.eval_split_name}_submission.jsonl"

        with torch.no_grad():
            metrics, metrics_nms, _, paths = eval_epoch(
                model, dataset, opt, submission_name, criterion=criterion
            )

        records.append(
            {
                "beta": beta,
                "metrics": metrics,
                "metrics_nms": metrics_nms,
                "files": [str(Path(path).resolve()) for path in paths],
            }
        )
        print(json.dumps({"beta": beta, "brief": metrics["brief"]}, ensure_ascii=False))

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "eval_split_name": opt.eval_split_name,
        "eval_path": opt.eval_path,
        "trained_beta": float(opt.query_cgp_beta),
        "runs": records,
    }
    summary_path = output_root / "beta_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
