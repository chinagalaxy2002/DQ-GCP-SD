"""Verify DQ beta-zero and DQ-stripped Sim-DETR outputs are identical."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.analyze_checkpoints import _collate_and_prepare
from causal_occurrence_lab.common import build_dataset, build_model_for_mode, default_split_path, save_json


def _max_delta(left, right):
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def verify(
    checkpoint: str,
    *,
    data_path: str,
    device: str | None,
    output: str,
    num_samples: int = 100,
    batch_size: int = 8,
    seed: int = 2023,
):
    beta_model, _, beta_opt, _ = build_model_for_mode(checkpoint, "dq_beta_zero", device=device)
    stripped_model, _, stripped_opt, _ = build_model_for_mode(checkpoint, "dq_stripped", device=device)
    beta_model.eval()
    stripped_model.eval()
    dataset = build_dataset(beta_opt, data_path, load_labels=True)
    rng = np.random.default_rng(seed)
    n = min(int(num_samples), len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda rows: rows)

    max_deltas = {}
    samples_seen = 0
    with torch.no_grad():
        for batch in loader:
            _, beta_inputs, _ = _collate_and_prepare(batch, beta_opt)
            _, stripped_inputs, _ = _collate_and_prepare(batch, stripped_opt)
            beta_out = beta_model(**beta_inputs)
            stripped_out = stripped_model(**stripped_inputs)
            for key in ("pred_logits", "pred_spans", "iou_scores", "saliency_scores"):
                max_deltas[key] = max(max_deltas.get(key, 0.0), _max_delta(beta_out[key], stripped_out[key]))
            beta_aux = beta_out.get("aux_outputs", [])
            stripped_aux = stripped_out.get("aux_outputs", [])
            if len(beta_aux) != len(stripped_aux):
                raise AssertionError("Auxiliary output counts differ")
            for layer_id, (left, right) in enumerate(zip(beta_aux, stripped_aux), start=1):
                for key in ("pred_logits", "pred_spans", "pred_masks", "iou_scores"):
                    name = f"d{layer_id}_{key}"
                    max_deltas[name] = max(max_deltas.get(name, 0.0), _max_delta(left[key], right[key]))
            samples_seen += len(batch)

    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "data": str(Path(data_path).resolve()),
        "num_samples": samples_seen,
        "max_abs_delta": max_deltas,
        "bitwise_equal": all(value == 0.0 for value in max_deltas.values()),
        "tolerance": 1e-6,
        "within_tolerance": all(value < 1e-6 for value in max_deltas.values()),
    }
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args(argv)
    verify(
        args.checkpoint,
        data_path=args.data or str(default_split_path(args.split)),
        device=args.device,
        output=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
