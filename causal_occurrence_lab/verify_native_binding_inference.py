"""Verify the NativeBind checkpoint and training-only capture contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.analyze_checkpoints import _collate_and_prepare
from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from causal_occurrence_lab.common import (
    build_dataset,
    build_model_for_mode,
    checkpoint_state,
    load_checkpoint,
    normalize_state_dict,
    save_json,
)


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_tensors(item, name))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_tensors(item, f"{prefix}[{index}]"))
        return result
    return {}


def _state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = load_checkpoint(path)
    return normalize_state_dict(checkpoint_state(checkpoint))


def verify(args: argparse.Namespace) -> dict:
    native_state = _state(args.checkpoint)
    result = {
        "checkpoint": Path(args.checkpoint).name,
        "state_keys": len(native_state),
        "state_numel": sum(value.numel() for value in native_state.values()),
        "query_cgp_keys": sum(key.startswith("query_cgp.") for key in native_state),
    }

    if args.baseline_checkpoint:
        baseline_state = _state(args.baseline_checkpoint)
        common = baseline_state.keys() & native_state.keys()
        shape_mismatches = {
            key: [list(baseline_state[key].shape), list(native_state[key].shape)]
            for key in common
            if baseline_state[key].shape != native_state[key].shape
        }
        result["baseline_contract"] = {
            "checkpoint": Path(args.baseline_checkpoint).name,
            "same_keyset": baseline_state.keys() == native_state.keys(),
            "shape_mismatches": shape_mismatches,
            "state_numel_delta": sum(value.numel() for value in native_state.values())
            - sum(value.numel() for value in baseline_state.values()),
        }

    if args.data:
        model, _, opt, _ = build_model_for_mode(
            args.checkpoint,
            "baseline",
            device=args.device,
            option_overrides={"num_workers": 0, "eval_bsz": args.batch_size},
        )
        opt.eval_path = str(Path(args.data).expanduser().resolve())
        opt.num_workers = 0
        opt.eval_bsz = int(args.batch_size)
        opt.pin_memory = opt.device.type == "cuda"
        model.eval()
        dataset = build_dataset(opt, args.data, load_labels=True)
        batch = next(iter(DataLoader(
            dataset,
            batch_size=opt.eval_bsz,
            shuffle=False,
            num_workers=0,
            collate_fn=list,
        )))
        _, model_inputs, _ = _collate_and_prepare(batch, opt)
        with torch.no_grad():
            plain = model(**model_inputs)
        capture = NativeCrossAttentionCapture(model).install()
        try:
            with torch.no_grad():
                wrapped = model(**model_inputs)
            attentions = capture.get()
        finally:
            capture.remove()
        plain_tensors = _flatten_tensors(plain)
        wrapped_tensors = _flatten_tensors(wrapped)
        if plain_tensors.keys() != wrapped_tensors.keys():
            raise RuntimeError("Capture changed the model output structure")
        max_difference = max(
            (plain_tensors[key] - wrapped_tensors[key]).abs().max().item()
            for key in plain_tensors
        )
        result["capture_equivalence"] = {
            "num_compared_output_tensors": len(plain_tensors),
            "captured_attention_shapes": [list(value.shape) for value in attentions],
            "max_abs_output_difference": max_difference,
            "exact": max_difference == 0.0,
        }

    if result["query_cgp_keys"] != 0:
        raise RuntimeError("NativeBind checkpoint unexpectedly contains query_cgp parameters")
    contract = result.get("baseline_contract")
    if contract and (
        not contract["same_keyset"]
        or contract["shape_mismatches"]
        or contract["state_numel_delta"] != 0
    ):
        raise RuntimeError("NativeBind and baseline checkpoint architectures differ")
    equivalence = result.get("capture_equivalence")
    if equivalence and not equivalence["exact"]:
        raise RuntimeError("Attention capture changed inference outputs")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = verify(args)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
