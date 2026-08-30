"""Evaluate Sim-DETR checkpoints with Layer-Consistent Binding diagnostics.

Computes the core metrics outlined in the LCB plan:
1. Layerwise AEC (D1, D2, D3, D4 Attention-Evidence Consistency)
2. Layerwise ECR (D1, D2, D3, D4 Evidence Collision Rate)
3. D1 -> D4 Ownership Persistence and mass retention
4. Headline MR mAP and multi-occurrence MR mAP
5. Fixed-K coverage and duplicate attribution rate (DAR)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_occurrence_lab.analyze_checkpoints import (
    _collate_and_prepare,
    _decode_spans,
    _filter_submission_qids,
    _formal_metrics,
    _formal_subset_rows,
    _indices_for_sample,
    _make_submission_line,
    _match_indices,
    _normalize_video_attention,
    _postprocess_submissions,
    _saliency,
)
from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from causal_occurrence_lab.common import (
    build_dataset,
    checkpoint_manifest,
    default_split_path,
    layer_outputs,
    save_json,
)
from causal_occurrence_lab.metrics import (
    binding_metrics_from_target_spans,
    fixed_k_metrics,
    is_clean_multi,
    occurrence_bucket,
)
from Layeerconsistentbinding.metrics import (
    compute_layerwise_ownership_summary,
    compute_multi_occurrence_stratified_metrics,
)


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    data_path: Optional[str] = None,
    split: str = "test",
    device: Optional[str] = None,
    output_dir: str,
    num_workers: int = 0,
    eval_bsz: int = 32,
    max_batches: Optional[int] = None,
    clean_iou: float = 0.1,
) -> Dict[str, Any]:
    """Run full LCB evaluation on a saved checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opt = argparse.Namespace(**checkpoint_data["opt"])
    opt.device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if opt.device.type == "cuda":
        torch.cuda.set_device(0)

    # Build model and load weights
    from sim_detr.model import build_model

    model, criterion = build_model(opt)
    model.load_state_dict(checkpoint_data["model"], strict=True)
    model.to(opt.device)
    criterion.to(opt.device)
    model.eval()

    if data_path is None:
        data_path = str(default_split_path(split))

    opt.eval_bsz = int(eval_bsz)
    opt.pin_memory = bool(opt.device.type == "cuda")
    dataset = build_dataset(opt, data_path, load_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=opt.eval_bsz,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        collate_fn=list,
    )

    capture = NativeCrossAttentionCapture(model).install()
    layer_count = len(model.transformer.decoder.layers)
    raw_submissions: List[List[Dict[str, Any]]] = [[] for _ in range(layer_count)]
    records: List[Dict[str, Any]] = []

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(loader, desc=f"Evaluating LCB ({split})")):
                if max_batches is not None and batch_index >= max_batches:
                    break
                metas, model_inputs, targets = _collate_and_prepare(batch, opt)
                capture.clear()
                outputs = model(**model_inputs)
                native_attention = capture.get()
                outputs_by_layer = layer_outputs(outputs)

                own_indices = [_match_indices(criterion, layer, targets) for layer in outputs_by_layer]
                final_indices = own_indices[-1]

                for sample_index, meta in enumerate(metas):
                    gt_windows = list(meta.get("relevant_windows", []))[: int(opt.max_windows)]
                    valid_length = int(model_inputs["src_vid_mask"][sample_index].sum().item())
                    record: Dict[str, Any] = {
                        "qid": meta.get("qid"),
                        "vid": meta.get("vid"),
                        "query": meta.get("query"),
                        "duration": float(meta.get("duration", 0.0)),
                        "num_gt": len(gt_windows),
                        "gt_windows": gt_windows,
                        "clean_multi": bool(is_clean_multi(gt_windows, clean_iou)),
                        "occurrence_bucket": occurrence_bucket(gt_windows, clean_iou),
                        "valid_video_length": valid_length,
                        "coverage": {},
                    }
                    target_spans = targets["span_labels"][sample_index]["spans"]

                    for layer_id, layer_output in enumerate(outputs_by_layer):
                        name = f"d{layer_id + 1}"
                        (
                            pred_windows,
                            query_order,
                            query_spans,
                            query_scores,
                            query_foreground,
                            query_iou,
                        ) = _decode_spans(layer_output, meta, opt, sample_index)
                        saliency = _saliency(outputs, valid_length, sample_index)
                        raw_submissions[layer_id].append(
                            _make_submission_line(meta, pred_windows, saliency)
                        )
                        record[f"{name}_raw_pred_windows"] = pred_windows
                        record[f"{name}_query_order"] = query_order
                        record[f"{name}_scores"] = [row[2] for row in pred_windows]

                        own_q, own_g = _indices_for_sample(own_indices[layer_id], sample_index)
                        final_q, final_g = _indices_for_sample(final_indices, sample_index)

                        if native_attention[layer_id] is not None:
                            native = native_attention[layer_id][sample_index, :, :valid_length]
                            native = _normalize_video_attention(native)
                            record[f"{name}_own"] = binding_metrics_from_target_spans(
                                native, target_spans, own_q, own_g,
                                valid_length=valid_length,
                                span_loss_type=opt.span_loss_type,
                            )
                            record[f"{name}_final"] = binding_metrics_from_target_spans(
                                native, target_spans, final_q, final_g,
                                valid_length=valid_length,
                                span_loss_type=opt.span_loss_type,
                            )
                        else:
                            record[f"{name}_own"] = None
                            record[f"{name}_final"] = None

                        record["coverage"][name] = fixed_k_metrics(pred_windows, gt_windows)

                    records.append(record)
    finally:
        capture.remove()

    processed_submissions = _postprocess_submissions(raw_submissions)
    by_qid = [
        {str(line["qid"]): line for line in submission}
        for submission in processed_submissions
    ]
    for record in records:
        qid = str(record["qid"])
        for layer_id in range(layer_count):
            name = f"d{layer_id + 1}"
            line = by_qid[layer_id].get(qid)
            if line is not None:
                record[f"{name}_pred_windows"] = line["pred_relevant_windows"]
                record["coverage"][name] = fixed_k_metrics(
                    [row[:2] for row in line["pred_relevant_windows"]],
                    record["gt_windows"],
                )

    # Compute formal retrieval metrics
    formal = {
        f"d{layer_id + 1}": _formal_metrics(submission, dataset.data)
        for layer_id, submission in enumerate(processed_submissions)
    }
    subset_names = ("single", "multi", "clean_multi", "two", "three_or_more")
    final_submission = processed_submissions[-1]
    formal["subsets"] = {}
    for subset in subset_names:
        subset_rows = _formal_subset_rows(dataset.data, opt.max_windows, subset)
        formal["subsets"][subset] = {
            "num_records": len(subset_rows),
            "d4": _formal_metrics(
                _filter_submission_qids(final_submission, subset_rows),
                subset_rows,
                full_only=True,
            ),
        }

    # Aggregate layerwise ownership and persistence metrics
    ownership_summary = compute_layerwise_ownership_summary(records, decoder_layers=layer_count)
    stratified_metrics = compute_multi_occurrence_stratified_metrics(records)

    result = {
        "manifest": checkpoint_manifest(checkpoint_path, checkpoint_data),
        "data": str(Path(data_path).expanduser().resolve()),
        "split": split,
        "clean_iou": float(clean_iou),
        "num_records": len(records),
        "decoder_layers": layer_count,
        "ownership_summary": ownership_summary,
        "stratified_metrics": stratified_metrics,
        "formal_metrics": formal,
        "records": records,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "lcb_analysis.json", result)
    for layer_id, submission in enumerate(processed_submissions):
        save_json(out_dir / f"d{layer_id + 1}_submission.json", submission)

    # Print clean formatted summary table
    print("\n" + "=" * 70)
    print(f" Sim-DETR Layer-Consistent Binding Evaluation: {split.upper()} set")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluated samples: {len(records)}")
    print("-" * 70)
    print(" [Layerwise Ownership Metrics (Final Hungarian Assignment)]")
    print(f"{'Layer':<8} | {'AEC (macro)':<12} | {'ECR (macro)':<12} | {'Mass':<10} | {'Margin':<10}")
    print("-" * 62)
    for l_name, metrics in ownership_summary.get("layers", {}).items():
        print(
            f"{l_name.upper():<8} | "
            f"{metrics['aec']:<12.4f} | "
            f"{metrics['ecr']:<12.4f} | "
            f"{metrics['own_mass']:<10.4f} | "
            f"{metrics['binding_margin']:<10.4f}"
        )
    print("-" * 70)
    print(" [Ownership Persistence from D1 Anchor]")
    persistence = ownership_summary.get("persistence", {})
    if persistence:
        print(f"D1 Correct Initial Rate:  {persistence.get('d1_correct_rate', 0.0):.4f}")
        print(f"D1 -> D2 Persistence:     {persistence.get('d1_to_d2_persistence', 0.0):.4f}")
        print(f"D1 -> D3 Persistence:     {persistence.get('d1_to_d3_persistence', 0.0):.4f}")
        print(f"D1 -> D4 Persistence:     {persistence.get('d1_to_d4_persistence', 0.0):.4f}")
        print(f"D1 -> D4 Mass Drop Rate:  {persistence.get('d1_to_d4_washout_drop_rate', 0.0):.4f}")
    print("-" * 70)
    print(" [Formal D4 Retrieval Performance]")
    d4_brief = formal.get("d4", {}).get("brief", {})
    for k, v in d4_brief.items():
        print(f"  {k:<20}: {v:.4f}")
    print("=" * 70 + "\n")

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--data", default=None, help="Path to jsonl evaluation data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True, help="Output directory for analysis results")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-bsz", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--clean-iou", type=float, default=0.1)
    args = parser.parse_args(argv)

    evaluate_checkpoint(
        args.checkpoint,
        data_path=args.data,
        split=args.split,
        device=args.device,
        output_dir=args.output,
        num_workers=args.num_workers,
        eval_bsz=args.eval_bsz,
        max_batches=args.max_batches,
        clean_iou=args.clean_iou,
    )


if __name__ == "__main__":
    main()
