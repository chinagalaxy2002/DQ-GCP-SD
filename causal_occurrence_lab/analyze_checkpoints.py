"""Analyze Sim-DETR checkpoints for occurrence-specific evidence binding."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any, Dict, List, Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from causal_occurrence_lab.common import (
    build_dataset,
    build_model_for_mode,
    checkpoint_manifest,
    default_split_path,
    layer_outputs,
    save_json,
)
from causal_occurrence_lab.controls import install_injection_control
from causal_occurrence_lab.metrics import (
    binding_metrics,
    fixed_k_metrics,
    is_clean_multi,
    occurrence_bucket,
    route_metrics,
)


def _collate_and_prepare(batch, opt):
    """Use the native collator and add matcher-only mask labels."""

    if getattr(opt, "a_feat_dir", None) is None:
        from sim_detr.start_end_dataset import start_end_collate, prepare_batch_inputs

        metas, batched = start_end_collate(batch)
        model_inputs, targets = prepare_batch_inputs(
            batched, opt.device, non_blocking=getattr(opt, "pin_memory", False)
        )
    else:
        from sim_detr.start_end_dataset_audio import (
            start_end_collate_audio,
            prepare_batch_inputs_audio,
        )

        metas, batched = start_end_collate_audio(batch)
        model_inputs, targets = prepare_batch_inputs_audio(
            batched, opt.device, non_blocking=getattr(opt, "pin_memory", False)
        )

    if targets is None:
        targets = {}
    video_length = model_inputs["src_vid"].shape[1]
    mask_labels = []
    for meta in metas:
        windows = list(meta.get("relevant_windows", []))[: int(opt.max_windows)]
        masks = torch.zeros(len(windows), video_length, device=opt.device)
        for window_id, window in enumerate(windows):
            start = max(0, int(float(window[0]) // int(opt.clip_length)))
            end = min(video_length, int(float(window[1]) // int(opt.clip_length)))
            if end > start:
                masks[window_id, start:end] = 1.0
        mask_labels.append(masks)
    targets["mask_labels"] = mask_labels
    return metas, model_inputs, targets


def _decode_spans(
    output: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    opt,
    batch_index: int,
):
    logits = output["pred_logits"][batch_index]
    foreground_probability = F.softmax(logits, dim=-1)[..., 0]
    iou_score = torch.ones_like(foreground_probability)
    if "iou_scores" in output:
        iou_score = output["iou_scores"][batch_index, ..., 0].sigmoid()
    score = foreground_probability * iou_score

    if opt.span_loss_type == "l1":
        from sim_detr.span_utils import span_cxw_to_xx

        spans = span_cxw_to_xx(output["pred_spans"][batch_index]) * float(meta["duration"])
    else:
        n_queries = output["pred_spans"].shape[1]
        span_logits = output["pred_spans"][batch_index].view(n_queries, 2, opt.max_v_l)
        _, spans = F.softmax(span_logits, dim=-1).max(-1)
        spans[..., 1] += 1
        spans = spans * int(opt.clip_length)

    query_spans = spans.detach().float().cpu().tolist()
    query_scores = score.detach().float().cpu().tolist()
    query_foreground = foreground_probability.detach().float().cpu().tolist()
    query_iou = iou_score.detach().float().cpu().tolist()
    order = torch.argsort(score, descending=True)
    sorted_spans, sorted_scores = spans[order], score[order]
    rows = torch.cat([sorted_spans, sorted_scores[:, None]], dim=-1).detach().cpu().tolist()
    return (
        [[float(value) for value in row] for row in rows],
        order.detach().cpu().tolist(),
        [[float(value) for value in row] for row in query_spans],
        [float(value) for value in query_scores],
        [float(value) for value in query_foreground],
        [float(value) for value in query_iou],
    )


def _saliency(output, valid_length: int, batch_index: int):
    values = output.get("saliency_scores")
    if values is None:
        return []
    return values[batch_index, :valid_length].detach().float().cpu().tolist()


def _normalize_video_attention(attention: torch.Tensor) -> torch.Tensor:
    """Renormalize native joint-memory attention after selecting video tokens."""

    denominator = attention.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(attention.dtype).eps
    )
    return attention / denominator


def _match_indices(criterion, output, targets):
    match_output = {
        key: output[key]
        for key in ("pred_logits", "pred_spans", "pred_masks", "iou_scores")
        if key in output
    }
    return criterion.matcher(match_output, targets)


def _indices_for_sample(indices, batch_index):
    src, dst = indices[batch_index]
    return src.detach().cpu().tolist(), dst.detach().cpu().tolist()


def _residual_summary(model):
    module = getattr(model, "query_cgp", None)
    qout = getattr(module, "last_output", None) if module is not None else None
    if qout is None:
        return None
    # residual_update is batch-first [B,Q,D], while adapted_state is the
    # sequence-first decoder convention [Q,B,D].
    residual = qout.residual_update.detach().float()
    adapted = qout.adapted_state.detach().float().transpose(0, 1)
    beta = float(module.beta.detach().cpu().item())
    update = residual * beta
    candidate = adapted - update
    ratio = update.norm(dim=-1) / candidate.norm(dim=-1).clamp_min(1e-8)
    return {
        "relative_update_mean": ratio.mean(dim=-1).cpu().tolist(),
        "relative_update_max": ratio.max(dim=-1).values.cpu().tolist(),
        "residual_norm_mean": residual.norm(dim=-1).mean(dim=-1).cpu().tolist(),
        "beta": beta,
    }


def _route_summary(model, final_indices, batch_index):
    module = getattr(model, "query_cgp", None)
    qout = getattr(module, "last_output", None) if module is not None else None
    if qout is None:
        return None
    routes = qout.basis_weights[batch_index].detach().float().cpu()
    selected = final_indices[batch_index][0].detach().cpu().tolist()
    return {"all": route_metrics(routes), "matched": route_metrics(routes, selected)}


def _make_submission_line(meta, pred_windows, saliency):
    return {
        "qid": meta["qid"],
        "query": meta.get("query", ""),
        "vid": meta.get("vid", ""),
        "pred_relevant_windows": pred_windows,
        "pred_saliency_scores": saliency,
    }


def _postprocess_submissions(submissions):
    from sim_detr.postprocessing_sim_detr import PostProcessorDETR

    processor = PostProcessorDETR(
        clip_length=2,
        min_ts_val=0,
        max_ts_val=150,
        min_w_l=2,
        max_w_l=150,
        move_window_method="left",
        process_func_names=("clip_ts", "round_multiple"),
    )
    return [processor(copy.deepcopy(lines)) for lines in submissions]


def _formal_metrics(submission, ground_truth):
    from standalone_eval.eval import eval_submission

    if not submission or not ground_truth:
        return None
    return eval_submission(submission, ground_truth, verbose=False, match_number=False)


def analyze(
    checkpoint: str,
    mode: str,
    *,
    data_path: str,
    device: str | None = None,
    output_dir: str,
    num_workers: int = 0,
    eval_bsz: int = 32,
    max_batches: int | None = None,
    skip_formal_metrics: bool = False,
    clean_iou: float = 0.1,
    inject_residual: bool | None = None,
):
    model, criterion, opt, checkpoint_data = build_model_for_mode(
        checkpoint,
        mode,
        device=device,
        option_overrides={"num_workers": num_workers, "eval_bsz": eval_bsz},
    )
    if inject_residual is not None and getattr(model, "query_cgp", None) is not None:
        install_injection_control(model, inject_residual)
    opt.eval_path = str(Path(data_path).expanduser().resolve())
    opt.num_workers = int(num_workers)
    opt.eval_bsz = int(eval_bsz)
    opt.pin_memory = bool(opt.device.type == "cuda")
    dataset = build_dataset(opt, data_path, load_labels=True)
    loader = DataLoader(
        dataset,
        batch_size=opt.eval_bsz,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        # Keep raw dataset records here; _collate_and_prepare adds matcher
        # masks after using the native collator.
        collate_fn=list,
    )

    capture = NativeCrossAttentionCapture(model).install()
    layer_count = len(model.transformer.decoder.layers)
    raw_submissions: List[List[Dict[str, Any]]] = [[] for _ in range(layer_count)]
    records: List[Dict[str, Any]] = []

    try:
        for batch_index, batch in enumerate(tqdm(loader, desc=f"analyze {mode}")):
            if max_batches is not None and batch_index >= max_batches:
                break
            metas, model_inputs, targets = _collate_and_prepare(batch, opt)
            capture.clear()
            outputs = model(**model_inputs)
            native_attention = capture.get()
            outputs_by_layer = layer_outputs(outputs)
            if len(outputs_by_layer) != layer_count:
                raise RuntimeError(
                    f"Expected {layer_count} decoder outputs, got {len(outputs_by_layer)}"
                )
            own_indices = [_match_indices(criterion, layer, targets) for layer in outputs_by_layer]
            final_indices = own_indices[-1]
            residual = _residual_summary(model)

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
                    "hungarian": {},
                    "coverage": {},
                }
                if residual is not None:
                    record.update({
                        "relative_update_mean": residual["relative_update_mean"][sample_index],
                        "relative_update_max": residual["relative_update_max"][sample_index],
                        "residual_norm_mean": residual["residual_norm_mean"][sample_index],
                        "dq_beta": residual["beta"],
                    })
                route = _route_summary(model, final_indices, sample_index)
                if route is not None:
                    record["route_all"] = route["all"]
                    record["route_matched"] = route["matched"]

                for layer_id, layer_output in enumerate(outputs_by_layer):
                    name = f"d{layer_id + 1}"
                    (
                        pred_windows,
                        query_order,
                        query_spans,
                        query_scores,
                        query_foreground,
                        query_iou,
                    ) = _decode_spans(
                        layer_output, meta, opt, sample_index
                    )
                    saliency = _saliency(outputs, valid_length, sample_index)
                    raw_submissions[layer_id].append(
                        _make_submission_line(meta, pred_windows, saliency)
                    )
                    record[f"{name}_raw_pred_windows"] = pred_windows
                    record[f"{name}_query_order"] = query_order
                    record[f"{name}_scores"] = [row[2] for row in pred_windows]
                    record[f"{name}_query_spans_seconds"] = query_spans
                    record[f"{name}_query_scores"] = query_scores
                    record[f"{name}_query_fg_probs"] = query_foreground
                    record[f"{name}_query_iou_scores"] = query_iou

                    own_q, own_g = _indices_for_sample(own_indices[layer_id], sample_index)
                    final_q, final_g = _indices_for_sample(final_indices, sample_index)
                    record["hungarian"][f"{name}_own"] = {"query": own_q, "gt": own_g}
                    record["hungarian"][f"{name}_final"] = {"query": final_q, "gt": final_g}
                    if native_attention[layer_id] is not None:
                        native = native_attention[layer_id][sample_index, :, :valid_length]
                        native = _normalize_video_attention(native)
                        record[f"{name}_own"] = binding_metrics(
                            native, gt_windows, own_q, own_g,
                            duration=float(meta["duration"]),
                        )
                        record[f"{name}_final"] = binding_metrics(
                            native, gt_windows, final_q, final_g,
                            duration=float(meta["duration"]),
                        )
                    else:
                        record[f"{name}_own"] = None
                        record[f"{name}_final"] = None
                    record["coverage"][name] = fixed_k_metrics(pred_windows, gt_windows)

                qout = getattr(getattr(model, "query_cgp", None), "last_output", None)
                if qout is not None:
                    dq_attention = qout.temporal_attention[sample_index, :, :valid_length]
                    final_q, final_g = _indices_for_sample(final_indices, sample_index)
                    record["dq_private"] = binding_metrics(
                        dq_attention, gt_windows, final_q, final_g,
                        duration=float(meta["duration"]),
                    )
                else:
                    record["dq_private"] = None

                d4_score = np.asarray(record["d4_scores"], dtype=float)
                record["active_query_count"] = {
                    str(threshold): int(np.sum(d4_score >= threshold))
                    for threshold in (0.3, 0.5, 0.7)
                }
                record["d4_top5_queries"] = record["d4_query_order"][:5]
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

    formal = None
    if not skip_formal_metrics:
        formal = {
            f"d{layer_id + 1}": _formal_metrics(submission, dataset.data)
            for layer_id, submission in enumerate(processed_submissions)
        }
    result = {
        "manifest": checkpoint_manifest(checkpoint, checkpoint_data),
        "mode": mode,
        "data": str(Path(data_path).expanduser().resolve()),
        "clean_iou": float(clean_iou),
        "num_records": len(records),
        "decoder_layers": layer_count,
        "prediction_protocol": {
            "ranking": "softmax(pred_logits)[...,0] * sigmoid(iou_scores[...,0])",
            "postprocess": "clip_ts + round_multiple, clip_length=2",
        },
        "formal_metrics": formal,
        "records": records,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "analysis.json", result)
    save_json(out_dir / "manifest.json", {
        **result["manifest"],
        "mode": mode,
        "data": result["data"],
        "num_records": len(records),
    })
    for layer_id, submission in enumerate(processed_submissions):
        save_json(out_dir / f"d{layer_id + 1}_submission.json", submission)
    print(json.dumps({
        "mode": mode,
        "checkpoint": result["manifest"]["checkpoint"],
        "num_records": len(records),
        "formal_metrics": formal,
        "output": str(out_dir.resolve()),
    }, indent=2, ensure_ascii=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["baseline", "dq_active", "dq_beta_zero", "dq_stripped"], required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-bsz", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-formal-metrics", action="store_true")
    parser.add_argument("--clean-iou", type=float, default=0.1)
    inject = parser.add_mutually_exclusive_group()
    inject.add_argument("--inject-residual", dest="inject_residual", action="store_true")
    inject.add_argument("--no-injection", dest="inject_residual", action="store_false")
    parser.set_defaults(inject_residual=None)
    args = parser.parse_args(argv)
    analyze(
        args.checkpoint,
        args.mode,
        data_path=args.data or str(default_split_path(args.split)),
        device=args.device,
        output_dir=args.output,
        num_workers=args.num_workers,
        eval_bsz=args.eval_bsz,
        max_batches=args.max_batches,
        skip_formal_metrics=args.skip_formal_metrics,
        clean_iou=args.clean_iou,
        inject_residual=args.inject_residual,
    )


if __name__ == "__main__":
    main()
