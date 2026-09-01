"""Prediction and official Soccer-GMR evaluation utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sim_detr.span_utils import span_cxw_to_xx

from .dataset import prepare_soccer_gmr_batch, soccer_gmr_collate


def _round_window(window, duration, clip_length):
    start, end = window
    start = max(0.0, min(float(duration), round(float(start) / clip_length) * clip_length))
    end = max(0.0, min(float(duration), round(float(end) / clip_length) * clip_length))
    if end <= start:
        if start < duration:
            end = min(float(duration), start + clip_length)
        else:
            start = max(0.0, float(duration) - clip_length)
            end = float(duration)
    return start, end


@torch.no_grad()
def generate_submission(model, dataset, opt):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=opt.eval_bsz, shuffle=False,
        num_workers=opt.num_workers, pin_memory=opt.pin_memory,
        collate_fn=soccer_gmr_collate,
    )
    submission = []
    for metadata, batched in loader:
        model_inputs, _ = prepare_soccer_gmr_batch(
            batched, opt.device, non_blocking=opt.pin_memory
        )
        outputs = model(**model_inputs)
        foreground = F.softmax(outputs["pred_logits"], dim=-1)[..., 0]
        localization_quality = torch.sigmoid(outputs["iou_scores"][..., 0])
        scores = foreground * localization_quality
        existence = torch.sigmoid(outputs["pred_exist_logits"])
        multiplier = torch.where(
            existence >= opt.exist_gate_thd, torch.ones_like(existence), existence
        )
        scores = scores * multiplier.unsqueeze(1)
        spans = span_cxw_to_xx(outputs["pred_spans"])
        for meta, sample_spans, sample_scores, exist_score in zip(
            metadata, spans, scores, existence
        ):
            duration = float(meta["duration"])
            windows = []
            for span, score in zip(sample_spans.cpu().tolist(), sample_scores.cpu().tolist()):
                start, end = _round_window(
                    (span[0] * duration, span[1] * duration), duration, opt.clip_length
                )
                windows.append([start, end, float(score)])
            windows.sort(key=lambda item: item[2], reverse=True)
            submission.append({
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": windows[:opt.num_queries],
                "pred_exist_score": float(exist_score.cpu()),
            })
    return submission


def official_gmr_metrics(submission, ground_truth, gmr_root, verbose=False):
    root = str(Path(gmr_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval.eval_main import evaluate_gmr

    return evaluate_gmr(
        submission,
        ground_truth,
        k_list=(1, 3, 5),
        max_pred_windows=10,
        cls_thresholds=(0.4, 0.6),
        gmiou_cls_threshold=0.4,
        map_num_workers=8,
        verbose=verbose,
    )


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(value, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def evaluate_model(model, dataset, opt, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = generate_submission(model, dataset, opt)
    submission_path = output_dir / f"{prefix}_predictions.jsonl"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    save_jsonl(submission, submission_path)
    metrics = official_gmr_metrics(
        submission, dataset.data, opt.gmr_root, verbose=opt.debug
    )
    save_json(metrics, metrics_path)
    return metrics, submission_path, metrics_path
