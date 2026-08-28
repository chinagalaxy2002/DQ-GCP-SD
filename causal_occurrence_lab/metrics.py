"""Occurrence-binding and fixed-K retrieval metrics.

All functions are dependency-light and operate on Python/numpy values so the
records written by the GPU analysis can be re-aggregated without a model.
Temporal windows are represented as ``[start_seconds, end_seconds]``.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Optional, Sequence

import numpy as np


EPS = 1e-12


def _array(value: Any, *, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _window_pairs(value: Any) -> np.ndarray:
    """Return only ``[start, end]`` from two- or three-column windows."""

    if len(value) == 0:
        return np.empty((0, 2), dtype=float)
    arr = _array(value)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2]
    return arr.reshape((-1, 2))


def interval_iou(a: Sequence[float], b: Sequence[float]) -> float:
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(0.0, a1 - a0) + max(0.0, b1 - b0) - inter
    return float(inter / union) if union > EPS else 0.0


def iou_matrix(
    pred_windows: Sequence[Sequence[float]],
    gt_windows: Sequence[Sequence[float]],
) -> np.ndarray:
    pred = _window_pairs(pred_windows)
    gt = _window_pairs(gt_windows)
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=float)
    left = np.maximum(pred[:, None, 0], gt[None, :, 0])
    right = np.minimum(pred[:, None, 1], gt[None, :, 1])
    intersection = np.maximum(0.0, right - left)
    pred_len = np.maximum(0.0, pred[:, 1] - pred[:, 0])[:, None]
    gt_len = np.maximum(0.0, gt[:, 1] - gt[:, 0])[None, :]
    union = pred_len + gt_len - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > EPS)


def pairwise_iou(windows: Sequence[Sequence[float]]) -> np.ndarray:
    windows = _window_pairs(windows)
    if len(windows) < 2:
        return np.empty((0,), dtype=float)
    return np.asarray(
        [interval_iou(windows[i], windows[j]) for i, j in combinations(range(len(windows)), 2)],
        dtype=float,
    )


def is_clean_multi(gt_windows: Sequence[Sequence[float]], threshold: float = 0.1) -> bool:
    return len(gt_windows) >= 2 and bool(np.all(pairwise_iou(gt_windows) < threshold))


def occurrence_bucket(gt_windows: Sequence[Sequence[float]], clean_threshold: float = 0.1) -> str:
    n = len(gt_windows)
    if n <= 1:
        return "single"
    if n == 2:
        return "two_clean" if is_clean_multi(gt_windows, clean_threshold) else "two"
    return "three_or_more_clean" if is_clean_multi(gt_windows, clean_threshold) else "three_or_more"


def _top_predictions(pred_windows: Sequence[Sequence[float]], k: int) -> np.ndarray:
    return _window_pairs(pred_windows)[: int(k)]


def fixed_k_metrics(
    pred_windows: Sequence[Sequence[float]],
    gt_windows: Sequence[Sequence[float]],
    *,
    k_list: Sequence[int] = (1, 3, 5, 10),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> Dict[str, float]:
    """Compute fixed-K coverage and corrected duplicate attribution.

    Each prediction is attributed to exactly one GT, namely its highest-IoU
    GT.  Predictions whose highest IoU is below the requested threshold are
    excluded from the valid-hit denominator.  This makes every returned DAR
    lie in ``[0, 1]`` and avoids negative multi-attribution values.
    """

    metrics: Dict[str, float] = {}
    gt = _window_pairs(gt_windows)
    for k in k_list:
        pred = _top_predictions(pred_windows, k)
        ious = iou_matrix(pred, gt)
        for threshold in iou_thresholds:
            tag = f"{int(k)}_{int(round(100 * threshold)):02d}"
            coverage = (
                float(np.mean(np.max(ious, axis=0) >= threshold))
                if len(gt) and len(pred) else 0.0
            )
            metrics[f"coverage@{tag}"] = coverage

            if len(pred) == 0 or len(gt) == 0:
                dar = 0.0
                valid_hits = 0
                unique_hits = 0
            else:
                best_gt = np.argmax(ious, axis=1)
                best_iou = np.max(ious, axis=1)
                valid = best_iou >= threshold
                attributed = best_gt[valid]
                valid_hits = int(valid.sum())
                unique_hits = int(len(np.unique(attributed)))
                dar = float(1.0 - unique_hits / valid_hits) if valid_hits else 0.0
            metrics[f"duplicate_rate@{tag}"] = float(np.clip(dar, 0.0, 1.0))
            metrics[f"dar@{tag}"] = float(np.clip(dar, 0.0, 1.0))
            metrics[f"valid_hits@{tag}"] = float(valid_hits)
            metrics[f"unique_attributed_gt@{tag}"] = float(unique_hits)

        metrics[f"pairwise_prediction_iou@{int(k)}"] = (
            float(pairwise_iou(pred).mean()) if len(pred) >= 2 else 0.0
        )
    return metrics


def video_bin_masks(
    gt_windows: Sequence[Sequence[float]],
    num_frames: int,
    duration: float,
) -> np.ndarray:
    """Return an ``[num_gt, num_frames]`` overlap mask for video bins."""

    num_frames = int(num_frames)
    gt = _window_pairs(gt_windows)
    if num_frames <= 0 or len(gt) == 0:
        return np.zeros((len(gt), max(0, num_frames)), dtype=bool)
    duration = max(float(duration), EPS)
    starts = np.arange(num_frames, dtype=float) * duration / num_frames
    ends = (np.arange(num_frames, dtype=float) + 1.0) * duration / num_frames
    return (starts[None, :] < gt[:, 1:2]) & (ends[None, :] > gt[:, 0:1])


def binding_metrics(
    attention: Any,
    gt_windows: Sequence[Sequence[float]],
    matched_queries: Sequence[int],
    matched_gt: Sequence[int],
    *,
    duration: float,
) -> Dict[str, Any]:
    """Compute metrics from GT windows expressed in seconds.

    This compatibility API is useful for generic metric experiments.  The
    checkpoint analyzer uses :func:`binding_metrics_from_target_spans` so its
    masks have exactly the same normalized-span semantics as production
    ``L_bind``.
    """

    attn = _array(attention, dtype=float)
    if attn.ndim != 2:
        raise ValueError(f"Expected [num_queries, num_frames] attention, got {attn.shape}")
    gt = _window_pairs(gt_windows)
    masks = video_bin_masks(gt, attn.shape[-1], duration)
    return _binding_metrics_from_masks(
        attn,
        masks,
        matched_queries,
        matched_gt,
        semantics="seconds_to_duration_bins",
    )


def _binding_metrics_from_masks(
    attention: Any,
    masks: Any,
    matched_queries: Sequence[int],
    matched_gt: Sequence[int],
    *,
    semantics: str,
) -> Dict[str, Any]:
    """Compute binding statistics from an explicit ``[GT, frame]`` mask."""

    attn = _array(attention, dtype=float)
    if attn.ndim != 2:
        raise ValueError(f"Expected [num_queries, num_frames] attention, got {attn.shape}")
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[-1] != attn.shape[-1]:
        raise ValueError(
            "Expected masks with shape [num_gt, num_frames] matching attention; "
            f"got {masks.shape} for {attn.shape}"
        )

    evidence = attn[:, None, :] * masks[None, :, :]
    evidence = evidence.sum(axis=-1) if len(masks) else np.zeros((attn.shape[0], 0))
    lengths = (
        masks.sum(axis=-1).astype(float) / max(attn.shape[-1], 1)
        if len(masks) else np.empty((0,))
    )
    enrich = evidence / np.maximum(lengths[None, :], EPS) if len(masks) else evidence

    rows = []
    for query, target in zip(matched_queries, matched_gt):
        query = int(query)
        target = int(target)
        if query < 0 or query >= attn.shape[0] or target < 0 or target >= len(masks):
            continue
        raw = evidence[query]
        norm = enrich[query]
        raw_dom = int(np.argmax(raw)) if len(raw) else -1
        norm_dom = int(np.argmax(norm)) if len(norm) else -1
        raw_other = np.delete(raw, target) if len(raw) > 1 else np.empty((0,))
        norm_other = np.delete(norm, target) if len(norm) > 1 else np.empty((0,))
        rows.append({
            "query": query,
            "target": target,
            "evidence": raw.tolist(),
            "enrichment": norm.tolist(),
            "target_mass": float(raw[target]),
            "target_enrichment": float(norm[target]),
            "dominant_gt": raw_dom,
            "dominant_gt_norm": norm_dom,
            "margin": float(raw[target] - np.max(raw_other)) if len(raw_other) else float(raw[target]),
            "margin_norm": float(norm[target] - np.max(norm_other)) if len(norm_other) else float(norm[target]),
        })

    targets = [row["target"] for row in rows]
    raw_dominants = [row["dominant_gt"] for row in rows]
    norm_dominants = [row["dominant_gt_norm"] for row in rows]
    pairs = [
        (i, j) for i, j in combinations(range(len(rows)), 2)
        if rows[i]["target"] != rows[j]["target"]
    ]
    raw_collisions = int(sum(raw_dominants[i] == raw_dominants[j] for i, j in pairs))
    norm_collisions = int(sum(norm_dominants[i] == norm_dominants[j] for i, j in pairs))
    raw_correct = int(sum(dom == target for dom, target in zip(raw_dominants, targets)))
    norm_correct = int(sum(dom == target for dom, target in zip(norm_dominants, targets)))
    return {
        "num_matched": len(rows),
        "num_correct": raw_correct,
        "num_correct_norm": norm_correct,
        "num_valid_pairs": len(pairs),
        "num_collisions": raw_collisions,
        "num_collisions_norm": norm_collisions,
        "aec": float(raw_correct / len(rows)) if rows else 0.0,
        "aec_norm": float(norm_correct / len(rows)) if rows else 0.0,
        "binding_margin": float(np.mean([row["margin"] for row in rows])) if rows else 0.0,
        "binding_margin_norm": float(np.mean([row["margin_norm"] for row in rows])) if rows else 0.0,
        "ecr": float(raw_collisions / len(pairs)) if pairs else 0.0,
        "ecr_norm": float(norm_collisions / len(pairs)) if pairs else 0.0,
        "own_mass": float(np.mean([row["target_mass"] for row in rows])) if rows else 0.0,
        "own_enrichment": float(np.mean([row["target_enrichment"] for row in rows])) if rows else 0.0,
        "mask_semantics": semantics,
        "matches": rows,
    }


def binding_metrics_from_target_spans(
    attention: Any,
    target_spans: Any,
    matched_queries: Sequence[int],
    matched_gt: Sequence[int],
    *,
    valid_length: int,
    span_loss_type: str,
) -> Dict[str, Any]:
    """Compute metrics with the exact normalized target geometry used by ``L_bind``.

    ``target_spans`` must be the collated production ``span_labels`` tensor:
    normalized ``(center,width)`` for ``l1`` or inclusive clip indices for
    ``ce``.  The overlap mask is deliberately delegated to the isolated
    control helper, which is the same implementation used by the causal
    training controls.
    """

    attn = _array(attention, dtype=float)
    if attn.ndim != 2:
        raise ValueError(f"Expected [num_queries, num_frames] attention, got {attn.shape}")
    valid_length = int(valid_length)
    if valid_length <= 0 or valid_length > attn.shape[-1]:
        raise ValueError(
            f"valid_length={valid_length} is incompatible with attention shape {attn.shape}"
        )

    import torch

    from causal_occurrence_lab.controls import _overlap_for_targets

    spans = torch.as_tensor(_array(target_spans, dtype=np.float32), dtype=torch.float32)
    masks = _overlap_for_targets(
        spans,
        valid_length,
        span_loss_type=span_loss_type,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).detach().cpu().numpy()
    return _binding_metrics_from_masks(
        attn[:, :valid_length],
        masks,
        matched_queries,
        matched_gt,
        semantics="production_normalized_target_spans",
    )


def route_metrics(weights: Any, selected_rows: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Report conditional/marginal routing entropy and effective bases."""

    arr = _array(weights, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {
            "num_routes": 0,
            "conditional_entropy": 0.0,
            "conditional_entropy_norm": 0.0,
            "marginal_entropy": 0.0,
            "marginal_entropy_norm": 0.0,
            "effective_basis": 0.0,
            "marginal_usage": [],
            "argmax_usage": [],
        }
    if selected_rows is not None:
        arr = arr[np.asarray(selected_rows, dtype=int)]
    arr = np.maximum(arr, EPS)
    arr = arr / arr.sum(axis=-1, keepdims=True)
    conditional = float(np.mean(-np.sum(arr * np.log(arr), axis=-1)))
    marginal = arr.mean(axis=0)
    marginal = np.maximum(marginal, EPS)
    marginal = marginal / marginal.sum()
    marginal_entropy = float(-np.sum(marginal * np.log(marginal)))
    num_basis = arr.shape[-1]
    usage = np.bincount(np.argmax(arr, axis=-1), minlength=num_basis).astype(int).tolist()
    return {
        "num_routes": int(arr.shape[0]),
        "conditional_entropy": conditional,
        "conditional_entropy_norm": conditional / math.log(max(num_basis, 2)),
        "marginal_entropy": marginal_entropy,
        "marginal_entropy_norm": marginal_entropy / math.log(max(num_basis, 2)),
        "effective_basis": float(math.exp(marginal_entropy)),
        "marginal_usage": marginal.tolist(),
        "argmax_usage": usage,
    }


__all__ = [
    "binding_metrics", "binding_metrics_from_target_spans", "fixed_k_metrics", "interval_iou", "iou_matrix",
    "is_clean_multi", "occurrence_bucket", "pairwise_iou", "route_metrics",
    "video_bin_masks",
]
