"""Occurrence binding and layerwise ownership persistence metrics.

This module provides functions to evaluate:
- D1 to D4 Attention-Evidence Consistency (AEC)
- D1 to D4 Evidence Collision Rate (ECR)
- D1 -> D4 Ownership Persistence (tracking whether queries that acquired ownership
  at early layers maintain that ownership through to final layers)
- Multi-occurrence performance breakdowns (single, multi, clean_multi, two, three_or_more)
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_occurrence_lab.metrics import (
    _array,
    _binding_metrics_from_masks,
    _window_pairs,
    fixed_k_metrics,
    interval_iou,
    iou_matrix,
    is_clean_multi,
    occurrence_bucket,
    pairwise_iou,
)


def compute_layerwise_ownership_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    decoder_layers: int = 4,
) -> Dict[str, Any]:
    """Aggregate layerwise AEC, ECR, and D1->D4 ownership persistence.

    Args:
        records: List of sample evaluation records containing layerwise binding stats.
        decoder_layers: Number of decoder layers (default: 4).

    Returns:
        Dict containing layerwise metrics (d1..d4 AEC, ECR, mass, margin) and
        persistence metrics (D1->D2, D1->D3, D1->D4 retention rates).
    """
    layer_names = [f"d{i + 1}" for i in range(decoder_layers)]
    summary: Dict[str, Any] = {
        "num_records": len(records),
        "layers": {},
        "persistence": {},
    }

    # Collect matched query trajectories across layers
    # Query key: (sample_qid/idx, query_idx, target_gt_idx)
    # Value: dict of layer_id -> {correct, mass, margin, dominant_gt}
    trajectories: List[Dict[str, Any]] = []

    for record_idx, record in enumerate(records):
        num_gt = record.get("num_gt", 0)
        if num_gt == 0:
            continue

        # Get matched query-GT pairs for D4 final assignment
        d4_binding = record.get("d4_final")
        if not d4_binding or not d4_binding.get("matches"):
            continue

        sample_matches = d4_binding["matches"]
        for match_idx, match_info in enumerate(sample_matches):
            q_id = match_info["query"]
            tgt_gt = match_info["target"]

            trajectory: Dict[str, Any] = {
                "record_idx": record_idx,
                "qid": record.get("qid"),
                "query": q_id,
                "target": tgt_gt,
                "num_gt": num_gt,
                "clean_multi": record.get("clean_multi", False),
                "layers": {},
            }

            for l_idx in range(decoder_layers):
                l_name = f"d{l_idx + 1}"
                l_binding = record.get(f"{l_name}_final")
                if l_binding and l_binding.get("matches"):
                    # Find the match corresponding to (q_id, tgt_gt)
                    l_match = next(
                        (
                            m
                            for m in l_binding["matches"]
                            if m["query"] == q_id and m["target"] == tgt_gt
                        ),
                        None,
                    )
                    if l_match is not None:
                        is_correct = bool(l_match["dominant_gt"] == tgt_gt)
                        trajectory["layers"][l_name] = {
                            "correct": is_correct,
                            "target_mass": float(l_match["target_mass"]),
                            "margin": float(l_match["margin"]),
                            "dominant_gt": int(l_match["dominant_gt"]),
                            "own_enrichment": float(l_match["target_enrichment"]),
                        }

            trajectories.append(trajectory)

    # 1. Layerwise AEC, ECR, Binding Margin, Mass
    for l_idx in range(decoder_layers):
        l_name = f"d{l_idx + 1}"
        aec_values = []
        ecr_values = []
        margin_values = []
        mass_values = []

        for record in records:
            l_binding = record.get(f"{l_name}_final")
            if l_binding is not None and l_binding.get("num_matched", 0) > 0:
                aec_values.append(float(l_binding.get("aec", 0.0)))
                margin_values.append(float(l_binding.get("binding_margin", 0.0)))
                mass_values.append(float(l_binding.get("own_mass", 0.0)))
                if l_binding.get("num_valid_pairs", 0) > 0:
                    ecr_values.append(float(l_binding.get("ecr", 0.0)))

        summary["layers"][l_name] = {
            "aec": float(np.mean(aec_values)) if aec_values else 0.0,
            "ecr": float(np.mean(ecr_values)) if ecr_values else 0.0,
            "binding_margin": float(np.mean(margin_values)) if margin_values else 0.0,
            "own_mass": float(np.mean(mass_values)) if mass_values else 0.0,
            "num_evaluated_samples": len(aec_values),
        }

    # 2. Ownership Persistence Metrics (from D1 to D2, D3, D4)
    if trajectories and "d1" in summary["layers"]:
        valid_traj_d1 = [t for t in trajectories if "d1" in t["layers"]]
        d1_correct_traj = [
            t for t in valid_traj_d1 if t["layers"]["d1"]["correct"]
        ]

        persistence: Dict[str, Any] = {
            "total_matched_trajectories": len(valid_traj_d1),
            "d1_correct_count": len(d1_correct_traj),
            "d1_correct_rate": len(d1_correct_traj) / max(len(valid_traj_d1), 1),
        }

        for target_layer in ("d2", "d3", "d4"):
            if target_layer in summary["layers"]:
                # Trajectories that were correct at D1 and are also present at target layer
                d1_target_pairs = [
                    t for t in d1_correct_traj if target_layer in t["layers"]
                ]
                retained = [
                    t
                    for t in d1_target_pairs
                    if t["layers"][target_layer]["correct"]
                ]
                persistence[f"d1_to_{target_layer}_persistence"] = (
                    float(len(retained) / len(d1_target_pairs))
                    if d1_target_pairs
                    else 0.0
                )

                # Mass change from D1 to target layer
                valid_both = [
                    t
                    for t in valid_traj_d1
                    if target_layer in t["layers"]
                ]
                mass_diffs = [
                    t["layers"][target_layer]["target_mass"]
                    - t["layers"]["d1"]["target_mass"]
                    for t in valid_both
                ]
                washout_drops = [
                    1.0
                    for diff in mass_diffs
                    if diff < -0.05
                ]
                persistence[f"d1_to_{target_layer}_mass_diff_mean"] = (
                    float(np.mean(mass_diffs)) if mass_diffs else 0.0
                )
                persistence[f"d1_to_{target_layer}_washout_drop_rate"] = (
                    float(len(washout_drops) / len(valid_both))
                    if valid_both
                    else 0.0
                )

        summary["persistence"] = persistence

    return summary


def compute_multi_occurrence_stratified_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    subsets: Sequence[str] = ("single", "multi", "clean_multi", "two", "three_or_more"),
) -> Dict[str, Any]:
    """Compute binding and coverage metrics stratified by occurrence count.

    Args:
        records: List of sample evaluation records.
        subsets: List of subset names to stratify by.

    Returns:
        Dict mapping subset name to subset summary metrics.
    """
    results: Dict[str, Any] = {}

    for subset in subsets:
        if subset == "single":
            subset_rows = [r for r in records if r.get("num_gt", 0) <= 1]
        elif subset == "multi":
            subset_rows = [r for r in records if r.get("num_gt", 0) >= 2]
        elif subset == "clean_multi":
            subset_rows = [
                r for r in records
                if r.get("num_gt", 0) >= 2 and r.get("clean_multi", False)
            ]
        elif subset == "two":
            subset_rows = [r for r in records if r.get("num_gt", 0) == 2]
        elif subset == "three_or_more":
            subset_rows = [r for r in records if r.get("num_gt", 0) >= 3]
        else:
            subset_rows = list(records)

        layer_summary = compute_layerwise_ownership_summary(subset_rows)

        # Also aggregate coverage @ 5_50 and 5_70 if present
        cov_5_50 = []
        cov_5_70 = []
        for r in subset_rows:
            cov = r.get("coverage", {}).get("d4", {})
            if "coverage@5_50" in cov:
                cov_5_50.append(float(cov["coverage@5_50"]))
            if "coverage@5_70" in cov:
                cov_5_70.append(float(cov["coverage@5_70"]))

        results[subset] = {
            "num_records": len(subset_rows),
            "layerwise": layer_summary["layers"],
            "persistence": layer_summary.get("persistence", {}),
            "coverage@5_50": float(np.mean(cov_5_50)) if cov_5_50 else 0.0,
            "coverage@5_70": float(np.mean(cov_5_70)) if cov_5_70 else 0.0,
        }

    return results


__all__ = [
    "compute_layerwise_ownership_summary",
    "compute_multi_occurrence_stratified_metrics",
    "fixed_k_metrics",
    "is_clean_multi",
    "occurrence_bucket",
]
