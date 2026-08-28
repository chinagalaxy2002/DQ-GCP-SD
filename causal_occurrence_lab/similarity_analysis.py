"""Stratify saved analyses by CLIP similarity between repeated occurrences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.common import save_json
from causal_occurrence_lab.metrics import fixed_k_metrics, video_bin_masks


def occurrence_similarity(record, feature_dir: str | Path):
    windows = record.get("gt_windows", [])
    if len(windows) < 2:
        return None
    path = Path(feature_dir) / f"{record['vid']}.npy"
    if not path.is_file():
        return None
    features = np.asarray(np.load(path), dtype=np.float32)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    if len(features) == 0:
        return None
    features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    masks = video_bin_masks(windows, len(features), record["duration"])
    pooled = []
    for mask in masks:
        if mask.any():
            value = features[mask].mean(axis=0)
        else:
            center = 0.5 * (float(windows[len(pooled)][0]) + float(windows[len(pooled)][1]))
            frame = int(round(center / max(float(record["duration"]), 1e-12) * len(features)))
            frame = min(max(frame, 0), len(features) - 1)
            value = features[frame]
        value = value / max(float(np.linalg.norm(value)), 1e-12)
        pooled.append(value)
    values = [float(np.dot(pooled[i], pooled[j])) for i in range(len(pooled)) for j in range(i + 1, len(pooled))]
    return float(np.mean(values)) if values else None


def summarize(records, feature_dir, *, groups=3):
    rows = []
    for record in records:
        similarity = occurrence_similarity(record, feature_dir)
        if similarity is not None:
            rows.append((similarity, record))
    rows.sort(key=lambda item: item[0])
    if not rows:
        return {"num_records": 0, "groups": {}}
    boundaries = [rows[int(len(rows) * i / groups)][0] for i in range(1, groups)]
    output = {"num_records": len(rows), "boundaries": boundaries, "groups": {}}
    for group_id in range(groups):
        start = int(len(rows) * group_id / groups)
        end = int(len(rows) * (group_id + 1) / groups)
        part = rows[start:end]
        group_name = ["low", "medium", "high"][group_id] if groups == 3 else str(group_id)
        aec = [r["d4_final"]["aec"] for _, r in part]
        ecr = [r["d4_final"]["ecr"] for _, r in part]
        coverage = [r["coverage"]["d4"]["coverage@5_50"] for _, r in part]
        output["groups"][group_name] = {
            "n": len(part),
            "similarity_mean": float(np.mean([value for value, _ in part])),
            "similarity_min": float(min(value for value, _ in part)),
            "similarity_max": float(max(value for value, _ in part)),
            "aec_d4_final": float(np.mean(aec)),
            "ecr_d4_final": float(np.mean(ecr)),
            "coverage@5_50": float(np.mean(coverage)),
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--clip-feature-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.analysis).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    result = summarize(data.get("records", []), args.clip_feature_dir)
    result["analysis"] = str(Path(args.analysis).resolve())
    result["clip_feature_dir"] = str(Path(args.clip_feature_dir).resolve())
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
