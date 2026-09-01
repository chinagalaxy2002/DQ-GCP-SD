"""Soccer-GMR NPZ feature adapter for native Sim-DETR inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sim_detr.span_utils import span_xx_to_cxw
from utils.basic_utils import l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d


class SoccerGMRDataset(Dataset):
    def __init__(
        self, data_path, v_feat_dirs, q_feat_dir, max_q_l=32, max_v_l=75,
        max_windows=8, clip_len=2, load_labels=True, data_ratio=1.0,
    ):
        self.data_path = Path(data_path)
        self.v_feat_dirs = [Path(path) for path in v_feat_dirs]
        self.q_feat_dir = Path(q_feat_dir)
        self.max_q_l = int(max_q_l)
        self.max_v_l = int(max_v_l)
        self.max_windows = int(max_windows)
        self.clip_len = float(clip_len)
        self.load_labels = bool(load_labels)
        with self.data_path.open(encoding="utf-8") as handle:
            self.data = [json.loads(line) for line in handle if line.strip()]
        if data_ratio != 1.0:
            self.data = self.data[:max(1, int(len(self.data) * data_ratio))]
        self._validate_features()

    def _validate_features(self):
        missing = []
        for row in self.data:
            paths = [directory / f"{row['vid']}.npz" for directory in self.v_feat_dirs]
            paths.append(self.q_feat_dir / f"qid{row['qid']}.npz")
            absent = [str(path) for path in paths if not path.exists()]
            if absent:
                missing.append({"qid": row["qid"], "paths": absent})
        if missing:
            raise FileNotFoundError(
                f"Missing features for {len(missing)}/{len(self.data)} records; first={missing[0]}"
            )

    def __len__(self):
        return len(self.data)

    def _query_feature(self, qid):
        with np.load(self.q_feat_dir / f"qid{qid}.npz") as archive:
            feature = archive["last_hidden_state"].astype(np.float32)
            attention = archive["attention_mask"].astype(bool)
        feature = feature[:self.max_q_l]
        attention = attention[:self.max_q_l]
        feature = feature[attention]
        if len(feature) == 0:
            raise ValueError(f"Query {qid} has no valid CLIP tokens")
        return torch.from_numpy(l2_normalize_np_array(feature))

    def _video_feature(self, vid):
        features = []
        for directory in self.v_feat_dirs:
            with np.load(directory / f"{vid}.npz") as archive:
                feature = archive["features"][:self.max_v_l].astype(np.float32)
            features.append(l2_normalize_np_array(feature))
        length = min(len(feature) for feature in features)
        feature = np.concatenate([item[:length] for item in features], axis=-1)
        tef_start = np.arange(length, dtype=np.float32) / max(length, 1)
        tef = np.stack([tef_start, tef_start + 1.0 / max(length, 1)], axis=-1)
        return torch.from_numpy(np.concatenate([feature, tef], axis=-1))

    def __getitem__(self, index):
        meta = self.data[index]
        query = self._query_feature(meta["qid"])
        video = self._video_feature(meta["vid"])
        inputs = {"query_feat": query, "video_feat": video}
        if self.load_labels:
            windows = list(meta.get("relevant_windows", []))[:self.max_windows]
            duration = len(video) * self.clip_len
            spans = torch.tensor(windows, dtype=torch.float32).reshape(-1, 2)
            if len(spans):
                spans = span_xx_to_cxw(spans / duration)
            masks = torch.zeros((len(windows), len(video)), dtype=torch.float32)
            positive_mask = torch.zeros(len(video), dtype=torch.float32)
            for window_index, (start, end) in enumerate(windows):
                start_index = max(0, min(len(video), int(math.floor(start / self.clip_len))))
                end_index = max(start_index + 1, int(math.ceil(end / self.clip_len)))
                end_index = min(len(video), end_index)
                masks[window_index, start_index:end_index] = 1.0
                positive_mask[start_index:end_index] = 1.0
            inputs.update({
                "span_labels": spans,
                "mask_labels": masks,
                "pos_mask": positive_mask,
                "exist_label": float(bool(windows)),
            })
        return {"meta": meta, "model_inputs": inputs}


def soccer_gmr_collate(batch):
    metadata = [item["meta"] for item in batch]
    inputs = {}
    for key in ("query_feat", "video_feat", "pos_mask"):
        if key in batch[0]["model_inputs"]:
            inputs[key] = pad_sequences_1d(
                [item["model_inputs"][key] for item in batch], dtype=torch.float32
            )
    if "span_labels" in batch[0]["model_inputs"]:
        inputs["span_labels"] = [
            {"spans": item["model_inputs"]["span_labels"]} for item in batch
        ]
        inputs["mask_labels"] = [item["model_inputs"]["mask_labels"] for item in batch]
        inputs["exist_label"] = torch.tensor(
            [item["model_inputs"]["exist_label"] for item in batch], dtype=torch.float32
        )
    return metadata, inputs


def prepare_soccer_gmr_batch(batched, device, non_blocking=False):
    model_inputs = {
        "src_txt": batched["query_feat"][0].to(device, non_blocking=non_blocking),
        "src_txt_mask": batched["query_feat"][1].to(device, non_blocking=non_blocking),
        "src_vid": batched["video_feat"][0].to(device, non_blocking=non_blocking),
        "src_vid_mask": batched["video_feat"][1].to(device, non_blocking=non_blocking),
    }
    if "span_labels" not in batched:
        return model_inputs, None
    video_length = model_inputs["src_vid"].shape[1]
    mask_labels = []
    for masks in batched["mask_labels"]:
        padded = torch.zeros((len(masks), video_length), dtype=torch.float32, device=device)
        if len(masks):
            padded[:, :masks.shape[1]] = masks.to(device, non_blocking=non_blocking)
        mask_labels.append(padded)
    targets = {
        "span_labels": [
            {"spans": item["spans"].to(device, non_blocking=non_blocking)}
            for item in batched["span_labels"]
        ],
        "mask_labels": mask_labels,
        "src_pos_mask": batched["pos_mask"][0].to(device, non_blocking=non_blocking),
        "exist_label": batched["exist_label"].to(device, non_blocking=non_blocking),
    }
    return model_inputs, targets
