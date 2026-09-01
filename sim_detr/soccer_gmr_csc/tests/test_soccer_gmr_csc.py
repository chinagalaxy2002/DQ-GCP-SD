import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sim_detr.soccer_gmr_csc.dataset import SoccerGMRDataset, soccer_gmr_collate
from sim_detr.soccer_gmr_csc.evaluation import official_gmr_metrics


class SoccerGMRCSCTest(unittest.TestCase):
    def test_dataset_handles_null_single_and_multi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip"
            slowfast = root / "slowfast"
            text = root / "text"
            for path in (clip, slowfast, text):
                path.mkdir()
            rows = [
                {"qid": 1, "vid": "a", "query": "null", "duration": 10, "relevant_windows": []},
                {"qid": 2, "vid": "b", "query": "single", "duration": 10, "relevant_windows": [[2, 4]]},
                {"qid": 3, "vid": "c", "query": "multi", "duration": 10, "relevant_windows": [[0, 2], [6, 10]]},
            ]
            labels = root / "labels.jsonl"
            labels.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            for row in rows:
                np.savez(clip / f"{row['vid']}.npz", features=np.ones((5, 2), np.float32))
                np.savez(slowfast / f"{row['vid']}.npz", features=np.ones((5, 3), np.float32))
                np.savez(
                    text / f"qid{row['qid']}.npz",
                    last_hidden_state=np.ones((6, 4), np.float32),
                    attention_mask=np.array([1, 1, 1, 0, 0, 0], np.float32),
                )
            dataset = SoccerGMRDataset(labels, [clip, slowfast], text, max_v_l=5, clip_len=2)
            items = [dataset[index] for index in range(3)]
            self.assertEqual([len(item["model_inputs"]["span_labels"]) for item in items], [0, 1, 2])
            self.assertEqual([item["model_inputs"]["query_feat"].shape[0] for item in items], [3, 3, 3])
            self.assertEqual(items[0]["model_inputs"]["pos_mask"].sum().item(), 0)
            self.assertEqual(items[1]["model_inputs"]["pos_mask"].sum().item(), 1)
            _, batch = soccer_gmr_collate(items)
            self.assertEqual(tuple(batch["video_feat"][0].shape), (3, 5, 7))

    def test_official_metrics_accept_native_existence_proxy(self):
        ground_truth = [
            {"qid": 1, "relevant_windows": []},
            {"qid": 2, "relevant_windows": [[2, 4]]},
            {"qid": 3, "relevant_windows": [[0, 2], [6, 8]]},
        ]
        submission = [
            {"qid": 1, "pred_relevant_windows": [[0, 2, 0.1]], "pred_exist_score": 0.1},
            {"qid": 2, "pred_relevant_windows": [[2, 4, 0.9]], "pred_exist_score": 0.9},
            {"qid": 3, "pred_relevant_windows": [[0, 2, 0.9], [6, 8, 0.8]], "pred_exist_score": 0.9},
        ]
        metrics = official_gmr_metrics(
            submission, ground_truth,
            "/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval",
        )
        self.assertEqual(metrics["brief"]["AUROC"], 100.0)
        self.assertEqual(metrics["brief"]["mAP"], 100.0)
        self.assertEqual(metrics["stats"]["num_multi_instance"], 1)


if __name__ == "__main__":
    unittest.main()
