import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sim_detr.soccer_gmr_csc.dataset import SoccerGMRDataset, soccer_gmr_collate
from sim_detr.soccer_gmr_csc.evaluation import official_gmr_metrics
from sim_detr.semantic_calibration.semantic_calibrator import softmax_evidence_weights


class SoccerGMRCSCTest(unittest.TestCase):
    def test_native_mask_logits_keep_candidate_specific_weights(self):
        logits = torch.tensor([[[3.0, 0.0, -2.0, 100.0],
                                [-2.0, 0.0, 3.0, 100.0]]])
        valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        weights = softmax_evidence_weights(logits, valid)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))
        self.assertTrue(torch.all(weights[..., -1] == 0))
        self.assertGreater((weights[:, 0] - weights[:, 1]).abs().sum().item(), 1.0)

    def test_null_background_weight_is_lower_than_foreground_weight(self):
        from sim_detr.soccer_gmr_csc.criterion import NullSafeCriterion

        class Native:
            foreground_label = 0
            background_label = 1

            @staticmethod
            def _get_src_permutation_idx(indices):
                return (
                    torch.cat([item[0] + batch for batch, item in enumerate(indices)]),
                    torch.cat([item[1] for item in indices]),
                )

        criterion = NullSafeCriterion.__new__(NullSafeCriterion)
        criterion.native = Native()
        criterion.background_focal_weight = 0.1
        criterion.null_background_focal_weight = 0.05
        logits = torch.zeros((2, 2, 2), requires_grad=True)
        outputs = {"pred_logits": logits}
        targets = {"exist_label": torch.tensor([1.0, 0.0])}
        indices = [(torch.tensor([0]), torch.tensor([0])),
                   (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        loss = criterion._null_aware_label_loss(outputs, targets, indices)["loss_label"]
        loss.backward()
        foreground_grad = logits.grad[0, 0].abs().sum()
        positive_background_grad = logits.grad[0, 1].abs().sum()
        null_background_grad = logits.grad[1].abs().sum() / 2
        self.assertGreater(foreground_grad, positive_background_grad)
        self.assertGreater(positive_background_grad, null_background_grad)

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

    def test_hungarian_d1_binding_loss(self):
        from sim_detr.soccer_gmr_csc.binding_loss import native_matched_binding_loss, span_overlap_mask

        # Test overlap mask
        spans = torch.tensor([[0.5, 0.4]])  # cx=0.5, w=0.4 -> [0.3, 0.7]
        overlap = span_overlap_mask(spans, 10, torch.float32, torch.device("cpu"))
        self.assertEqual(overlap.shape, (1, 10))
        # frames 3, 4, 5, 6 should be True
        self.assertTrue(torch.all(overlap[0, 3:7]))

        # Test matched binding loss computation
        attention = torch.zeros((1, 2, 10), requires_grad=True)
        # Put attention at positive frames
        with torch.no_grad():
            attention[0, 0, 3:7] = 0.25
            attention[0, 1, 0:2] = 0.5
        vmask = torch.ones((1, 10), dtype=torch.bool)
        targets = {
            "span_labels": [{"spans": spans}],
            "exist_label": torch.tensor([1.0]),
        }
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = native_matched_binding_loss(attention, vmask, targets, indices)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

        # Test null safety
        indices_null = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        null_loss = native_matched_binding_loss(attention, vmask, targets, indices_null)
        self.assertEqual(null_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
