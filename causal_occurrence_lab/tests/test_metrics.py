import numpy as np
import unittest
import torch

from causal_occurrence_lab.metrics import (
    binding_metrics,
    binding_metrics_from_target_spans,
    fixed_k_metrics,
    is_clean_multi,
)


class MetricTests(unittest.TestCase):
    def test_duplicate_attribution_is_bounded_and_single_assignment(self):
        result = fixed_k_metrics(
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, 2.0], [4.0, 6.0]],
            k_list=(2,),
            iou_thresholds=(0.5,),
        )
        self.assertEqual(result["duplicate_rate@2_50"], 0.5)
        self.assertGreaterEqual(result["duplicate_rate@2_50"], 0.0)
        self.assertLessEqual(result["duplicate_rate@2_50"], 1.0)
        self.assertEqual(result["unique_attributed_gt@2_50"], 1.0)

    def test_fixed_k_accepts_scored_prediction_rows(self):
        result = fixed_k_metrics(
            [[0.0, 2.0, 0.9], [8.0, 10.0, 0.1]],
            [[0.0, 2.0]],
            k_list=(1,),
            iou_thresholds=(0.5,),
        )
        self.assertEqual(result["coverage@1_50"], 1.0)
        self.assertEqual(result["valid_hits@1_50"], 1.0)


    def test_clean_multi_occurrence(self):
        self.assertTrue(is_clean_multi([[0, 2], [5, 7]], threshold=0.1))
        self.assertFalse(is_clean_multi([[0, 4], [2, 6]], threshold=0.1))


    def test_length_normalized_binding_can_remove_window_length_bias(self):
        attention = np.asarray([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
        gt = [[0, 1], [2, 4]]
        result = binding_metrics(attention, gt, [0, 1], [0, 1], duration=4)
        self.assertEqual(result["aec"], 1.0)
        self.assertEqual(result["aec_norm"], 1.0)
        self.assertEqual(result["ecr"], 0.0)

    def test_target_span_metrics_use_production_normalized_geometry(self):
        # With valid_length=4, these normalized cx/w spans cover bins [0, 1]
        # and [2, 3], respectively.  The test also exercises the count fields
        # used by the summary's micro aggregation.
        attention = torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
            dtype=torch.float32,
        )
        target_spans = torch.tensor(
            [[0.375, 0.25], [0.875, 0.25]], dtype=torch.float32
        )
        result = binding_metrics_from_target_spans(
            attention,
            target_spans,
            [0, 1],
            [0, 1],
            valid_length=4,
            span_loss_type="l1",
        )
        self.assertEqual(result["mask_semantics"], "production_normalized_target_spans")
        self.assertEqual(result["num_matched"], 2)
        self.assertEqual(result["num_correct"], 2)
        self.assertEqual(result["num_valid_pairs"], 1)
        self.assertEqual(result["num_collisions"], 0)
        self.assertEqual(result["aec"], 1.0)
        self.assertEqual(result["ecr"], 0.0)
