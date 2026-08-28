import numpy as np
import unittest

from causal_occurrence_lab.metrics import (
    binding_metrics,
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
