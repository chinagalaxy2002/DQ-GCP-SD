import sys
from pathlib import Path
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Layeerconsistentbinding.metrics import (
    compute_layerwise_ownership_summary,
    compute_multi_occurrence_stratified_metrics,
)


class TestLCBMetrics(unittest.TestCase):
    def _create_mock_records(self):
        # Sample 0: 2 GT occurrences (multi-occurrence). D1 correct for both, D4 correct for both.
        # Sample 1: 2 GT occurrences. D1 correct for both, D4 washes out for query 0 (dominant_gt 1 instead of 0).
        record_0 = {
            "qid": 101,
            "num_gt": 2,
            "clean_multi": True,
            "d1_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "d2_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "d3_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "d4_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "coverage": {"d4": {"coverage@5_50": 1.0, "coverage@5_70": 0.5}},
        }

        record_1 = {
            "qid": 102,
            "num_gt": 2,
            "clean_multi": True,
            "d1_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "d2_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 1.0,
                "ecr": 0.0,
                "own_mass": 0.8,
                "binding_margin": 0.6,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 0, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.8, "margin": 0.6, "target_enrichment": 1.5},
                ],
            },
            "d3_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 0.5,
                "ecr": 1.0,
                "own_mass": 0.4,
                "binding_margin": -0.2,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 1, "target_mass": 0.3, "margin": -0.2, "target_enrichment": 0.5},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.5, "margin": 0.2, "target_enrichment": 1.0},
                ],
            },
            "d4_final": {
                "num_matched": 2,
                "num_valid_pairs": 1,
                "aec": 0.5,
                "ecr": 1.0,
                "own_mass": 0.4,
                "binding_margin": -0.2,
                "matches": [
                    {"query": 0, "target": 0, "dominant_gt": 1, "target_mass": 0.2, "margin": -0.4, "target_enrichment": 0.3},
                    {"query": 1, "target": 1, "dominant_gt": 1, "target_mass": 0.6, "margin": 0.4, "target_enrichment": 1.1},
                ],
            },
            "coverage": {"d4": {"coverage@5_50": 0.5, "coverage@5_70": 0.0}},
        }
        return [record_0, record_1]

    def test_layerwise_ownership_summary(self):
        records = self._create_mock_records()
        summary = compute_layerwise_ownership_summary(records, decoder_layers=4)

        # Layer metrics
        layers = summary["layers"]
        self.assertEqual(layers["d1"]["aec"], 1.0)
        self.assertEqual(layers["d1"]["ecr"], 0.0)
        self.assertEqual(layers["d4"]["aec"], 0.75)  # mean of 1.0 and 0.5
        self.assertEqual(layers["d4"]["ecr"], 0.5)   # mean of 0.0 and 1.0

        # Persistence metrics
        persistence = summary["persistence"]
        self.assertEqual(persistence["total_matched_trajectories"], 4)
        self.assertEqual(persistence["d1_correct_count"], 4)
        self.assertEqual(persistence["d1_to_d2_persistence"], 1.0)
        self.assertEqual(persistence["d1_to_d3_persistence"], 0.75)  # 3 of 4 remained correct
        self.assertEqual(persistence["d1_to_d4_persistence"], 0.75)  # 3 of 4 remained correct

        # Washout drop rate (query 0 in record 1 dropped from 0.8 to 0.2, which is > 0.05 drop)
        self.assertGreater(persistence["d1_to_d4_washout_drop_rate"], 0.0)

    def test_multi_occurrence_stratified_metrics(self):
        records = self._create_mock_records()
        stratified = compute_multi_occurrence_stratified_metrics(records)

        self.assertIn("multi", stratified)
        self.assertIn("clean_multi", stratified)
        self.assertEqual(stratified["multi"]["num_records"], 2)
        self.assertEqual(stratified["multi"]["coverage@5_50"], 0.75)


if __name__ == "__main__":
    unittest.main()
