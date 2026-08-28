import torch
import unittest

from causal_occurrence_lab.common import normalize_state_dict


class EquivalenceTests(unittest.TestCase):
    def test_stripped_state_removes_only_query_cgp_namespace(self):
        state = {
            "transformer.decoder.layers.0.weight": torch.ones(2),
            "query_cgp.router.weight": torch.ones(2),
            "module.query_cgp.beta": torch.tensor(0.05),
        }
        normalized = normalize_state_dict(state)
        stripped = {
            key: value
            for key, value in normalized.items()
            if not key.startswith("query_cgp.")
        }
        self.assertEqual(list(stripped), ["transformer.decoder.layers.0.weight"])
