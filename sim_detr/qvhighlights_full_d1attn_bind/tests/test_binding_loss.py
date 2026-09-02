import unittest

import torch

from sim_detr.qvhighlights_full_d1attn_bind.binding_loss import (
    matched_binding_loss,
    span_overlap_mask,
)


class BindingLossTest(unittest.TestCase):
    def test_overlap_and_matched_mass(self):
        spans = torch.tensor([[0.25, 0.50]])
        overlap = span_overlap_mask(spans, 4)
        self.assertEqual(overlap.tolist(), [[True, True, False, False]])

        attention = torch.tensor([[[0.4, 0.4, 0.1, 0.1]]], requires_grad=True)
        mask = torch.ones(1, 4, dtype=torch.bool)
        targets = {"span_labels": [{"spans": spans}]}
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = matched_binding_loss(attention, mask, targets, indices)
        self.assertTrue(torch.allclose(loss, -torch.log(torch.tensor(0.8))))
        loss.backward()
        self.assertIsNotNone(attention.grad)

    def test_no_matches_returns_differentiable_zero(self):
        attention = torch.softmax(torch.randn(1, 2, 4, requires_grad=True), dim=-1)
        mask = torch.ones(1, 4, dtype=torch.bool)
        targets = {"span_labels": [{"spans": torch.empty(0, 2)}]}
        empty = torch.empty(0, dtype=torch.long)
        loss = matched_binding_loss(attention, mask, targets, [(empty, empty)])
        self.assertEqual(float(loss), 0.0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
