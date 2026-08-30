import math
import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Layeerconsistentbinding.controls import (
    compute_layer_consistent_binding_losses,
    install_layer_consistent_binding_control,
    js_divergence,
    normalize_cross_attention,
    remove_layer_consistent_binding_control,
)


class TestLCBControls(unittest.TestCase):
    def test_js_divergence_properties(self):
        # 1. Identical distributions -> JS = 0
        p = torch.tensor([[0.2, 0.5, 0.3]])
        js_zero = js_divergence(p, p)
        self.assertAlmostEqual(float(js_zero.item()), 0.0, places=5)

        # 2. Symmetry: JS(p, q) == JS(q, p)
        q = torch.tensor([[0.6, 0.1, 0.3]])
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        self.assertAlmostEqual(float(js_pq.item()), float(js_qp.item()), places=5)

        # 3. Disjoint / orthogonal distributions -> JS = ln(2)
        p_orth = torch.tensor([[1.0, 0.0]])
        q_orth = torch.tensor([[0.0, 1.0]])
        js_orth = js_divergence(p_orth, q_orth)
        self.assertAlmostEqual(float(js_orth.item()), math.log(2.0), places=4)

        # 4. Non-negativity
        p_rand = torch.rand(4, 5)
        q_rand = torch.rand(4, 5)
        js_rand = js_divergence(p_rand, q_rand)
        self.assertTrue((js_rand >= 0.0).all())

    def test_normalize_cross_attention(self):
        att = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1, 1, 4]
        mask = torch.tensor([[True, True, True, False]])  # [1, 4] valid length = 3
        norm = normalize_cross_attention(att, mask)
        self.assertEqual(norm.shape, (1, 1, 4))
        self.assertAlmostEqual(float(norm[0, 0, 3].item()), 0.0, places=6)
        self.assertAlmostEqual(float(norm[0, 0, :3].sum().item()), 1.0, places=5)

    def test_compute_lcb_losses_math_and_gradients(self):
        # B=1, Q=2, T=4
        # Two GT occurrences at [0.0, 0.5] (clips 0,1) and [0.5, 1.0] (clips 2,3)
        # Query 0 matched to GT 0; Query 1 matched to GT 1
        targets = {
            "span_labels": [{"spans": torch.tensor([[0.25, 0.5], [0.75, 0.5]])}],
        }
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        # 4 layers of attention
        # D1: Query 0 focuses on clips 0,1; Query 1 focuses on clips 2,3 (perfect ownership)
        att_d1 = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # D2: Query 0 keeps [0.5, 0.5, 0.0, 0.0]; Query 1 keeps [0.0, 0.0, 0.5, 0.5]
        att_d2 = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # D3: Same
        att_d3 = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # D4: Same
        att_d4 = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )

        outputs = {
            "pred_logits": torch.zeros(1, 2, 2),
            "video_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        losses = compute_layer_consistent_binding_losses(
            [att_d1, att_d2, att_d3, att_d4],
            outputs,
            targets,
            indices,
            span_loss_type="l1",
            drop_margin=0.05,
        )

        self.assertIn("loss_lcb_d1_bind", losses)
        self.assertIn("loss_lcb_late_bind", losses)
        self.assertIn("loss_lcb_owner_cons", losses)
        self.assertIn("loss_lcb_drop", losses)

        # When attention perfectly matches and is identical across layers:
        # 1. m_jk = 1.0 for all queries and layers -> L_D1-bind = -log(1.0) = 0.0
        self.assertAlmostEqual(float(losses["loss_lcb_d1_bind"].item()), 0.0, places=5)
        # 2. L_late-bind = -log(1.0) = 0.0
        self.assertAlmostEqual(float(losses["loss_lcb_late_bind"].item()), 0.0, places=5)
        # 3. p(D1) == p(D2..D4) -> L_owner_cons = 0.0
        self.assertAlmostEqual(float(losses["loss_lcb_owner_cons"].item()), 0.0, places=5)
        # 4. m(D1) == m(D2..D4) -> diff = -0.05 <= 0 -> L_drop = 0.0
        self.assertAlmostEqual(float(losses["loss_lcb_drop"].item()), 0.0, places=5)

    def test_d1_stopgrad_in_consistency_loss(self):
        # Verify that gradient of L_owner_cons does NOT flow to D1 (stopgrad property)
        targets = {
            "span_labels": [{"spans": torch.tensor([[0.25, 0.5], [0.75, 0.5]])}],
        }
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        att_d1 = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # D2 has drifted attention
        att_d2 = torch.tensor(
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = {
            "pred_logits": torch.zeros(1, 2, 2),
            "video_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        losses = compute_layer_consistent_binding_losses(
            [att_d1, att_d2],
            outputs,
            targets,
            indices,
            span_loss_type="l1",
            layers=(0, 1),
        )

        cons_loss = losses["loss_lcb_owner_cons"]
        self.assertGreater(float(cons_loss.item()), 0.0)

        cons_loss.backward()
        # Gradient on D1 must be None or zero due to stopgrad
        if att_d1.grad is not None:
            self.assertTrue(torch.allclose(att_d1.grad, torch.zeros_like(att_d1)))
        # Gradient on D2 must be non-zero
        self.assertIsNotNone(att_d2.grad)
        self.assertFalse(torch.allclose(att_d2.grad, torch.zeros_like(att_d2)))

    def test_anti_washout_hinge_loss(self):
        # D1 mass is high (1.0), D2 mass is washed out (0.2)
        # drop_margin = 0.05
        # Penalty should be (1.0 - 0.2 - 0.05)^2 = 0.75^2 = 0.5625
        targets = {
            "span_labels": [{"spans": torch.tensor([[0.25, 0.5]])}],
        }
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        att_d1 = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)  # mass = 1.0
        att_d2 = torch.tensor([[[0.1, 0.1, 0.4, 0.4]]], dtype=torch.float32)  # mass = 0.2

        outputs = {
            "pred_logits": torch.zeros(1, 1, 2),
            "video_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        losses = compute_layer_consistent_binding_losses(
            [att_d1, att_d2],
            outputs,
            targets,
            indices,
            span_loss_type="l1",
            layers=(0, 1),
            drop_margin=0.05,
        )

        expected_drop = (1.0 - 0.2 - 0.05) ** 2
        self.assertAlmostEqual(float(losses["loss_lcb_drop"].item()), expected_drop, places=4)

    def test_empty_match_safety(self):
        # No matches in batch
        outputs = {
            "pred_logits": torch.zeros(2, 10, 2, requires_grad=True),
            "video_mask": torch.ones(2, 10, dtype=torch.bool),
        }
        targets = {"span_labels": [{"spans": torch.empty((0, 2))}, {"spans": torch.empty((0, 2))}]}
        indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(2)]

        losses = compute_layer_consistent_binding_losses(
            [torch.zeros(2, 10, 10), torch.zeros(2, 10, 10)],
            outputs,
            targets,
            indices,
            span_loss_type="l1",
        )
        total_loss = (
            losses["loss_lcb_d1_bind"]
            + losses["loss_lcb_late_bind"]
            + losses["loss_lcb_owner_cons"]
            + losses["loss_lcb_drop"]
        )
        self.assertEqual(float(total_loss.item()), 0.0)
        total_loss.backward()  # Should not error

    def test_install_and_remove_control(self):
        class DummyMatcher:
            def __call__(self, outputs, targets):
                return [(torch.tensor([0]), torch.tensor([0]))]

        class DummyCriterion(nn.Module):
            def __init__(self):
                super().__init__()
                self.matcher = DummyMatcher()
                self.span_loss_type = "l1"
                self.weight_dict = {"loss_base": 1.0}

            def forward(self, outputs, targets):
                return {"loss_base": torch.tensor(1.0, requires_grad=True)}

        class DummyCapture:
            def get(self):
                return [
                    torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
                    torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
                ]

        criterion = DummyCriterion()
        capture = DummyCapture()

        install_layer_consistent_binding_control(
            criterion,
            capture,
            d1_bind_coef=0.5,
            late_bind_coef=0.1,
            owner_cons_coef=0.1,
            drop_coef=0.1,
            layers=(0, 1),
        )

        self.assertIn("loss_lcb_d1_bind", criterion.weight_dict)
        self.assertIn("loss_lcb_late_bind", criterion.weight_dict)
        self.assertIn("loss_lcb_owner_cons", criterion.weight_dict)
        self.assertIn("loss_lcb_drop", criterion.weight_dict)
        self.assertEqual(criterion.weight_dict["loss_lcb_d1_bind"], 0.5)
        self.assertEqual(criterion.weight_dict["loss_lcb_late_bind"], 0.1)

        outputs = {
            "pred_logits": torch.zeros(1, 1, 2),
            "video_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        targets = {"span_labels": [{"spans": torch.tensor([[0.25, 0.5]])}]}

        out_losses = criterion(outputs, targets)
        self.assertIn("loss_base", out_losses)
        self.assertIn("loss_lcb_d1_bind", out_losses)
        self.assertIn("loss_lcb_late_bind", out_losses)
        self.assertIn("loss_lcb_owner_cons", out_losses)
        self.assertIn("loss_lcb_drop", out_losses)

        # Removal
        remove_layer_consistent_binding_control(criterion)
        self.assertNotIn("loss_lcb_d1_bind", criterion.weight_dict)
        self.assertNotIn("loss_lcb_late_bind", criterion.weight_dict)
        out_losses_clean = criterion(outputs, targets)
        self.assertNotIn("loss_lcb_d1_bind", out_losses_clean)


if __name__ == "__main__":
    unittest.main()
