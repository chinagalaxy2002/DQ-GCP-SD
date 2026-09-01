import unittest
import torch

from sim_detr.ls_dq_cgp.cgp_module import LateSemanticCGP


def make_module():
    return LateSemanticCGP(
        hidden_dim=16, num_basis=4, prompt_length=3,
        router_hidden_dim=12, frf_hidden_dim=24,
    )


class TestCGPModule(unittest.TestCase):
    def test_shapes_probabilities_and_finite_values(self):
        module = make_module()
        output = module(torch.randn(2, 5, 16), torch.randn(2, 16), torch.randn(2, 5, 16))
        self.assertEqual(output.adapted_semantic.shape, (2, 5, 16))
        self.assertEqual(output.basis_weights.shape, (2, 5, 4))
        self.assertEqual(output.pooled_prompt.shape, (2, 5, 16))
        self.assertEqual(output.pred_logits.shape, (2, 5, 2))
        self.assertTrue(torch.allclose(output.basis_weights.sum(-1), torch.ones(2, 5), atol=1e-6))
        self.assertTrue(torch.isfinite(output.pred_logits).all())

    def test_semantic_matching_cannot_backpropagate_through_visual_context(self):
        module = make_module()
        visual = torch.randn(2, 5, 16, requires_grad=True)
        output = module(visual, torch.randn(2, 16), torch.randn(2, 5, 16))
        output.pred_logits.sum().backward()
        self.assertIsNone(visual.grad)
        self.assertIsNotNone(module.router[0].weight.grad)
        self.assertIsNotNone(module.basis_prompts.grad)
        self.assertIsNotNone(module.frf[0].weight.grad)

    def test_static_bypass_changes_only_match_semantic_choice(self):
        module = make_module().eval()
        visual, static, queries = torch.randn(2, 5, 16), torch.randn(2, 16), torch.randn(2, 5, 16)
        active = module(visual, static, queries, static_bypass=False)
        bypass = module(visual, static, queries, static_bypass=True)
        self.assertTrue(torch.equal(active.adapted_semantic, bypass.adapted_semantic))
        self.assertFalse(torch.allclose(active.semantic_scores, bypass.semantic_scores))

    def test_bad_temperature_and_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            LateSemanticCGP(hidden_dim=16, temperature=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            make_module()(torch.randn(2, 5, 16), torch.randn(2, 16), torch.randn(2, 4, 16))
