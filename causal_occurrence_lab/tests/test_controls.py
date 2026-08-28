import torch
import unittest

from causal_occurrence_lab.common import ensure_repo_on_path

ensure_repo_on_path()

from causal_occurrence_lab.controls import (  # noqa: E402
    controlled_query_cgp_loss,
    install_injection_control,
    remove_injection_control,
)
from sim_detr.dq_cgp.loss import compute_routing_loss, loss_query_cgp  # noqa: E402
from sim_detr.dq_cgp.query_cgp import DETRQueryCGP  # noqa: E402


def _targets():
    return {
        "span_labels": [{"spans": torch.tensor([[0.25, 0.25], [0.75, 0.25]])}],
    }


def _outputs():
    attention = torch.tensor(
        [[[0.7, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.7]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    routes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, requires_grad=True
    )
    return {
        "pred_logits": torch.zeros(1, 2, 2),
        "query_cgp_temporal_attention": attention,
        "query_cgp_basis_weights": routes,
        "query_cgp_video_mask": torch.ones(1, 4, dtype=torch.bool),
    }


class ControlTests(unittest.TestCase):
    def test_control_loss_matches_production_for_matched_target(self):
        outputs = _outputs()
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        expected = loss_query_cgp(outputs, _targets(), indices, span_loss_type="l1")
        actual = controlled_query_cgp_loss(
            outputs, _targets(), indices, span_loss_type="l1", binding_target="matched"
        )
        self.assertTrue(
            torch.allclose(actual["loss_query_cgp_bind"], expected["loss_query_cgp_bind"])
        )
        self.assertTrue(
            torch.allclose(actual["loss_query_cgp_route"], expected["loss_query_cgp_route"])
        )


    def test_production_route_objective_prefers_confident_global_diversity(self):
        diverse = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(
            torch.allclose(compute_routing_loss(diverse), torch.tensor(-0.69314718), atol=1e-5)
        )
        self.assertTrue(torch.allclose(compute_routing_loss(collapsed), torch.tensor(0.0), atol=1e-6))

    def test_control_route_loss_aggregates_matches_across_batch(self):
        attention = torch.full((2, 1, 2), 0.5)
        routes = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
        )
        outputs = {
            "pred_logits": torch.zeros(2, 1, 2),
            "query_cgp_temporal_attention": attention,
            "query_cgp_basis_weights": routes,
            "query_cgp_video_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.25, 0.5]])},
                {"spans": torch.tensor([[0.75, 0.5]])},
            ]
        }
        indices = [
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([0]), torch.tensor([0])),
        ]
        result = controlled_query_cgp_loss(
            outputs, targets, indices, span_loss_type="l1", binding_target="matched"
        )
        expected = compute_routing_loss(routes.reshape(-1, 2))
        self.assertTrue(torch.allclose(result["loss_query_cgp_route"], expected))


    def test_supervision_only_computes_diagnostics_but_returns_identity(self):
        module = DETRQueryCGP(
            hidden_dim=8, num_basis=3, prompt_length=2,
            router_hidden_dim=8, frf_hidden_dim=12, beta=0.05,
        )
        model = type("Model", (), {})()
        model.query_cgp = module
        install_injection_control(model, inject_residual=False)
        state = torch.randn(4, 2, 8)
        memory = torch.randn(6, 2, 8)
        padding = torch.zeros(2, 6, dtype=torch.bool)
        semantic = torch.randn(2, 8)
        result = module(state, memory, padding, semantic)
        self.assertTrue(torch.allclose(result, state))
        self.assertIsNotNone(module.last_output)
        self.assertEqual(module.last_output.temporal_attention.shape, (2, 4, 6))
        remove_injection_control(model)
