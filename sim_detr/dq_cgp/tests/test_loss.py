"""Tests for DQ-CGP loss functions (binding + routing).

Run from the Sim-DETR project root:
    PYTHONPATH=. python -m pytest sim_detr/dq_cgp/tests/test_loss.py -v
"""

import torch
from torch import nn

from sim_detr.dq_cgp.loss import (
    compute_binding_loss,
    compute_routing_loss,
    loss_query_cgp,
)
from sim_detr.dq_cgp.model_builder import DQCGPSetCriterion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BSZ = 2
NQ  = 10
TV  = 75
NK  = 16


def _make_uniform_attention():
    """All attention mass uniformly spread over valid frames."""
    attn = torch.ones(BSZ, NQ, TV) / TV
    return attn


def _make_gt_targets(windows):
    """Create Sim-DETR's real dict-of-batched-fields target contract.

    windows: list of (center, width) in normalised [0,1] coords.
    """
    spans = torch.tensor([[c, w] for c, w in windows], dtype=torch.float32)
    return {
        "span_labels": [
            {"spans": spans.clone()}
            for _ in range(BSZ)
        ]
    }


def _uniform_indices(num_gt):
    """Match query 0..num_gt-1 to GT 0..num_gt-1."""
    idx = torch.arange(num_gt)
    return [(idx, idx)] * BSZ


# ---------------------------------------------------------------------------
# Binding loss
# ---------------------------------------------------------------------------

class TestBindingLoss:
    def test_zero_mass_in_window_gives_large_loss(self):
        """Attention entirely OUTSIDE GT window → loss > large threshold."""
        attn = torch.zeros(BSZ, NQ, TV)
        # Put all mass in second half, GT window is in first half
        attn[:, 0, :TV // 2] = 0.0
        attn[:, 0, TV // 2:] = 1.0 / (TV - TV // 2)

        targets = _make_gt_targets([(0.1, 0.1)])   # 0–15% of video
        indices = _uniform_indices(1)
        video_mask = torch.ones(BSZ, TV)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() > 2.0, f"Expected large loss, got {loss.item():.4f}"

    def test_perfect_mass_in_window_gives_small_loss(self):
        """All attention mass INSIDE GT window → loss ≈ 0."""
        attn = torch.zeros(BSZ, NQ, TV)
        # GT window: center=0.2, width=0.2 → frames 7..22
        st, ed = int(0.1 * TV), int(0.3 * TV)
        n_frames = ed - st
        attn[:, 0, st:ed] = 1.0 / n_frames

        targets = _make_gt_targets([(0.2, 0.2)])
        indices = _uniform_indices(1)
        video_mask = torch.ones(BSZ, TV)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() < 0.1, f"Expected small loss, got {loss.item():.4f}"

    def test_loss_is_finite(self):
        attn = torch.softmax(torch.randn(BSZ, NQ, TV), dim=-1)
        targets = _make_gt_targets([(0.5, 0.3)])
        indices = _uniform_indices(1)
        video_mask = torch.ones(BSZ, TV)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_multi_window_candidate_specificity(self):
        """Two different matched queries attend to their own GT windows."""
        attn = torch.zeros(BSZ, NQ, TV)
        # Query 0 attends to window 0 (center=0.1, width=0.2 → frames 0-15)
        st0, ed0 = 0, int(0.2 * TV)
        attn[:, 0, st0:ed0] = 1.0 / (ed0 - st0)
        # Query 1 attends to window 1 (center=0.7, width=0.2 → frames 45-60)
        st1, ed1 = int(0.6 * TV), int(0.8 * TV)
        attn[:, 1, st1:ed1] = 1.0 / (ed1 - st1)

        targets = _make_gt_targets([(0.1, 0.2), (0.7, 0.2)])
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))] * BSZ
        video_mask = torch.ones(BSZ, TV)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() < 0.1, \
            f"Both queries attend correctly, loss should be small: {loss.item():.4f}"

    def test_empty_indices_returns_zero(self):
        attn = torch.softmax(
            torch.randn(BSZ, NQ, TV), dim=-1
        ).requires_grad_()
        targets = _make_gt_targets([(0.5, 0.2)])
        indices = [
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            for _ in range(BSZ)
        ]
        video_mask = torch.ones(BSZ, TV)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert attn.grad is not None

    def test_uses_each_samples_unpadded_valid_length(self):
        """A short sample must map normalized GT spans using its own length."""
        attn = torch.zeros(BSZ, NQ, 10)
        # Sample 0 has five valid clips. GT [0.7, 0.9] overlaps clips 3 and 4.
        attn[0, 0, 3:5] = 0.5
        # Sample 1 has ten valid clips. The same GT overlaps clips 7 and 8.
        attn[1, 0, 7:9] = 0.5
        video_mask = torch.tensor(
            [[1] * 5 + [0] * 5, [1] * 10], dtype=torch.bool
        )
        targets = _make_gt_targets([(0.8, 0.2)])
        indices = _uniform_indices(1)

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() < 1e-6

    def test_padding_only_wrong_window_is_not_silently_skipped(self):
        """Regression: padded-T coordinate mapping previously returned zero."""
        attn = torch.zeros(BSZ, NQ, 10)
        attn[0, 0, 0] = 1.0  # completely outside sample-0 GT
        video_mask = torch.tensor(
            [[1] * 5 + [0] * 5, [1] * 10], dtype=torch.bool
        )
        targets = _make_gt_targets([(0.8, 0.2)])
        empty = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        indices = [(torch.tensor([0]), torch.tensor([0])), empty]

        loss = compute_binding_loss(attn, video_mask, targets, indices)
        assert loss.item() > 10.0

    def test_ce_targets_use_inclusive_clip_indices(self):
        attn = torch.zeros(BSZ, NQ, 8)
        attn[:, 0, 2:4] = 0.5
        video_mask = torch.ones(BSZ, 8, dtype=torch.bool)
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[2, 3]], dtype=torch.long)}
                for _ in range(BSZ)
            ]
        }
        loss = compute_binding_loss(
            attn,
            video_mask,
            targets,
            _uniform_indices(1),
            span_loss_type="ce",
        )
        assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# Routing loss
# ---------------------------------------------------------------------------

class TestRoutingLoss:
    def test_collapsed_routing_gives_negative_loss(self):
        """All queries use same basis → H(W|C) is 0, H(W) is 0 → loss = 0."""
        # Perfectly collapsed: all mass on basis 0
        bw = torch.zeros(BSZ, NQ, NK)
        bw[:, :, 0] = 1.0
        loss = compute_routing_loss(bw)
        assert abs(loss.item()) < 0.1, \
            f"Collapsed routing loss should be ≈ 0, got {loss.item():.4f}"

    def test_diverse_routing_reduces_loss(self):
        """Diverse uniform routing should produce lower loss than collapsed."""
        # Collapsed
        bw_collapsed = torch.zeros(BSZ, NQ, NK)
        bw_collapsed[:, :, 0] = 1.0

        # Each query uses a different basis (perfectly diverse)
        bw_diverse = torch.zeros(BSZ, NQ, NK)
        for q in range(NQ):
            bw_diverse[:, q, q % NK] = 1.0

        loss_collapsed = compute_routing_loss(bw_collapsed)
        loss_diverse   = compute_routing_loss(bw_diverse)
        assert loss_diverse.item() < loss_collapsed.item(), \
            "Diverse routing must yield a lower loss than collapsed routing"

    def test_loss_is_finite(self):
        bw = torch.softmax(torch.randn(BSZ, NQ, NK), dim=-1)
        loss = compute_routing_loss(bw)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Combined loss_query_cgp
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    def _make_outputs(self, attn_val=None):
        attn = (
            torch.softmax(torch.randn(BSZ, NQ, TV), dim=-1)
            if attn_val is None
            else attn_val
        )
        return {
            "pred_logits": torch.randn(BSZ, NQ, 2),
            "query_cgp_temporal_attention": attn,
            "query_cgp_basis_weights": torch.softmax(
                torch.randn(BSZ, NQ, NK), dim=-1
            ),
            "query_cgp_video_mask": torch.ones(BSZ, TV, dtype=torch.bool),
        }

    def test_returns_two_keys(self):
        outputs = self._make_outputs()
        targets = _make_gt_targets([(0.5, 0.3)])
        indices = _uniform_indices(1)
        result = loss_query_cgp(outputs, targets, indices)
        assert "loss_query_cgp_bind" in result
        assert "loss_query_cgp_route" in result

    def test_both_losses_finite(self):
        outputs = self._make_outputs()
        targets = _make_gt_targets([(0.5, 0.3)])
        indices = _uniform_indices(1)
        result = loss_query_cgp(outputs, targets, indices)
        assert torch.isfinite(result["loss_query_cgp_bind"])
        assert torch.isfinite(result["loss_query_cgp_route"])

    def test_missing_outputs_returns_zeros(self):
        """If DQ-CGP was inactive (beta=0), outputs lacks the keys → return 0."""
        pred_logits = torch.zeros(BSZ, NQ, 2, requires_grad=True)
        result = loss_query_cgp(
            outputs={"pred_logits": pred_logits},
            targets=_make_gt_targets([(0.5, 0.2)]),
            indices=_uniform_indices(1),
        )
        assert result["loss_query_cgp_bind"].item() == 0.0
        assert result["loss_query_cgp_route"].item() == 0.0
        zero = result["loss_query_cgp_bind"] + result["loss_query_cgp_route"]
        assert zero.requires_grad
        zero.backward()
        assert pred_logits.grad is not None

    def test_route_loss_ignores_unmatched_queries(self):
        outputs_a = self._make_outputs()
        outputs_b = {k: v.clone() for k, v in outputs_a.items()}

        # Two matched positives deliberately specialize to different bases.
        for outputs in (outputs_a, outputs_b):
            outputs["query_cgp_basis_weights"].zero_()
            outputs["query_cgp_basis_weights"][0, 0, 0] = 1.0
            outputs["query_cgp_basis_weights"][1, 0, 1] = 1.0

        # Give unmatched queries radically different routing in each copy.
        outputs_a["query_cgp_basis_weights"][:, 1:, 2] = 1.0
        outputs_b["query_cgp_basis_weights"][:, 1:, 7] = 1.0
        indices = [
            (torch.tensor([0]), torch.tensor([0]))
            for _ in range(BSZ)
        ]
        targets = _make_gt_targets([(0.5, 0.3)])

        route_a = loss_query_cgp(outputs_a, targets, indices)[
            "loss_query_cgp_route"
        ]
        route_b = loss_query_cgp(outputs_b, targets, indices)[
            "loss_query_cgp_route"
        ]
        assert torch.allclose(route_a, route_b)
        assert route_a.item() < -0.6

    def test_no_matches_returns_graph_connected_diagnostic_zeros(self):
        outputs = self._make_outputs()
        outputs["query_cgp_temporal_attention"].requires_grad_()
        outputs["query_cgp_basis_weights"].requires_grad_()
        empty_indices = [
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            for _ in range(BSZ)
        ]
        result = loss_query_cgp(
            outputs, _make_gt_targets([(0.5, 0.2)]), empty_indices
        )
        total = result["loss_query_cgp_bind"] + result["loss_query_cgp_route"]
        assert total.item() == 0.0
        assert total.requires_grad
        total.backward()
        assert outputs["query_cgp_temporal_attention"].grad is not None
        assert outputs["query_cgp_basis_weights"].grad is not None


class _StaticMatcher(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, outputs, targets):
        del outputs, targets
        return self.indices


class TestCriterionIntegration:
    def test_real_targets_and_aux_skip(self):
        indices = _uniform_indices(1)
        criterion = DQCGPSetCriterion(
            matcher=_StaticMatcher(indices),
            weight_dict={
                "loss_query_cgp_bind": 0.2,
                "loss_query_cgp_route": 0.01,
            },
            eos_coef=0.1,
            losses=["query_cgp"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=TV,
            use_matcher=True,
            clip_len=2,
        )
        outputs = TestCombinedLoss()._make_outputs()
        outputs["aux_outputs"] = [
            {"pred_logits": torch.zeros(BSZ, NQ, 2)}
        ]
        losses = criterion(outputs, _make_gt_targets([(0.5, 0.3)]))
        assert set(losses) == {
            "loss_query_cgp_bind",
            "loss_query_cgp_route",
        }
