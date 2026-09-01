import os
import tempfile
import unittest

import torch
from torch import nn

from sim_detr.semantic_calibration.diagnostics import roll_candidate_context
from sim_detr.semantic_calibration.model_builder import load_semantic_checkpoint
from sim_detr.semantic_calibration.semantic_calibrator import (
    CandidateSemanticCalibrator,
    normalize_evidence_weights,
    pool_video_evidence,
)
from sim_detr.semantic_calibration.semantic_model import SimDETRWithSemanticCalibration


class FakeTransformer(nn.Module):
    def __init__(self, hidden_dim=4, queries=3, video_length=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.queries = queries
        self.video_length = video_length

    def forward(self):
        hs = torch.arange(2 * 1 * self.queries * self.hidden_dim, dtype=torch.float32)
        hs = hs.reshape(2, 1, self.queries, self.hidden_dim)
        memory = torch.arange(1 * self.video_length * self.hidden_dim, dtype=torch.float32)
        memory = memory.reshape(1, self.video_length, self.hidden_dim)
        return hs, None, memory, torch.ones(1, self.video_length)


class FakeBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        self.transformer()
        return {
            "pred_logits": self.weight * torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
            "pred_spans": torch.tensor([[[0.1, 0.2], [0.3, 0.2], [0.5, 0.1]]]),
            "pred_masks": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0, 0.0]]]),
            "iou_scores": torch.ones(1, 3, 1),
            "saliency_scores": torch.ones(1, 5),
            "aux_outputs": [{"pred_logits": torch.ones(1, 3, 2)}],
            "src_txt_cls_ed": torch.ones(1, 4),
            "video_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
        }


class SemanticCalibrationTest(unittest.TestCase):
    def test_evidence_pooling_masks_padding_and_one_hot(self):
        support = torch.tensor([[[1.0, 2.0, 0.0, 100.0, 100.0]]])
        valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        weights = normalize_evidence_weights(support, valid)
        self.assertTrue(torch.all(weights[..., 3:] == 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 1)))
        context = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        one_hot = torch.zeros_like(support)
        one_hot[..., 1] = 1
        pooled = pool_video_evidence(normalize_evidence_weights(one_hot, valid), context)
        self.assertTrue(torch.equal(pooled[0, 0], context[0, 1]))

    def test_identity_native_and_scale_zero(self):
        native = FakeBaseModel()
        wrapper = SimDETRWithSemanticCalibration(native, hidden_dim=4, semantic_variant="native")
        wrapper.eval()
        output = wrapper()
        self.assertTrue(torch.equal(output["pred_logits"], output["pred_logits_native"]))
        self.assertIsNone(output["semantic_scores"])
        self.assertTrue(torch.equal(output["pred_spans"], torch.tensor([[[0.1, 0.2], [0.3, 0.2], [0.5, 0.1]]])))

        base = FakeBaseModel()
        full = SimDETRWithSemanticCalibration(base, hidden_dim=4, semantic_variant="full")
        full.eval()
        full.semantic_scale_override = 0.0
        output = full()
        self.assertTrue(torch.equal(output["pred_logits"], output["pred_logits_native"]))

    def test_static_semantics_are_identical_across_candidates(self):
        module = CandidateSemanticCalibrator(hidden_dim=4)
        query = torch.randn(2, 3, 4)
        semantic = torch.randn(2, 4)
        logits = torch.randn(2, 3, 2)
        output = module(query, semantic, torch.randn(2, 3, 4), logits, variant="static")
        self.assertTrue(torch.equal(output.conditioned_semantics[:, 0], output.conditioned_semantics[:, 1]))
        self.assertFalse(torch.equal(output.semantic_scores[:, 0], output.semantic_scores[:, 1]))

    def test_context_roll(self):
        context = torch.tensor([[[1.0], [2.0], [3.0]]])
        rolled = roll_candidate_context(context)
        self.assertTrue(torch.equal(rolled, torch.tensor([[[3.0], [1.0], [2.0]]])))

    def test_uniform_context_expands_to_every_candidate(self):
        wrapper = SimDETRWithSemanticCalibration(
            FakeBaseModel(), hidden_dim=4, semantic_variant="full", diagnostic_mode=True
        )
        wrapper.semantic_context_variant = "uniform"
        output = wrapper()
        weights = output["evidence_weights"]
        self.assertEqual(tuple(weights.shape), (1, 3, 5))
        self.assertTrue(torch.equal(weights[:, 0], weights[:, 1]))
        self.assertTrue(torch.equal(weights[:, 1], weights[:, 2]))
        self.assertTrue(torch.all(weights[..., 3:] == 0))

    def test_gradients_and_checkpoint_reload(self):
        wrapper = SimDETRWithSemanticCalibration(FakeBaseModel(), hidden_dim=4, semantic_variant="full")
        output = wrapper()
        loss = output["pred_logits"][..., 0].sum()
        loss.backward()
        self.assertIsNotNone(wrapper.semantic_calibrator.semantic_delta[0].weight.grad)
        self.assertIsNotNone(wrapper.semantic_calibrator.q_proj.weight.grad)
        self.assertIsNotNone(wrapper.semantic_calibrator.e_proj.weight.grad)
        self.assertIsNotNone(wrapper.semantic_calibrator.raw_semantic_scale.grad)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt")
            torch.save({"model": wrapper.state_dict(), "epoch": 3}, path)
            restored = SimDETRWithSemanticCalibration(FakeBaseModel(), hidden_dim=4, semantic_variant="full")
            checkpoint = load_semantic_checkpoint(restored, path)
            self.assertEqual(checkpoint["epoch"], 3)

    def test_native_bypass_has_no_semantic_gradient(self):
        wrapper = SimDETRWithSemanticCalibration(FakeBaseModel(), hidden_dim=4, semantic_variant="native")
        output = wrapper()
        output["pred_logits"].sum().backward()
        self.assertIsNone(wrapper.semantic_calibrator.q_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
