from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sim_detr.ls_dq_cgp.attention_capture import NativeD1AttentionCapture
from sim_detr.ls_dq_cgp.loss import binding_loss


class FakeAttention(nn.Module):
    def forward(self, weights):
        return weights.new_zeros(1), weights


def make_decoder(num_layers=2):
    layers = [SimpleNamespace(cross_attn=FakeAttention()) for _ in range(num_layers)]
    return SimpleNamespace(num_layers=num_layers, layers=layers)


class TestAttentionAndLoss(unittest.TestCase):
    def test_capture_masks_padding_and_renormalizes(self):
        decoder = make_decoder()
        capture = NativeD1AttentionCapture(decoder)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        capture.reset(mask)
        weights = torch.tensor([[[0.2, 0.3, 0.5]], [[0.1, 0.2, 0.7]]])
        decoder.layers[0].cross_attn(weights)
        normalized = capture.normalized()
        self.assertTrue(torch.allclose(normalized.sum(-1), torch.ones(2, 1)))
        self.assertEqual(normalized[0, 0, 2], 0)
        capture.close()

    def test_binding_loss_uses_each_samples_valid_length(self):
        attention = torch.zeros(2, 1, 10, requires_grad=True)
        with torch.no_grad():
            attention[0, 0, 3:5] = 0.5
            attention[1, 0, 7:9] = 0.5
        masks = torch.tensor([[1] * 5 + [0] * 5, [1] * 10], dtype=torch.bool)
        targets = {"span_labels": [
            {"spans": torch.tensor([[0.8, 0.2]])},
            {"spans": torch.tensor([[0.8, 0.2]])},
        ]}
        indices = [(torch.tensor([0]), torch.tensor([0])) for _ in range(2)]
        loss = binding_loss(attention, masks, targets, indices)
        self.assertLess(loss.item(), 1e-6)
        loss.backward()
        self.assertIsNotNone(attention.grad)
