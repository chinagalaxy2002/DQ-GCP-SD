import torch
import unittest

from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture


class _CrossAttention(torch.nn.Module):
    def forward(self, query, key, value, need_weights=True, **kwargs):
        del key, value, kwargs
        weights = torch.softmax(torch.ones(query.shape[1], query.shape[0], 3), dim=-1)
        return query, weights if need_weights else None


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _CrossAttention()


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()


class AttentionCaptureTests(unittest.TestCase):
    def test_capture_preserves_tuple_and_records_each_layer(self):
        model = _Model()
        capture = NativeCrossAttentionCapture(model).install()
        query = torch.randn(4, 2, 5)
        for layer in model.transformer.decoder.layers:
            layer.cross_attn(query, query, query)
        values = capture.get()
        self.assertEqual(
            [tuple(value.shape) for value in values], [(2, 4, 3), (2, 4, 3)]
        )
        capture.remove()
