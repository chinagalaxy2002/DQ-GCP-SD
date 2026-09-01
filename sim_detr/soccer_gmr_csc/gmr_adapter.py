"""Query-level existence adapter following Moment-DETR-GMR."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class GMRAdapter(nn.Module):
    def __init__(self, hidden_dim=256, pool="max"):
        super().__init__()
        self.pool = pool
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        ])

    def forward(self, decoder_queries):
        if self.pool == "mean":
            value = decoder_queries.mean(dim=1)
        else:
            value = decoder_queries.max(dim=1).values
        return self.layers[1](F.relu(self.layers[0](value))).squeeze(-1)


class SoccerGMRModel(nn.Module):
    """Add the same existence head to every native/static/full CSC variant."""

    def __init__(self, semantic_model, hidden_dim=256, exist_pool="max"):
        super().__init__()
        from sim_detr.semantic_calibration.transformer_capture import TransformerOutputCapture

        self.semantic_model = semantic_model
        self.exist_head = GMRAdapter(hidden_dim=hidden_dim, pool=exist_pool)
        self.exist_capture = TransformerOutputCapture(semantic_model.base_model.transformer)

    def forward(self, *args, **kwargs):
        outputs = self.semantic_model(*args, **kwargs)
        decoder_states, _ = self.exist_capture.consume()
        outputs["pred_exist_logits"] = self.exist_head(decoder_states[-1])
        return outputs

    def set_counterfactual(self, semantic_variant=None, context_variant=None):
        self.semantic_model.set_counterfactual(semantic_variant, context_variant)

    @property
    def semantic_calibrator(self):
        return self.semantic_model.semantic_calibrator

    def close(self):
        self.exist_capture.close()
        self.semantic_model.close()
