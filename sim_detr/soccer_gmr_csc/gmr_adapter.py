"""Query-level existence adapter and D1 attention capture for Soccer-GMR."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

from sim_detr.semantic_calibration.semantic_calibrator import pool_video_evidence
from sim_detr.semantic_calibration.transformer_capture import TransformerOutputCapture

from .attention_capture import NativeD1AttentionCapture


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
    """Wrap Sim-DETR CSC variant with GMR existence head and D1 attention binding capture."""

    def __init__(
        self,
        semantic_model,
        hidden_dim: int = 256,
        exist_pool: str = "max",
        evidence_source: str = "native_mask_logits",
        binding_loss_coef: float = 0.0,
    ):
        super().__init__()
        self.semantic_model = semantic_model
        self.evidence_source = evidence_source
        self.binding_loss_coef = float(binding_loss_coef)
        self.exist_head = GMRAdapter(hidden_dim=hidden_dim, pool=exist_pool)
        self.exist_capture = TransformerOutputCapture(semantic_model.base_model.transformer)
        self.d1_capture = NativeD1AttentionCapture(semantic_model.base_model.transformer)

    def forward(self, *args, **kwargs):
        outputs = self.semantic_model(*args, **kwargs)
        decoder_states, video_memory = self.exist_capture.consume()
        outputs["pred_exist_logits"] = self.exist_head(decoder_states[-1])

        # Capture and attach normalized D1 cross-attention for binding loss / pooling
        video_mask = outputs.get("video_mask")
        d1_attn = self.d1_capture.normalized_attention(video_mask)
        outputs["d1_attention"] = d1_attn

        # If evidence_source is d1_attention and variant is full, re-pool evidence using bound D1 attention
        if self.semantic_model.semantic_variant == "full" and self.evidence_source == "d1_attention":
            video_context = pool_video_evidence(d1_attn, video_memory)
            semantic_output = self.semantic_model.semantic_calibrator(
                query_states=decoder_states[-1],
                static_semantic=outputs["src_txt_cls_ed"],
                video_context=video_context,
                native_logits=outputs["pred_logits_native"],
                variant="full",
                semantic_scale_override=self.semantic_model.semantic_scale_override,
            )
            outputs["pred_logits"] = semantic_output.pred_logits
            outputs["semantic_scores"] = semantic_output.semantic_scores

        return outputs

    def set_counterfactual(self, semantic_variant=None, context_variant=None):
        self.semantic_model.set_counterfactual(semantic_variant, context_variant)

    @property
    def semantic_calibrator(self):
        return self.semantic_model.semantic_calibrator

    def close(self):
        self.d1_capture.close()
        self.exist_capture.close()
        self.semantic_model.close()
