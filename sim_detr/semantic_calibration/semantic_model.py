"""Native Sim-DETR wrapper for late candidate semantic calibration."""

from __future__ import annotations

import torch
from torch import nn

from .semantic_calibrator import (
    CandidateSemanticCalibrator,
    normalize_evidence_weights,
    pool_video_evidence,
    softmax_evidence_weights,
)
from .transformer_capture import TransformerOutputCapture
from .diagnostics import (
    farthest_candidate_context,
    random_deranged_context,
    roll_candidate_context,
)


CONTEXT_VARIANTS = {
    "aligned", "roll", "roll-1", "roll-2", "roll-3",
    "random-derangement", "farthest-context", "uniform",
}


class SimDETRWithSemanticCalibration(nn.Module):
    """Wrap a native Sim-DETR and replace only its final ``pred_logits``."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        semantic_scale_init: float = 1.0,
        semantic_variant: str = "full",
        detach_support: bool = True,
        diagnostic_mode: bool = False,
        evidence_source: str = "native_pred_mask",
    ):
        super().__init__()
        if semantic_variant not in {"native", "static", "full"}:
            raise ValueError(f"Unknown semantic_variant: {semantic_variant}")
        if evidence_source not in {"native_pred_mask", "native_mask_logits"}:
            raise ValueError(f"Unknown evidence_source: {evidence_source}")
        self.base_model = base_model
        self.semantic_calibrator = CandidateSemanticCalibrator(
            hidden_dim=hidden_dim,
            dropout=dropout,
            semantic_scale_init=semantic_scale_init,
        )
        self.transformer_capture = TransformerOutputCapture(base_model.transformer)
        self.semantic_variant = semantic_variant
        self.detach_support = bool(detach_support)
        self.diagnostic_mode = bool(diagnostic_mode)
        self.evidence_source = evidence_source
        self.semantic_context_variant = "aligned"
        self.semantic_counterfactual_seed = 2017
        self._counterfactual_sample_offset = 0
        self.semantic_scale_override = None

    @property
    def semantic_scale(self):
        return self.semantic_calibrator.semantic_scale

    def set_counterfactual(self, semantic_variant=None, context_variant=None):
        if semantic_variant is not None:
            if semantic_variant not in {"native", "static", "full"}:
                raise ValueError(f"Unknown semantic_variant: {semantic_variant}")
            self.semantic_variant = semantic_variant
        if context_variant is not None:
            if context_variant not in CONTEXT_VARIANTS:
                raise ValueError(f"Unknown context variant: {context_variant}")
            self.semantic_context_variant = context_variant
            self._counterfactual_sample_offset = 0

    def reset_counterfactual_state(self):
        self._counterfactual_sample_offset = 0

    def _native_mask_logits(self, query_states, video_memory):
        """Reconstruct the unmodified native mask field before sigmoid."""
        if query_states.shape[0] < 2:
            raise RuntimeError("Sim-DETR native mask uses the penultimate decoder layer")
        support_queries = query_states[-2]
        support_queries = support_queries + self.base_model.mask_head(support_queries)
        query_norm = torch.nn.functional.normalize(support_queries, p=2, dim=-1)
        video_norm = torch.nn.functional.normalize(video_memory, p=2, dim=-1)
        similarity = torch.bmm(query_norm, video_norm.transpose(1, 2))
        return self.base_model.logit_scale.exp() * similarity

    def forward(self, *args, **kwargs):
        native_outputs = self.base_model(*args, **kwargs)
        all_query_states, video_memory = self.transformer_capture.consume()
        native_logits = native_outputs["pred_logits"]
        native_outputs["pred_logits_native"] = native_logits

        if self.semantic_variant == "native":
            native_outputs["semantic_scores"] = None
            native_outputs["semantic_scale"] = native_logits.new_zeros(())
            return native_outputs

        if all_query_states.ndim != 4:
            raise RuntimeError(f"Expected decoder states [L,B,Q,D], got {all_query_states.shape}")
        query_states = all_query_states[-1]
        static_semantic = native_outputs["src_txt_cls_ed"]
        weights = None
        video_context = None
        if self.semantic_variant == "full":
            if self.evidence_source == "native_pred_mask":
                support = native_outputs["pred_masks"]
            else:
                support = self._native_mask_logits(all_query_states, video_memory)
            if self.detach_support:
                support = support.detach()

            video_mask = native_outputs.get("video_mask")
            if video_mask is None:
                raise KeyError("Native Sim-DETR output does not contain video_mask")
            if self.evidence_source == "native_pred_mask":
                weights = normalize_evidence_weights(support, video_mask)
            else:
                weights = softmax_evidence_weights(support, video_mask)
            if self.semantic_context_variant == "uniform":
                valid = video_mask.to(device=support.device, dtype=support.dtype).unsqueeze(1)
                weights = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
                weights = weights.expand(-1, support.shape[1], -1)
            video_context = pool_video_evidence(weights, video_memory)
            context_variant = self.semantic_context_variant
            if context_variant in {"roll", "roll-1", "roll-2", "roll-3"}:
                shift = 1 if context_variant == "roll" else int(context_variant.rsplit("-", 1)[1])
                video_context = roll_candidate_context(video_context, shift=shift)
            elif context_variant == "random-derangement":
                video_context, _ = random_deranged_context(
                    video_context,
                    seed=self.semantic_counterfactual_seed,
                    sample_offset=self._counterfactual_sample_offset,
                )
                self._counterfactual_sample_offset += video_context.shape[0]
            elif context_variant == "farthest-context":
                video_context, _ = farthest_candidate_context(video_context)

        semantic_output = self.semantic_calibrator(
            query_states=query_states,
            static_semantic=static_semantic,
            video_context=video_context,
            native_logits=native_logits,
            variant=self.semantic_variant,
            semantic_scale_override=self.semantic_scale_override,
        )
        native_outputs["pred_logits"] = semantic_output.pred_logits
        native_outputs["semantic_scores"] = semantic_output.semantic_scores
        native_outputs["semantic_scale"] = self.semantic_scale
        if self.diagnostic_mode:
            native_outputs["conditioned_semantics"] = semantic_output.conditioned_semantics
            native_outputs["semantic_delta"] = semantic_output.semantic_delta
            native_outputs["evidence_weights"] = weights
            native_outputs["video_evidence"] = video_context
        return native_outputs

    def close(self):
        self.transformer_capture.close()
