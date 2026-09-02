"""Full semantic calibration driven by native D1 cross-attention."""

from __future__ import annotations

from torch import nn

from sim_detr.semantic_calibration.semantic_calibrator import pool_video_evidence
from sim_detr.semantic_calibration.transformer_capture import TransformerOutputCapture

from .attention_capture import NativeD1AttentionCapture


class QVHighlightsFullD1BindingModel(nn.Module):
    """Re-pool Full semantic evidence from D1 attention and expose it to loss."""

    def __init__(self, semantic_model: nn.Module) -> None:
        super().__init__()
        self.semantic_model = semantic_model
        transformer = semantic_model.base_model.transformer
        self.transformer_capture = TransformerOutputCapture(transformer)
        self.d1_capture = NativeD1AttentionCapture(transformer)

    def forward(self, *args, **kwargs):
        outputs = self.semantic_model(*args, **kwargs)
        decoder_states, video_memory = self.transformer_capture.consume()
        video_mask = outputs.get("video_mask")
        if video_mask is None:
            raise KeyError("Native Sim-DETR output does not contain video_mask")
        d1_attention = self.d1_capture.normalized_attention(video_mask)
        outputs["d1_attention"] = d1_attention

        video_context = pool_video_evidence(d1_attention, video_memory)
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

    def close(self) -> None:
        self.d1_capture.close()
        self.transformer_capture.close()
        self.semantic_model.close()
