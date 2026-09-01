"""Candidate-conditioned semantic residual calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SemanticCalibrationOutput:
    pred_logits: torch.Tensor
    semantic_scores: torch.Tensor
    conditioned_semantics: torch.Tensor
    semantic_delta: torch.Tensor


def normalize_evidence_weights(
    support: torch.Tensor,
    valid_video_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mask and normalize native candidate temporal support.

    Args:
        support: ``[B, Q, T]`` non-negative native support, usually
            ``outputs['pred_masks']``.
        valid_video_mask: ``[B, T]`` with 1/True for real clips.

    Degenerate all-zero supports use a uniform distribution over valid clips;
    this preserves the intended probability-simplex invariant while keeping
    padded positions exactly zero.
    """
    if support.ndim != 3 or valid_video_mask.ndim != 2:
        raise ValueError("support must be [B,Q,T] and mask must be [B,T]")
    if support.shape[0] != valid_video_mask.shape[0] or support.shape[2] != valid_video_mask.shape[1]:
        raise ValueError("support and valid_video_mask have incompatible shapes")

    valid = valid_video_mask.to(device=support.device, dtype=support.dtype).unsqueeze(1)
    masked = support.clamp_min(0) * valid
    denom = masked.sum(dim=-1, keepdim=True)
    normalized = masked / denom.clamp_min(eps)

    valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    fallback = valid / valid_count
    degenerate = denom <= eps
    return torch.where(degenerate, fallback, normalized)


def pool_video_evidence(weights: torch.Tensor, video_context: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,T,D]`` video memory using ``[B,Q,T]`` evidence weights."""
    if weights.ndim != 3 or video_context.ndim != 3:
        raise ValueError("weights must be [B,Q,T] and video_context must be [B,T,D]")
    if weights.shape[0] != video_context.shape[0] or weights.shape[2] != video_context.shape[1]:
        raise ValueError("weights and video_context have incompatible shapes")
    return torch.bmm(weights, video_context)


class CandidateSemanticCalibrator(nn.Module):
    """Add candidate-specific semantic evidence as a foreground residual."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        semantic_scale_init: float = 1.0,
    ):
        super().__init__()
        if semantic_scale_init <= 0:
            raise ValueError("semantic_scale_init must be positive")
        self.hidden_dim = int(hidden_dim)
        self.semantic_delta = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.e_condition_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.e_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.raw_semantic_scale = nn.Parameter(
            torch.tensor(self._inverse_softplus(float(semantic_scale_init)))
        )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        import math
        return math.log(math.expm1(value)) if value < 20 else value

    @property
    def semantic_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_semantic_scale)

    def forward(
        self,
        query_states: torch.Tensor,
        static_semantic: torch.Tensor,
        video_context: Optional[torch.Tensor],
        native_logits: torch.Tensor,
        variant: str = "full",
        semantic_scale_override: Optional[float] = None,
    ) -> SemanticCalibrationOutput:
        if variant not in {"native", "static", "full"}:
            raise ValueError(f"Unknown semantic variant: {variant}")
        if native_logits.shape[-1] != 2:
            raise ValueError("Sim-DETR semantic calibration expects two class logits")
        if variant == "native":
            return SemanticCalibrationOutput(
                pred_logits=native_logits,
                semantic_scores=native_logits[..., 0].new_zeros(native_logits.shape[:-1]),
                conditioned_semantics=static_semantic.unsqueeze(1).expand_as(query_states),
                semantic_delta=torch.zeros_like(query_states),
            )

        if query_states.ndim != 3 or static_semantic.ndim != 2:
            raise ValueError("query_states must be [B,Q,D], static_semantic must be [B,D]")
        if query_states.shape[0] != static_semantic.shape[0]:
            raise ValueError("query_states and static_semantic batch sizes differ")

        static = static_semantic.unsqueeze(1).expand(-1, query_states.shape[1], -1)
        if variant == "static":
            delta = torch.zeros_like(static)
            conditioned = static
        else:
            if video_context is None:
                raise ValueError("full semantic calibration requires video_context")
            if video_context.shape != query_states.shape:
                raise ValueError("video_context must have the same shape as query_states")
            condition_input = torch.cat(
                [F.layer_norm(static, (self.hidden_dim,)),
                 F.layer_norm(video_context, (self.hidden_dim,))], dim=-1
            )
            delta = self.semantic_delta(condition_input)
            conditioned = self.e_condition_norm(static + delta)

        q_normalized = F.normalize(self.q_proj(query_states), p=2, dim=-1)
        e_normalized = F.normalize(self.e_proj(conditioned), p=2, dim=-1)
        semantic_scores = (q_normalized * e_normalized).sum(dim=-1)

        scale = self.semantic_scale if semantic_scale_override is None else native_logits.new_tensor(semantic_scale_override)
        if semantic_scale_override is not None and float(semantic_scale_override) == 0.0:
            calibrated_logits = native_logits
        else:
            calibrated_logits = native_logits.clone()
            calibrated_logits[..., 0] = calibrated_logits[..., 0] + scale * semantic_scores

        return SemanticCalibrationOutput(
            pred_logits=calibrated_logits,
            semantic_scores=semantic_scores,
            conditioned_semantics=conditioned,
            semantic_delta=delta,
        )
