"""Hungarian-matched D1 temporal Binding Loss for QVHighlights."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from sim_detr.span_utils import span_cxw_to_xx


def span_overlap_mask(spans: Tensor, valid_length: int) -> Tensor:
    """Project normalized ``(center, width)`` spans onto the valid clip grid."""

    spans_xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
    starts = torch.arange(
        valid_length, dtype=spans.dtype, device=spans.device
    ) / float(valid_length)
    ends = starts + 1.0 / float(valid_length)
    overlap = (
        (starts.unsqueeze(0) < spans_xx[:, 1:])
        & (ends.unsqueeze(0) > spans_xx[:, :1])
    )
    empty = ~overlap.any(dim=1)
    if bool(empty.any()):
        centers = 0.5 * (starts + ends)
        nearest = (centers.unsqueeze(0) - spans_xx[:, :1]).abs().argmin(dim=1)
        overlap[empty] = False
        overlap[empty, nearest[empty]] = True
    return overlap


def matched_binding_loss(
    attention: Optional[Tensor],
    video_mask: Optional[Tensor],
    targets: Optional[Dict],
    indices: Optional[List[Tuple[Tensor, Tensor]]],
) -> Tensor:
    """Return ``-log`` D1 attention mass inside each matched GT window."""

    if attention is None:
        raise ValueError("D1 attention is required for Binding Loss")
    if video_mask is None or targets is None or indices is None:
        return attention.sum() * 0.0
    if "span_labels" not in targets:
        return attention.sum() * 0.0

    terms = []
    for batch_index, (source_indices, target_indices) in enumerate(indices):
        if source_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].bool().sum().item())
        if valid_length <= 0:
            continue
        source_indices = source_indices.to(attention.device)
        target_indices = target_indices.to(attention.device)
        spans = targets["span_labels"][batch_index]["spans"].to(attention.device)
        spans = spans[target_indices]
        positive = span_overlap_mask(spans, valid_length)
        matched_attention = attention[
            batch_index, source_indices, :valid_length
        ]
        mass = (matched_attention * positive.to(attention.dtype)).sum(dim=1)
        terms.append(-mass.clamp_min(torch.finfo(attention.dtype).eps).log())

    if not terms:
        return attention.sum() * 0.0
    return torch.cat(terms).mean()
