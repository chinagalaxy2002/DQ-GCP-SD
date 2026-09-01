"""Hungarian-matched D1 temporal binding loss for Soccer-GMR.

References production-verified formulation in DQ-CGP-github-publish/ls_dq_cgp_lab:
1. Projects ground-truth spans to frame/clip overlap boolean masks.
2. Computes the matched Hungarian query attention mass over the positive time span.
3. Supervised via negative log-likelihood: -log(sum_{t in GT} Attn_1(q_i, t)).
4. Null-safe: returns exact 0.0 tensor with preserved gradient graph when batch is all-null.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor

from sim_detr.span_utils import span_cxw_to_xx


def span_overlap_mask(
    spans: Tensor,
    valid_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Compute binary overlap matrix between normalized spans and video clip grid.

    Args:
        spans: [N, 2] in normalized (cx, w) format.
        valid_length: number of valid video clips T.
    Returns:
        overlap: [N, valid_length] bool tensor.
    """
    xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
    starts = torch.arange(valid_length, dtype=dtype, device=device) / float(valid_length)
    ends = starts + 1.0 / float(valid_length)
    overlap = (starts.unsqueeze(0) < xx[:, 1:]) & (ends.unsqueeze(0) > xx[:, :1])
    empty = ~overlap.any(dim=1)
    if bool(empty.any()):
        centers = 0.5 * (starts + ends)
        nearest = (centers.unsqueeze(0) - xx[:, :1]).abs().argmin(dim=1)
        overlap[empty] = False
        overlap[empty, nearest[empty]] = True
    return overlap


def native_matched_binding_loss(
    attention: Optional[Tensor],
    video_mask: Optional[Tensor],
    targets: Dict,
    indices: Optional[List[Tuple[Tensor, Tensor]]],
) -> Tensor:
    """Production-equivalent Hungarian matched GT-mass loss on D1 cross-attention.

    Args:
        attention: [B, Q, T] D1 cross-attention weights.
        video_mask: [B, T] valid video clip mask.
        targets: dict with 'span_labels' and 'exist_label'.
        indices: Hungarian matching output [(src_idx, tgt_idx), ...].
    """
    if attention is None or video_mask is None or indices is None or 'span_labels' not in targets:
        return attention.sum() * 0.0 if attention is not None else torch.tensor(0.0)

    terms = []
    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].bool().sum().item())
        if valid_length <= 0:
            continue
        src_indices = src_indices.to(attention.device)
        target_indices = target_indices.to(attention.device)
        spans = targets['span_labels'][batch_index]['spans'].to(attention.device)[target_indices]
        if spans.numel() == 0:
            continue
        positive = span_overlap_mask(spans, valid_length, attention.dtype, attention.device)
        matched_attn = attention[batch_index, src_indices, :valid_length]
        mass = (matched_attn * positive.to(attention.dtype)).sum(dim=1)
        terms.append(-mass.clamp_min(torch.finfo(attention.dtype).eps).log())

    if not terms:
        return attention.sum() * 0.0
    return torch.cat(terms).mean()
