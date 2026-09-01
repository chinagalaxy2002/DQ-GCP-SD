"""Hungarian-matched D1 temporal binding loss for LS-DQ-CGP."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from sim_detr.span_utils import span_cxw_to_xx


def binding_loss(
    attention: Tensor,
    video_mask: Tensor,
    targets: dict,
    indices: Optional[List[Tuple[Tensor, Tensor]]],
    span_loss_type: str = "l1",
) -> Tensor:
    if indices is None or "span_labels" not in targets:
        return attention.sum() * 0.0
    masses = []
    for batch_index, (source_ids, target_ids) in enumerate(indices):
        if source_ids.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].bool().sum().item())
        if valid_length <= 0:
            continue
        source_ids = source_ids.to(attention.device)
        target_ids = target_ids.to(attention.device)
        spans = targets["span_labels"][batch_index]["spans"].to(attention.device)[target_ids]
        if span_loss_type == "l1":
            spans_xx = span_cxw_to_xx(spans).clamp(0, 1)
            starts = torch.arange(valid_length, device=attention.device, dtype=attention.dtype)
            starts = starts / float(valid_length)
            ends = starts + 1.0 / float(valid_length)
            positive = (starts.unsqueeze(0) < spans_xx[:, 1:]) & (
                ends.unsqueeze(0) > spans_xx[:, :1]
            )
            empty = ~positive.any(dim=1)
            if bool(empty.any()):
                centers = 0.5 * (starts + ends)
                nearest = (centers.unsqueeze(0) - spans_xx[:, :1]).abs().argmin(dim=1)
                positive[empty] = False
                positive[empty, nearest[empty]] = True
        elif span_loss_type == "ce":
            clips = torch.arange(valid_length, device=attention.device).unsqueeze(0)
            positive = (clips >= spans[:, :1]) & (clips <= spans[:, 1:])
        else:
            raise ValueError(f"unsupported span_loss_type: {span_loss_type}")
        matched = attention[batch_index, source_ids, :valid_length]
        masses.append((matched * positive.to(matched.dtype)).sum(dim=1))
    if not masses:
        return attention.sum() * 0.0
    mass = torch.cat(masses)
    return -mass.clamp_min(torch.finfo(mass.dtype).eps).log().mean()
