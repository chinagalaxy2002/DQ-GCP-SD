"""Utility helpers shared across the DQ-CGP adaptation package."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_mean_pool(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Masked mean pooling over the token dimension.

    Parameters
    ----------
    x : Tensor
        ``[batch, length, hidden_dim]`` – token features (already projected).
    mask : Tensor
        ``[batch, length]`` – 1 for valid tokens, 0 for padding / dropped.
    eps : float
        Denominator clamp to avoid division by zero.

    Returns
    -------
    Tensor
        ``[batch, hidden_dim]`` – masked mean representation.

    Notes
    -----
    In Sim-DETR ``src_txt_mask`` marks valid tokens with **1** and padding
    with **0**, which is the natural polarity for this function.  The mask
    is cast to ``x.dtype`` internally so it works with both fp32 and fp16.
    """
    mask = mask.to(dtype=x.dtype).unsqueeze(-1)          # [B, L, 1]
    numerator = (x * mask).sum(dim=1)                    # [B, D]
    denominator = mask.sum(dim=1).clamp_min(eps)          # [B, 1]
    return numerator / denominator                        # [B, D]


__all__ = ["masked_mean_pool"]
