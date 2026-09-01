"""Capture and normalize native D1 decoder cross-attention."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class NativeD1AttentionCapture:
    def __init__(self, decoder: nn.Module) -> None:
        if int(getattr(decoder, "num_layers", 0)) != 2:
            raise ValueError("LS-DQ-CGP requires exactly two decoder layers")
        self.attention: Optional[Tensor] = None
        self.video_mask: Optional[Tensor] = None
        self.handle = decoder.layers[0].cross_attn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        del module, inputs
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("D1 cross-attention did not return (features, weights)")
        weights = output[1]
        if weights is None or weights.ndim != 3:
            shape = None if weights is None else tuple(weights.shape)
            raise RuntimeError(f"expected D1 attention [B,Q,T], got {shape}")
        self.attention = weights

    def reset(self, video_mask: Tensor) -> None:
        if video_mask.ndim != 2:
            raise ValueError("video_mask must have shape [B,T]")
        if bool((video_mask.bool().sum(dim=1) == 0).any()):
            raise ValueError("every sample must contain at least one valid video clip")
        self.attention = None
        self.video_mask = video_mask.bool()

    def normalized(self) -> Tensor:
        if self.attention is None or self.video_mask is None:
            raise RuntimeError("no D1 attention captured for the current forward")
        if self.attention.shape[0] != self.video_mask.shape[0]:
            raise RuntimeError("captured attention batch does not match video mask")
        width = self.video_mask.shape[1]
        if self.attention.shape[-1] != width:
            raise RuntimeError(
                f"decoder memory must be pure video of width {width}; "
                f"captured width is {self.attention.shape[-1]}"
            )
        valid = self.video_mask.unsqueeze(1).to(self.attention.dtype)
        attention = self.attention * valid
        eps = torch.finfo(attention.dtype).eps
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)

    def close(self) -> None:
        self.handle.remove()
