"""Capture native D1 decoder cross-attention without changing Sim-DETR."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class NativeD1AttentionCapture:
    """Capture and normalize D1 cross-attention as ``[B, Q, T]``."""

    def __init__(self, transformer: nn.Module) -> None:
        decoder = getattr(transformer, "decoder", None)
        if decoder is None or not hasattr(decoder, "layers") or not decoder.layers:
            raise ValueError("Transformer has no decoder layers")
        self.attention: Optional[Tensor] = None
        self.handle = decoder.layers[0].cross_attn.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        del module, inputs
        self.attention = None
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            weights = output[1]
            if weights is not None and weights.ndim == 3:
                self.attention = weights

    def normalized_attention(self, video_mask: Tensor) -> Tensor:
        if self.attention is None:
            raise RuntimeError("No D1 cross-attention was captured")
        valid = video_mask.to(
            device=self.attention.device, dtype=self.attention.dtype
        ).unsqueeze(1)
        attention = self.attention[..., : video_mask.shape[-1]] * valid
        eps = torch.finfo(attention.dtype).eps
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __del__(self) -> None:
        self.close()
