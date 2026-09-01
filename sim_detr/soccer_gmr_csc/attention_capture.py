"""Capture native D1 decoder cross-attention from Sim-DETR without modifying base model."""

from __future__ import annotations

from typing import Optional
import torch
from torch import Tensor, nn


class NativeD1AttentionCapture:
    """Capture native D1 cross-attention weights [B, Q, T] from decoder layer 0."""

    def __init__(self, transformer: nn.Module) -> None:
        self.attention: Optional[Tensor] = None
        self.video_mask: Optional[Tensor] = None
        decoder = getattr(transformer, 'decoder', None)
        if decoder is None or not hasattr(decoder, 'layers') or len(decoder.layers) == 0:
            raise ValueError('Transformer has no decoder layers')
        self.module = decoder.layers[0].cross_attn
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        del module, inputs
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            weights = output[1]
            if weights is not None and weights.ndim == 3:
                self.attention = weights

    def record_mask(self, video_mask: Tensor) -> None:
        self.attention = None
        self.video_mask = video_mask.bool()

    def normalized_attention(self, video_mask: Optional[Tensor] = None) -> Tensor:
        """Return masked and normalized [B, Q, T] attention weights."""
        if self.attention is None:
            raise RuntimeError('No D1 cross-attention was captured for the current forward pass')
        mask = video_mask if video_mask is not None else self.video_mask
        if mask is None:
            return self.attention
        valid = mask.to(device=self.attention.device, dtype=self.attention.dtype).unsqueeze(1)
        attn = self.attention * valid
        eps = torch.finfo(attn.dtype).eps
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)

    def consume(self) -> Optional[Tensor]:
        attn = self.attention
        self.attention = None
        return attn

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __del__(self) -> None:
        self.close()
