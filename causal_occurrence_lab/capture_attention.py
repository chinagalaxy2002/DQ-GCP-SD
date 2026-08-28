"""Runtime capture of Sim-DETR's native decoder cross-attention.

Sim-DETR's custom ``MultiheadAttention`` already computes and returns
head-averaged weights.  The decoder discards the second return value, so this
module wraps only the instance method and records that value.  The wrapped
method returns the original tuple unchanged; model outputs and state-dict
names therefore remain untouched.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, List, Optional

import torch

from causal_occurrence_lab.common import ensure_repo_on_path

ensure_repo_on_path()


def as_batch_query_time(weights: Any) -> torch.Tensor:
    """Normalize custom-MHA weights to ``[batch, query, source]``."""

    if weights is None:
        raise RuntimeError("Cross-attention returned no weights")
    if weights.ndim == 3:
        return weights
    if weights.ndim == 4:
        # A future attention implementation may expose heads.  Average only
        # here; the current Sim-DETR implementation has already averaged them.
        if weights.shape[1] <= weights.shape[2]:
            return weights.mean(dim=1)
        return weights.mean(dim=0)
    raise ValueError(f"Unexpected cross-attention shape {tuple(weights.shape)}")


class NativeCrossAttentionCapture:
    """Capture all decoder-layer cross-attention tensors for one model."""

    def __init__(self, model: torch.nn.Module):
        decoder = getattr(getattr(model, "transformer", None), "decoder", None)
        if decoder is None or not hasattr(decoder, "layers"):
            raise TypeError("Expected a Sim-DETR model with transformer.decoder.layers")
        self.model = model
        self.decoder = decoder
        self._originals = []
        self._installed = False

    def install(self) -> "NativeCrossAttentionCapture":
        if self._installed:
            return self
        self._originals = []
        for layer_id, layer in enumerate(self.decoder.layers):
            module = getattr(layer, "cross_attn", None)
            if module is None:
                raise AttributeError(f"Decoder layer {layer_id} has no cross_attn")
            original = module.forward
            self._originals.append((module, original))

            def wrapped(
                this,
                *args,
                _original=original,
                _layer_id=layer_id,
                **kwargs,
            ):
                kwargs["need_weights"] = True
                result = _original(*args, **kwargs)
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise RuntimeError(
                        f"Decoder cross-attention layer {_layer_id} returned "
                        f"{type(result)!r}, expected (output, weights)"
                    )
                this._causal_occurrence_attention = result[1]
                return result

            module.forward = MethodType(wrapped, module)
            module._causal_occurrence_attention = None
        self._installed = True
        return self

    def clear(self) -> None:
        for layer in self.decoder.layers:
            layer.cross_attn._causal_occurrence_attention = None

    def get(self, *, normalize: bool = True) -> List[Optional[torch.Tensor]]:
        values = []
        for layer in self.decoder.layers:
            value = getattr(layer.cross_attn, "_causal_occurrence_attention", None)
            if value is None:
                values.append(None)
            else:
                values.append(as_batch_query_time(value) if normalize else value)
        return values

    def remove(self) -> None:
        if not self._installed:
            return
        for module, original in self._originals:
            module.forward = original
            if hasattr(module, "_causal_occurrence_attention"):
                delattr(module, "_causal_occurrence_attention")
        self._originals = []
        self._installed = False

    def __enter__(self) -> "NativeCrossAttentionCapture":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


__all__ = ["NativeCrossAttentionCapture", "as_batch_query_time"]
