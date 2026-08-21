"""State-dict-preserving dual-stage DQ-CGP hook for Sim-DETR.

The first implementation replaced ``transformer.decoder`` with a wrapper
module.  Although its forward computation was correct, that extra registered
module changed every decoder checkpoint key from ``transformer.decoder.*`` to
``transformer.decoder.decoder.*``.  This module instead installs an equivalent
forward method directly on the *existing* ``TransformerDecoder`` instance.

The native decoder therefore keeps:

* its object identity and public attributes (``layers``, ``bbox_embed``, ...),
* its original ``transformer.decoder.*`` state-dict namespace, and
* its native reference refinement and auxiliary-output semantics.

Two operations are added: independent DQ-CGP adapters refine the content state
after decoder layers 1 and 3 (0-indexed layers 0 and 2).  Reference refinement
and auxiliary supervision remain native and pre-adapter at both insertion
points.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from sim_detr.transformer import gen_sineembed_for_position, inverse_sigmoid
except ImportError:  # pragma: no cover - isolated import fallback
    def inverse_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    def gen_sineembed_for_position(pos_tensor):
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        center_embed = pos_tensor[:, :, 0] * scale
        pos_x = center_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        span_embed = pos_tensor[:, :, 1] * scale
        pos_w = span_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return torch.cat((pos_x, pos_w), dim=2)


_HOOK_FLAG = "_dq_cgp_hook_installed"


def _validate_after_layer(decoder, after_layer: int) -> int:
    if isinstance(after_layer, bool) or not isinstance(after_layer, int):
        raise TypeError("after_layer must be an integer")
    if not hasattr(decoder, "num_layers"):
        raise TypeError("decoder must expose num_layers")
    num_layers = int(decoder.num_layers)
    if num_layers < 2:
        raise ValueError("DQ-CGP requires at least two decoder layers")
    if not 0 <= after_layer < num_layers - 1:
        raise ValueError(
            f"after_layer must lie in [0, {num_layers - 2}] for a "
            f"{num_layers}-layer decoder; got {after_layer}"
        )
    return after_layer


def _set_adapter(self, adapter) -> None:
    """Backward-compatible setter for a single installed adapter."""
    layers = tuple(self._dq_cgp_adapters_by_layer)
    if len(layers) != 1:
        raise RuntimeError("set_adapter is only valid for a single-stage hook")
    object.__setattr__(self, "_dq_cgp_adapters_by_layer", {layers[0]: adapter})


def _set_adapters(self, adapters_by_layer: Mapping[int, object]) -> None:
    """Update the plain-Python adapter mapping without registering modules."""
    normalized = _validate_adapters_by_layer(self, adapters_by_layer)
    object.__setattr__(self, "_dq_cgp_adapters_by_layer", normalized)


def _set_adapter_kwargs(self, **kwargs) -> None:
    """Set per-forward tensors forwarded to DQ-CGP."""
    object.__setattr__(self, "_dq_cgp_adapter_kwargs", kwargs)


def _dq_cgp_decoder_forward(
    self,
    tgt: Tensor,
    memory: Tensor,
    tgt_mask: Optional[Tensor] = None,
    memory_mask: Optional[Tensor] = None,
    tgt_key_padding_mask: Optional[Tensor] = None,
    memory_key_padding_mask: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    refpoints_unsigmoid: Optional[Tensor] = None,
):
    """Native Sim-DETR decoder loop with configured inter-layer adapters."""
    output = tgt
    nq, bs, _ = output.size()

    intermediate = []
    reference_points = refpoints_unsigmoid.sigmoid()
    ref_points = [reference_points]

    for layer_id, layer in enumerate(self.layers):
        obj_center = reference_points[..., : self.query_dim]
        query_sine_embed = gen_sineembed_for_position(obj_center)
        query_pos = self.ref_point_head(query_sine_embed)

        if self.query_scale_type != "fix_elewise":
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)
        else:
            pos_transformation = self.query_scale.weight[layer_id]
        query_sine_embed = query_sine_embed * pos_transformation

        if self.modulate_t_attn:
            reft_cond = self.ref_anchor_head(output).sigmoid()
            query_sine_embed *= (
                reft_cond[..., 0] / obj_center[..., 1]
            ).unsqueeze(-1)

        # Preserve Sim-DETR's native query-competition timing.  From D2 onward
        # ``output`` is the DQ-CGP-adapted content state, while
        # ``reference_points`` was refined from the preceding native output.
        if layer_id == 0:
            competition = torch.zeros(nq, nq, bs, device=output.device)
        else:
            cls_score = F.softmax(self.class_embed(output), dim=-1)[..., 0]
            cls_score_row = cls_score.unsqueeze(1).repeat(1, nq, 1)
            cls_score_col = cls_score.unsqueeze(0).repeat(nq, 1, 1)

            iou_score = self.iou_head(output)[..., 0].sigmoid()
            iou_score_row = iou_score.unsqueeze(1).repeat(1, nq, 1)
            iou_score_col = iou_score.unsqueeze(0).repeat(nq, 1, 1)

            score_row = cls_score_row * iou_score_row
            score_col = cls_score_col * iou_score_col
            rank_relation = (score_row >= score_col).float() * 2 - 1

            spans = self.span_embed(output)
            spans += inverse_sigmoid(reference_points)
            spans = spans.sigmoid()
            spans_st = spans[..., 0] + 0.5 * spans[..., 1]
            spans_ed = spans[..., 0] - 0.5 * spans[..., 1]
            spans = torch.stack([spans_st, spans_ed], dim=-1)
            spans_row = spans.unsqueeze(1).repeat(1, nq, 1, 1)
            spans_col = spans.unsqueeze(0).repeat(nq, 1, 1, 1)
            span_relation = 1 - torch.mean(
                (spans_row - spans_col) ** 2, dim=-1
            )
            competition = rank_relation * span_relation

        native_output = layer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=(layer_id == 0),
            competition_matrix=competition,
        )

        # Reference refinement deliberately consumes the native layer output.
        if self.bbox_embed is not None:
            if self.bbox_embed_diff_each_layer:
                tmp = self.bbox_embed[layer_id](native_output)
            else:
                tmp = self.bbox_embed(native_output)
            tmp[..., : self.query_dim] += inverse_sigmoid(reference_points)
            new_reference_points = tmp[..., : self.query_dim].sigmoid()
            if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

        # Layer-N auxiliary supervision remains pre-adapter.
        if self.return_intermediate:
            intermediate.append(self.norm(native_output))

        adapter = self._dq_cgp_adapters_by_layer.get(layer_id)
        if adapter is not None:
            adapted_output = adapter(
                decoder_state=native_output,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                **self._dq_cgp_adapter_kwargs,
            )
            if adapted_output.shape != native_output.shape:
                raise ValueError(
                    "decoder inter-layer adapter must preserve shape: "
                    f"expected {tuple(native_output.shape)}, got "
                    f"{tuple(adapted_output.shape)}"
                )
            output = adapted_output
        else:
            output = native_output

    if self.norm is not None:
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

    if self.return_intermediate:
        if self.bbox_embed is not None:
            return [
                torch.stack(intermediate).transpose(1, 2),
                torch.stack(ref_points).transpose(1, 2),
            ]
        return [
            torch.stack(intermediate).transpose(1, 2),
            reference_points.unsqueeze(0).transpose(1, 2),
        ]

    return output.unsqueeze(0)


def _validate_adapters_by_layer(decoder, adapters_by_layer):
    if not isinstance(adapters_by_layer, Mapping) or not adapters_by_layer:
        raise ValueError("adapters_by_layer must be a non-empty mapping")
    normalized = {}
    for after_layer, adapter in adapters_by_layer.items():
        layer = _validate_after_layer(decoder, after_layer)
        if layer in normalized:
            raise ValueError(f"duplicate adapter insertion layer: {layer}")
        normalized[layer] = adapter
    return dict(sorted(normalized.items()))


def install_dq_cgp_decoder_hooks(decoder, adapters_by_layer):
    """Install one or more DQ-CGP adapters without replacing the decoder.

    The model owns and registers the adapters; the decoder holds only a plain
    Python mapping.  Its original ``transformer.decoder.*`` keys are unchanged.
    """
    adapters_by_layer = _validate_adapters_by_layer(decoder, adapters_by_layer)
    if getattr(decoder, _HOOK_FLAG, False):
        raise RuntimeError("DQ-CGP decoder hook is already installed")

    object.__setattr__(decoder, "_dq_cgp_native_forward", decoder.forward)
    object.__setattr__(decoder, "_dq_cgp_adapters_by_layer", adapters_by_layer)
    object.__setattr__(decoder, "_dq_cgp_adapter_kwargs", {})
    object.__setattr__(decoder, _HOOK_FLAG, True)
    object.__setattr__(decoder, "set_adapter", MethodType(_set_adapter, decoder))
    object.__setattr__(decoder, "set_adapters", MethodType(_set_adapters, decoder))
    object.__setattr__(
        decoder, "set_adapter_kwargs", MethodType(_set_adapter_kwargs, decoder)
    )
    object.__setattr__(
        decoder, "forward", MethodType(_dq_cgp_decoder_forward, decoder)
    )
    return decoder


def install_dq_cgp_decoder_hook(decoder, adapter, after_layer: int = 0):
    """Compatibility helper for a single insertion point."""
    return install_dq_cgp_decoder_hooks(
        decoder=decoder,
        adapters_by_layer={after_layer: adapter},
    )


def remove_dq_cgp_decoder_hook(decoder):
    """Restore the decoder's original forward method (primarily for tests)."""
    if not getattr(decoder, _HOOK_FLAG, False):
        return decoder
    native_forward = decoder._dq_cgp_native_forward
    object.__setattr__(decoder, "forward", native_forward)
    for name in (
        "_dq_cgp_native_forward",
        "_dq_cgp_adapters_by_layer",
        "_dq_cgp_adapter_kwargs",
        _HOOK_FLAG,
        "set_adapter",
        "set_adapters",
        "set_adapter_kwargs",
    ):
        object.__delattr__(decoder, name)
    return decoder


def set_dq_cgp_adapter_kwargs(decoder, **kwargs) -> None:
    """Set adapter inputs while keeping Transformer.forward's API unchanged."""
    if not getattr(decoder, _HOOK_FLAG, False):
        raise RuntimeError("DQ-CGP decoder hook is not installed")
    _set_adapter_kwargs(decoder, **kwargs)


class DQCGPDecoderWrapper:
    """Backward-compatible factory that now returns the original decoder.

    New code should call :func:`install_dq_cgp_decoder_hook`.  Keeping this
    factory avoids breaking earlier experiment utilities while ensuring that
    no wrapper ``nn.Module`` is introduced into the state-dict hierarchy.
    """

    def __new__(cls, decoder, after_layer: int = 0):
        return install_dq_cgp_decoder_hook(
            decoder=decoder, adapter=None, after_layer=after_layer
        )


__all__ = [
    "DQCGPDecoderWrapper",
    "install_dq_cgp_decoder_hook",
    "install_dq_cgp_decoder_hooks",
    "remove_dq_cgp_decoder_hook",
    "set_dq_cgp_adapter_kwargs",
]
