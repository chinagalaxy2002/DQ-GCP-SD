"""Runtime-only causal controls for Sim-DETR DQ-CGP experiments.

The production DQ module intentionally has a beta-zero early return.  That
is useful for deployment, but it is not a valid training control because it
also removes the temporal attention from the binding-loss computation.  The
wrapper below calls the original forward with the fast path disabled and then
returns the unmodified decoder state when ``inject_residual=False``.

Routing is delegated to the production ``compute_routing_loss`` helper.  In
particular, matched routes from the whole batch are concatenated before
computing ``H(W|C) - H(W)``.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


def install_injection_control(model: torch.nn.Module, inject_residual: bool) -> None:
    """Make DQ-CGP compute diagnostics while optionally suppressing its update."""

    module = getattr(model, "query_cgp", None)
    if module is None:
        raise ValueError("Injection control requires a DQ-CGP model")
    if getattr(module, "_causal_control_installed", False):
        module._causal_inject_residual = bool(inject_residual)
        return

    original = module.forward

    def controlled_forward(
        self,
        decoder_state,
        *args,
        _original=original,
        **kwargs,
    ):
        old_fast_path = self._beta_is_zero
        self._beta_is_zero = False
        try:
            adapted = _original(decoder_state, *args, **kwargs)
        finally:
            self._beta_is_zero = old_fast_path

        if bool(getattr(self, "_causal_inject_residual", True)):
            return adapted

        # Keep the complete diagnostic output for loss/trajectory analysis.
        # The adapted state is not exposed to the decoder in this branch, but
        # retaining it in the transient diagnostics lets the analysis recover
        # the true pre-residual candidate when it reports update magnitude.
        return decoder_state

    module.forward = MethodType(controlled_forward, module)
    module._causal_inject_residual = bool(inject_residual)
    module._causal_control_original_forward = original
    module._causal_control_installed = True


def remove_injection_control(model: torch.nn.Module) -> None:
    module = getattr(model, "query_cgp", None)
    if module is None or not getattr(module, "_causal_control_installed", False):
        return
    module.forward = module._causal_control_original_forward
    for name in (
        "_causal_inject_residual",
        "_causal_control_original_forward",
        "_causal_control_installed",
    ):
        if hasattr(module, name):
            delattr(module, name)


def _overlap_for_targets(
    target_spans: Tensor,
    valid_length: int,
    *,
    span_loss_type: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build target-to-frame overlap masks using production loss semantics."""

    if span_loss_type == "l1":
        from sim_detr.span_utils import span_cxw_to_xx

        target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
        starts = torch.arange(valid_length, device=device, dtype=dtype) / float(valid_length)
        ends = starts + 1.0 / float(valid_length)
        overlap = (
            (starts.unsqueeze(0) < target_xx[:, 1:])
            & (ends.unsqueeze(0) > target_xx[:, :1])
        )
        empty = ~overlap.any(dim=1)
        if bool(empty.any()):
            centers = 0.5 * (starts + ends)
            nearest = (centers.unsqueeze(0) - target_xx[:, :1]).abs().argmin(dim=1)
            overlap[empty] = False
            overlap[empty, nearest[empty]] = True
        return overlap
    if span_loss_type == "ce":
        clip_indices = torch.arange(valid_length, device=device).unsqueeze(0)
        return (clip_indices >= target_spans[:, :1]) & (clip_indices <= target_spans[:, 1:])
    raise ValueError(f"Unsupported span_loss_type={span_loss_type!r}")


def collect_control_masses_and_routes(
    temporal_attention: Tensor,
    basis_weights: Optional[Tensor],
    video_mask: Tensor,
    targets: Optional[Mapping[str, Any]],
    indices: Optional[Sequence[Tuple[Tensor, Tensor]]],
    *,
    span_loss_type: str,
    binding_target: str = "matched",
) -> Tuple[List[Tensor], List[Tensor]]:
    """Collect per-match masses for matched, union, or rolled supervision."""

    if binding_target not in {"matched", "union", "rolled"}:
        raise ValueError(f"Unknown binding target: {binding_target}")
    if targets is None or "span_labels" not in targets or indices is None:
        return [], []

    masses: List[Tensor] = []
    routes: List[Tensor] = []
    device = temporal_attention.device
    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].bool().sum().item())
        if valid_length <= 0:
            continue
        src_indices = src_indices.to(device)
        target_indices = target_indices.to(device)
        spans = targets["span_labels"][batch_index]["spans"].to(device)
        matched_attention = temporal_attention[batch_index, src_indices, :valid_length]
        if binding_target == "matched":
            selected_spans = spans[target_indices]
        elif binding_target == "union":
            selected_spans = spans
        else:
            selected_spans = (
                spans[target_indices.roll(1)]
                if len(target_indices) > 1
                else spans[target_indices]
            )

        overlap = _overlap_for_targets(
            selected_spans,
            valid_length,
            span_loss_type=span_loss_type,
            dtype=temporal_attention.dtype,
            device=device,
        )
        if binding_target == "union":
            positive = overlap.any(dim=0, keepdim=True).expand(len(src_indices), -1)
        else:
            positive = overlap
        masses.append((matched_attention * positive.to(matched_attention.dtype)).sum(dim=1))
        if basis_weights is not None:
            routes.append(basis_weights[batch_index, src_indices])
    return masses, routes


def controlled_query_cgp_loss(
    outputs: Mapping[str, Tensor],
    targets: Optional[Mapping[str, Any]],
    indices: Optional[Sequence[Tuple[Tensor, Tensor]]],
    *,
    span_loss_type: str,
    binding_target: str,
) -> Mapping[str, Tensor]:
    """Return unweighted DQ losses for a causal criterion control."""

    required = {
        "query_cgp_temporal_attention",
        "query_cgp_basis_weights",
        "query_cgp_video_mask",
    }
    if not required.issubset(outputs) or targets is None or indices is None:
        zero = outputs["pred_logits"].sum() * 0.0
        return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": zero}

    attention = outputs["query_cgp_temporal_attention"]
    routes = outputs["query_cgp_basis_weights"]
    video_mask = outputs["query_cgp_video_mask"].bool()
    masses, matched_routes = collect_control_masses_and_routes(
        attention, routes, video_mask, targets, indices,
        span_loss_type=span_loss_type, binding_target=binding_target,
    )
    if not masses:
        zero = attention.sum() * 0.0
        return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": routes.sum() * 0.0}

    eps = torch.finfo(attention.dtype).eps
    binding = -torch.cat(masses).clamp_min(eps).log().mean()
    if matched_routes:
        # Exact production objective: concatenate matched routes from the
        # complete batch before computing conditional minus marginal entropy.
        from sim_detr.dq_cgp.loss import compute_routing_loss

        route = compute_routing_loss(torch.cat(matched_routes, dim=0))
    else:
        route = routes.sum() * 0.0
    return {"loss_query_cgp_bind": binding, "loss_query_cgp_route": route}


def install_query_cgp_loss_control(
    criterion: torch.nn.Module,
    *,
    binding_target: str = "matched",
) -> None:
    """Patch only the criterion method used for top-level DQ-CGP loss."""

    if not hasattr(criterion, "loss_query_cgp"):
        raise ValueError("Criterion has no DQ-CGP loss method")
    original = criterion.loss_query_cgp

    def controlled_loss(self, outputs, targets, indices, log=True, _target=binding_target):
        del log
        return controlled_query_cgp_loss(
            outputs, targets, indices,
            span_loss_type=self.span_loss_type,
            binding_target=_target,
        )

    criterion.loss_query_cgp = MethodType(controlled_loss, criterion)
    criterion._causal_original_loss_query_cgp = original
    criterion._causal_binding_target = binding_target


def remove_query_cgp_loss_control(criterion: torch.nn.Module) -> None:
    original = getattr(criterion, "_causal_original_loss_query_cgp", None)
    if original is None:
        return
    criterion.loss_query_cgp = original
    delattr(criterion, "_causal_original_loss_query_cgp")
    if hasattr(criterion, "_causal_binding_target"):
        delattr(criterion, "_causal_binding_target")


def install_native_binding_control(
    criterion: torch.nn.Module,
    attention_capture: Any,
    *,
    binding_coef: float = 0.2,
) -> None:
    """Add matched binding loss on D1 native cross-attention to base criterion."""

    original_forward = criterion.forward

    def controlled_forward(self, outputs, targets, _original=original_forward):
        losses = _original(outputs, targets)
        attentions = attention_capture.get()
        native = attentions[0] if attentions else None
        if native is None or targets is None:
            zero = outputs["pred_logits"].sum() * 0.0
            losses["loss_native_bind"] = zero
            return losses
        final_outputs = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.matcher(final_outputs, targets)
        video_mask = outputs.get("video_mask")
        if video_mask is None:
            video_mask = torch.ones(
                (native.shape[0], native.shape[-1]),
                dtype=torch.bool,
                device=native.device,
            )
        native = native[..., :video_mask.shape[-1]]
        native = native * video_mask[:, None, :].to(native.dtype)
        native = native / native.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(native.dtype).eps
        )
        masses, _ = collect_control_masses_and_routes(
            native, None, video_mask, targets, indices,
            span_loss_type=self.span_loss_type, binding_target="matched",
        )
        if masses:
            eps = torch.finfo(native.dtype).eps
            loss = -torch.cat(masses).clamp_min(eps).log().mean()
        else:
            loss = native.sum() * 0.0
        losses["loss_native_bind"] = loss
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict["loss_native_bind"] = float(binding_coef)
    criterion._causal_original_forward = original_forward
    criterion._causal_native_binding_capture = attention_capture


def remove_native_binding_control(criterion: torch.nn.Module) -> None:
    original = getattr(criterion, "_causal_original_forward", None)
    if original is not None:
        criterion.forward = original
        delattr(criterion, "_causal_original_forward")
    criterion.weight_dict.pop("loss_native_bind", None)
    if hasattr(criterion, "_causal_native_binding_capture"):
        delattr(criterion, "_causal_native_binding_capture")


__all__ = [
    "collect_control_masses_and_routes", "controlled_query_cgp_loss",
    "install_injection_control", "install_native_binding_control",
    "install_query_cgp_loss_control", "remove_injection_control",
    "remove_native_binding_control", "remove_query_cgp_loss_control",
]
