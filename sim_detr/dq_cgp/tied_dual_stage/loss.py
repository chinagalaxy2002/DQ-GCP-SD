"""DQ-CGP losses adapted to Sim-DETR's target container.

The loss semantics intentionally follow DQ-CGP V3: each Hungarian-matched
query is bound to its own ground-truth window, and routing regularisation is
computed only over those matched positive queries.  Sim-DETR supplies targets
as a dict of batched fields, in particular::

    targets["span_labels"][batch_index]["spans"]

Auxiliary decoder outputs do not carry DQ-CGP diagnostics and are skipped by
``DQCGPSetCriterion``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from sim_detr.span_utils import span_cxw_to_xx


def _collect_matched_masses_and_routes(
    temporal_attention: Tensor,
    basis_weights: Optional[Tensor],
    video_mask: Tensor,
    targets: dict,
    indices: List[Tuple[Tensor, Tensor]],
    span_loss_type: str,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Collect per-match GT attention masses and optional routing weights."""
    binding_masses: List[Tensor] = []
    matched_routes: List[Tensor] = []
    device = temporal_attention.device

    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue

        # GT spans are normalized by each sample's unpadded video length.
        valid_length = int(video_mask[batch_index].bool().sum().item())
        if valid_length <= 0:
            continue

        src_indices = src_indices.to(device)
        target_indices = target_indices.to(device)
        matched_attention = temporal_attention[
            batch_index, src_indices, :valid_length
        ]
        target_spans = targets["span_labels"][batch_index]["spans"].to(device)
        target_spans = target_spans[target_indices]

        if span_loss_type == "l1":
            target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
            clip_starts = torch.arange(
                valid_length,
                device=device,
                dtype=temporal_attention.dtype,
            ) / float(valid_length)
            clip_ends = clip_starts + 1.0 / float(valid_length)
            overlap = (
                (clip_starts.unsqueeze(0) < target_xx[:, 1:])
                & (clip_ends.unsqueeze(0) > target_xx[:, :1])
            )

            # Preserve V3's robust fallback for degenerate/boundary windows.
            empty_overlap = ~overlap.any(dim=1)
            if bool(empty_overlap.any()):
                clip_centers = 0.5 * (clip_starts + clip_ends)
                nearest = (
                    clip_centers.unsqueeze(0) - target_xx[:, :1]
                ).abs().argmin(dim=1)
                overlap[empty_overlap] = False
                overlap[empty_overlap, nearest[empty_overlap]] = True
        elif span_loss_type == "ce":
            clip_indices = torch.arange(valid_length, device=device).unsqueeze(0)
            overlap = (
                (clip_indices >= target_spans[:, :1])
                & (clip_indices <= target_spans[:, 1:])
            )
        else:
            raise ValueError(
                f"Unsupported span_loss_type={span_loss_type!r}; expected 'l1' or 'ce'"
            )

        target_mass = (
            matched_attention * overlap.to(temporal_attention.dtype)
        ).sum(dim=1)
        binding_masses.append(target_mass)
        if basis_weights is not None:
            matched_routes.append(basis_weights[batch_index, src_indices])

    return binding_masses, matched_routes


def compute_binding_loss(
    temporal_attention: Tensor,
    video_mask: Tensor,
    targets: dict,
    indices: Optional[List[Tuple[Tensor, Tensor]]],
    clip_len: int = 2,
    eps: Optional[float] = None,
    span_loss_type: str = "l1",
) -> Tensor:
    """Bind each Hungarian-matched query to its own GT temporal window.

    ``clip_len`` is retained for API compatibility.  L1 targets are already
    normalized by video duration, while CE targets are clip indices, so the V3
    construction does not otherwise require it.
    """
    del clip_len
    if indices is None or targets is None or "span_labels" not in targets:
        return temporal_attention.sum() * 0.0

    binding_masses, _ = _collect_matched_masses_and_routes(
        temporal_attention=temporal_attention,
        basis_weights=None,
        video_mask=video_mask,
        targets=targets,
        indices=indices,
        span_loss_type=span_loss_type,
    )
    if not binding_masses:
        return temporal_attention.sum() * 0.0

    mass = torch.cat(binding_masses)
    clamp_eps = (
        torch.finfo(mass.dtype).eps
        if eps is None
        else max(float(eps), torch.finfo(mass.dtype).eps)
    )
    return -mass.clamp_min(clamp_eps).log().mean()


def compute_routing_loss(basis_weights: Tensor) -> Tensor:
    """Return V3's routing objective ``H(W|C) - H(W)``.

    The caller passes only Hungarian-matched positive routes.  Leading
    dimensions are flattened to keep this helper convenient for direct tests.
    """
    if basis_weights.numel() == 0:
        return basis_weights.sum() * 0.0

    routes = basis_weights.reshape(-1, basis_weights.shape[-1])
    eps = torch.finfo(routes.dtype).eps
    conditional_entropy = -(
        routes * routes.clamp_min(eps).log()
    ).sum(dim=-1).mean()
    marginal = routes.mean(dim=0)
    marginal_entropy = -(
        marginal * marginal.clamp_min(eps).log()
    ).sum()
    return conditional_entropy - marginal_entropy


def _stage_query_cgp_losses(
    temporal_attention: Tensor,
    basis_weights: Tensor,
    video_mask: Tensor,
    targets: dict,
    indices: List[Tuple[Tensor, Tensor]],
    span_loss_type: str,
) -> Tuple[Tensor, Tensor]:
    """Compute binding and routing for one decoder-depth adapter."""
    binding_masses, matched_routes = _collect_matched_masses_and_routes(
        temporal_attention=temporal_attention,
        basis_weights=basis_weights,
        video_mask=video_mask,
        targets=targets,
        indices=indices,
        span_loss_type=span_loss_type,
    )
    if not binding_masses:
        return temporal_attention.sum() * 0.0, basis_weights.sum() * 0.0

    eps = torch.finfo(temporal_attention.dtype).eps
    target_mass = torch.cat(binding_masses)
    binding_loss = -target_mass.clamp_min(eps).log().mean()
    route_loss = compute_routing_loss(torch.cat(matched_routes, dim=0))
    return binding_loss, route_loss


def loss_query_cgp(
    outputs: dict,
    targets: Optional[dict],
    indices: Optional[List[Tuple[Tensor, Tensor]]],
    clip_len: int = 2,
    binding_weight: float = 1.0,
    routing_weight: float = 1.0,
    span_loss_type: str = "l1",
) -> Dict[str, Tensor]:
    """Average the two depth-specific V3 losses for the tied module.

    Averaging preserves the single-stage experiment's total auxiliary-loss
    budget.  Routing is computed separately per stage before averaging; this
    prevents specialization across depths from hiding within-stage collapse.
    """
    del clip_len
    required = {
        "query_cgp_temporal_attention",
        "query_cgp_basis_weights",
        "query_cgp_late_temporal_attention",
        "query_cgp_late_basis_weights",
        "query_cgp_video_mask",
    }
    if (
        targets is None
        or "span_labels" not in targets
        or indices is None
        or not required.issubset(outputs)
    ):
        zero = outputs["pred_logits"].sum() * 0.0
        return {
            "loss_query_cgp_bind": zero,
            "loss_query_cgp_route": zero,
        }

    video_mask = outputs["query_cgp_video_mask"].bool()
    early_bind, early_route = _stage_query_cgp_losses(
        temporal_attention=outputs["query_cgp_temporal_attention"],
        basis_weights=outputs["query_cgp_basis_weights"],
        video_mask=video_mask,
        targets=targets,
        indices=indices,
        span_loss_type=span_loss_type,
    )
    late_bind, late_route = _stage_query_cgp_losses(
        temporal_attention=outputs["query_cgp_late_temporal_attention"],
        basis_weights=outputs["query_cgp_late_basis_weights"],
        video_mask=video_mask,
        targets=targets,
        indices=indices,
        span_loss_type=span_loss_type,
    )
    binding_loss = 0.5 * (early_bind + late_bind)
    route_loss = 0.5 * (early_route + late_route)

    return {
        "loss_query_cgp_bind": binding_loss * float(binding_weight),
        "loss_query_cgp_route": route_loss * float(routing_weight),
        "query_cgp_early_bind_raw": early_bind.detach(),
        "query_cgp_late_bind_raw": late_bind.detach(),
        "query_cgp_early_route_raw": early_route.detach(),
        "query_cgp_late_route_raw": late_route.detach(),
    }


__all__ = [
    "compute_binding_loss",
    "compute_routing_loss",
    "loss_query_cgp",
]
