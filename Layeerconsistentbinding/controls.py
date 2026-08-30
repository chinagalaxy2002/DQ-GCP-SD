"""Runtime controls and loss functions for Layer-Consistent Binding (LCB-Full).

This module implements training-only supervision on Sim-DETR's native decoder
cross-attention across layers D1–D4:
1. All-layer matched binding loss (L_layer_bind): supervises the attention mass
   of the final Hungarian-matched queries on their corresponding GT occurrences
   across all decoder layers (D1 to D4).
2. Occurrence-level consistency loss (L_owner_cons): computes occurrence-level
   distributions (K GT occurrences + background) and penalizes drift from D1
   using Jensen-Shannon divergence with stopgrad on D1.
3. Anti-washout loss (L_drop): hinge loss preventing matched occurrence mass in
   subsequent layers (D2–D4) from dropping below D1's mass by more than a margin.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Ensure repository root is on path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_occurrence_lab.controls import _overlap_for_targets


def js_divergence(
    p: Tensor,
    q: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Compute Jensen-Shannon divergence between discrete distributions p and q.

    Args:
        p: Tensor of shape [..., C] representing probability distributions.
        q: Tensor of shape [..., C] representing probability distributions.
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor of shape [...] containing JS divergence values.
    """
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_p_m = (p * (p.log() - m.log())).sum(dim=-1)
    kl_q_m = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_p_m + kl_q_m)


def normalize_cross_attention(
    attention: Tensor,
    video_mask: Optional[Tensor] = None,
    *,
    eps: Optional[float] = None,
) -> Tensor:
    """Normalize cross-attention over valid video tokens.

    Args:
        attention: Tensor of shape [B, Q, T] or [B, Q, T_full].
        video_mask: Boolean tensor of shape [B, T_vid] indicating valid video clips.
        eps: Epsilon for division. Defaults to torch.finfo(attention.dtype).eps.

    Returns:
        Normalized attention tensor of shape [B, Q, T_vid] summing to 1 over valid clips.
    """
    if eps is None:
        eps = torch.finfo(attention.dtype).eps

    if video_mask is not None:
        valid_t = video_mask.shape[-1]
        attention = attention[..., :valid_t]
        attention = attention * video_mask[:, None, :].to(attention.dtype)

    denominator = attention.sum(dim=-1, keepdim=True).clamp_min(eps)
    return attention / denominator


def compute_layer_consistent_binding_losses(
    attentions: Sequence[Optional[Tensor]],
    outputs: Mapping[str, Tensor],
    targets: Optional[Mapping[str, Any]],
    indices: Optional[Sequence[Tuple[Tensor, Tensor]]],
    *,
    span_loss_type: str,
    layers: Sequence[int] = (0, 1, 2, 3),
    drop_margin: float = 0.05,
    detach_d1_in_drop: bool = True,
) -> Dict[str, Tensor]:
    """Compute the three LCB loss terms across decoder layers.

    Args:
        attentions: List of attention tensors [B, Q, T] for each decoder layer.
        outputs: Model outputs dict containing 'video_mask' and 'pred_logits'.
        targets: Targets dict containing 'span_labels' with GT spans.
        indices: Hungarian matching indices [(src_idx, tgt_idx), ...] from final D4 layer.
        span_loss_type: 'l1' or 'ce'.
        layers: Tuple of layer indices to supervise (defaults to 0, 1, 2, 3).
        drop_margin: Margin delta for anti-washout loss (default: 0.05).
        detach_d1_in_drop: Whether to detach D1 mass as anchor in drop loss (default: True).

    Returns:
        Dict with keys:
        - 'loss_lcb_layer_bind': All-layer matched binding loss.
        - 'loss_lcb_owner_cons': Occurrence-level consistency loss.
        - 'loss_lcb_drop': Anti-washout hinge loss.
    """
    # Create zero tensor connected to gradient graph if prerequisites not met
    ref_tensor = outputs["pred_logits"] if "pred_logits" in outputs else None
    if ref_tensor is None:
        for att in attentions:
            if att is not None:
                ref_tensor = att
                break
    zero = ref_tensor.sum() * 0.0 if ref_tensor is not None else torch.tensor(0.0)

    empty_result = {
        "loss_lcb_layer_bind": zero,
        "loss_lcb_owner_cons": zero,
        "loss_lcb_drop": zero,
    }

    if (
        targets is None
        or "span_labels" not in targets
        or indices is None
        or not attentions
    ):
        return empty_result

    # Filter available requested layers
    active_layer_indices = [
        idx for idx in layers if idx < len(attentions) and attentions[idx] is not None
    ]
    if not active_layer_indices:
        return empty_result

    video_mask = outputs.get("video_mask")
    device = attentions[active_layer_indices[0]].device
    dtype = attentions[active_layer_indices[0]].dtype
    eps = torch.finfo(dtype).eps

    # Normalize attention per layer
    norm_attentions: Dict[int, Tensor] = {}
    for idx in active_layer_indices:
        att = attentions[idx]
        if att is not None:
            norm_attentions[idx] = normalize_cross_attention(att, video_mask, eps=eps)

    # We require layer 0 (D1) for consistency and drop losses
    has_d1 = 0 in norm_attentions

    # Collect per-sample mass and distributions
    masses_by_layer: Dict[int, List[Tensor]] = {idx: [] for idx in active_layer_indices}
    p_by_layer: Dict[int, List[Tensor]] = {idx: [] for idx in active_layer_indices}

    num_total_matches = 0

    for batch_index, (src_indices, tgt_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue

        if video_mask is not None:
            valid_length = int(video_mask[batch_index].bool().sum().item())
        else:
            valid_length = norm_attentions[active_layer_indices[0]].shape[-1]

        if valid_length <= 0:
            continue

        src_indices = src_indices.to(device)
        tgt_indices = tgt_indices.to(device)
        spans = targets["span_labels"][batch_index]["spans"].to(device)
        num_gt = spans.shape[0]
        if num_gt == 0:
            continue

        # Overlap mask for all GT occurrences in this video: [num_gt, valid_length]
        overlap = _overlap_for_targets(
            spans,
            valid_length,
            span_loss_type=span_loss_type,
            dtype=dtype,
            device=device,
        )

        # Background mask: clips not in any GT occurrence [1, valid_length]
        bg_mask = (~overlap.any(dim=0, keepdim=True)).to(dtype)
        # Full category masks: [num_gt + 1, valid_length]
        masks_with_bg = torch.cat([overlap.to(dtype), bg_mask], dim=0)

        num_matches_sample = len(src_indices)
        num_total_matches += num_matches_sample

        for layer_idx in active_layer_indices:
            att = norm_attentions[layer_idx]
            # Matched query attention: [num_matches, valid_length]
            matched_att = att[batch_index, src_indices, :valid_length]

            # Attention mass on each bin (K GT occurrences + 1 background): [num_matches, num_gt + 1]
            dist_raw = torch.matmul(matched_att, masks_with_bg.t())

            # Distribution normalized across occurrences + bg
            dist_norm = dist_raw / dist_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
            p_by_layer[layer_idx].append(dist_norm)

            # Attention mass on matched GT occurrence: [num_matches]
            # tgt_indices contains the matched GT index (0 .. num_gt-1) for each matched query
            matched_mass = dist_raw[torch.arange(num_matches_sample, device=device), tgt_indices]
            masses_by_layer[layer_idx].append(matched_mass)

    if num_total_matches == 0:
        return empty_result

    # 1. All-layer matched binding loss (L_layer_bind)
    # L_layer_bind = - 1/(L * |M|) sum_{l=1}^L sum_{(j,k)} log(m_jk^(l) + eps)
    layer_bind_terms: List[Tensor] = []
    for layer_idx in active_layer_indices:
        m_layer = torch.cat(masses_by_layer[layer_idx], dim=0)
        layer_bind_terms.append(-m_layer.clamp_min(eps).log().mean())

    loss_layer_bind = torch.stack(layer_bind_terms).mean() if layer_bind_terms else zero

    # 2. Occurrence-level consistency loss (L_owner_cons)
    # JS divergence between stopgrad(p^(1)) and p^(l) for l in {2, 3, 4}
    loss_owner_cons = zero
    subsequent_layers = [idx for idx in active_layer_indices if idx > 0]

    if has_d1 and subsequent_layers and p_by_layer[0]:
        cons_layer_terms: List[Tensor] = []
        for layer_idx in subsequent_layers:
            js_sample_list: List[Tensor] = []
            for sample_p1, sample_pl in zip(p_by_layer[0], p_by_layer[layer_idx]):
                # Stopgrad on D1 anchor distribution
                p1_anchor = sample_p1.detach()
                js = js_divergence(p1_anchor, sample_pl, eps=eps)
                js_sample_list.append(js)
            if js_sample_list:
                cons_layer_terms.append(torch.cat(js_sample_list, dim=0).mean())
        if cons_layer_terms:
            loss_owner_cons = torch.stack(cons_layer_terms).mean()

    # 3. Anti-washout loss (L_drop)
    # [ m^(1)_jk - m^(l)_jk - delta ]_+^2 for l in {2, 3, 4}
    loss_drop = zero
    if has_d1 and subsequent_layers and masses_by_layer[0]:
        m1_all = torch.cat(masses_by_layer[0], dim=0)
        if detach_d1_in_drop:
            m1_all = m1_all.detach()

        drop_layer_terms: List[Tensor] = []
        for layer_idx in subsequent_layers:
            ml_all = torch.cat(masses_by_layer[layer_idx], dim=0)
            diff = m1_all - ml_all - float(drop_margin)
            drop_penalty = F.relu(diff).pow(2)
            drop_layer_terms.append(drop_penalty.mean())
        if drop_layer_terms:
            loss_drop = torch.stack(drop_layer_terms).mean()

    return {
        "loss_lcb_layer_bind": loss_layer_bind,
        "loss_lcb_owner_cons": loss_owner_cons,
        "loss_lcb_drop": loss_drop,
    }


def install_layer_consistent_binding_control(
    criterion: nn.Module,
    attention_capture: Any,
    *,
    layer_bind_coef: float = 0.5,
    owner_cons_coef: float = 0.1,
    drop_coef: float = 0.1,
    drop_margin: float = 0.05,
    layers: Sequence[int] = (0, 1, 2, 3),
    detach_d1_in_drop: bool = True,
) -> None:
    """Install Layer-Consistent Binding loss control on a Sim-DETR criterion.

    This wraps the criterion's forward method to capture native cross-attentions
    from D1 to D4, compute the three LCB loss terms using final D4 Hungarian
    matches, and register the corresponding loss weights.

    Args:
        criterion: Sim-DETR SetCriterion instance.
        attention_capture: NativeCrossAttentionCapture instance.
        layer_bind_coef: Weight for L_layer_bind (default: 0.5).
        owner_cons_coef: Weight for L_owner_cons (default: 0.1).
        drop_coef: Weight for L_drop (default: 0.1).
        drop_margin: Anti-washout margin delta (default: 0.05).
        layers: Tuple of layer indices to supervise (default: (0, 1, 2, 3)).
        detach_d1_in_drop: Whether to detach D1 anchor in drop loss (default: True).
    """
    if getattr(criterion, "_lcb_control_installed", False):
        criterion.weight_dict["loss_lcb_layer_bind"] = float(layer_bind_coef)
        criterion.weight_dict["loss_lcb_owner_cons"] = float(owner_cons_coef)
        criterion.weight_dict["loss_lcb_drop"] = float(drop_coef)
        criterion._lcb_attention_capture = attention_capture
        criterion._lcb_drop_margin = drop_margin
        criterion._lcb_layers = tuple(layers)
        criterion._lcb_detach_d1_in_drop = detach_d1_in_drop
        return

    original_forward = criterion.forward

    def controlled_forward(
        self,
        outputs: Mapping[str, Any],
        targets: Optional[Mapping[str, Any]],
        _original=original_forward,
    ):
        # 1. Compute standard Sim-DETR losses
        losses = _original(outputs, targets)

        # 2. Get captured native cross-attentions across decoder layers
        attentions = attention_capture.get()
        if not attentions or targets is None:
            zero = outputs["pred_logits"].sum() * 0.0
            losses["loss_lcb_layer_bind"] = zero
            losses["loss_lcb_owner_cons"] = zero
            losses["loss_lcb_drop"] = zero
            return losses

        # 3. Hungarian matching on final D4 predictions only
        final_outputs = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.matcher(final_outputs, targets)

        # 4. Compute LCB-Full losses
        lcb_losses = compute_layer_consistent_binding_losses(
            attentions,
            outputs,
            targets,
            indices,
            span_loss_type=self.span_loss_type,
            layers=getattr(self, "_lcb_layers", layers),
            drop_margin=getattr(self, "_lcb_drop_margin", drop_margin),
            detach_d1_in_drop=getattr(self, "_lcb_detach_d1_in_drop", detach_d1_in_drop),
        )
        losses.update(lcb_losses)
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict["loss_lcb_layer_bind"] = float(layer_bind_coef)
    criterion.weight_dict["loss_lcb_owner_cons"] = float(owner_cons_coef)
    criterion.weight_dict["loss_lcb_drop"] = float(drop_coef)
    criterion._lcb_original_forward = original_forward
    criterion._lcb_attention_capture = attention_capture
    criterion._lcb_drop_margin = float(drop_margin)
    criterion._lcb_layers = tuple(layers)
    criterion._lcb_detach_d1_in_drop = bool(detach_d1_in_drop)
    criterion._lcb_control_installed = True


def remove_layer_consistent_binding_control(criterion: nn.Module) -> None:
    """Remove Layer-Consistent Binding control and restore the criterion."""
    original = getattr(criterion, "_lcb_original_forward", None)
    if original is not None:
        criterion.forward = original
        delattr(criterion, "_lcb_original_forward")

    criterion.weight_dict.pop("loss_lcb_layer_bind", None)
    criterion.weight_dict.pop("loss_lcb_owner_cons", None)
    criterion.weight_dict.pop("loss_lcb_drop", None)

    for attr in (
        "_lcb_attention_capture",
        "_lcb_drop_margin",
        "_lcb_layers",
        "_lcb_detach_d1_in_drop",
        "_lcb_control_installed",
    ):
        if hasattr(criterion, attr):
            delattr(criterion, attr)


__all__ = [
    "compute_layer_consistent_binding_losses",
    "install_layer_consistent_binding_control",
    "js_divergence",
    "normalize_cross_attention",
    "remove_layer_consistent_binding_control",
]
