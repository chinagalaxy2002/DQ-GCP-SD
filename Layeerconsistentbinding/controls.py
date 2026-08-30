"""Runtime controls and loss functions for Layer-Consistent Binding (LCB Acquire -> Preserve).

This module implements the two-stage 'Acquire -> Preserve' objective for Sim-DETR:
1. D1 Ownership Acquisition (L_D1-bind, coef=0.5):
   Supervises D1 native cross-attention on final Hungarian-matched GT occurrences,
   matching the verified NativeBind learning signal:
   L_D1-bind = - 1/|M| sum_{(j,k)} log(m_jk^(1) + eps)

2. D2–D4 Direct Ownership Maintenance (L_late-bind, coef=0.1):
   Directly supervises D2–D4 cross-attentions to prevent subsequent layers from losing the GT:
   L_late-bind = - 1/(3|M|) sum_{l=2}^4 sum_{(j,k)} log(m_jk^(l) + eps)

3. D1 -> D2–D4 Ownership Consistency (L_owner-cons, coef=0.1):
   Penalizes occurrence-identity drift from D1 anchor to D2–D4 using Jensen-Shannon divergence:
   L_owner-cons = 1/(3|M|) sum_{l=2}^4 sum_{(j,k)} JS(stopgrad(p_j^(1)), p_j^(l))

4. Anti-Washout Protection (L_drop, coef=0.1, margin=0.05):
   Hinge loss preventing matched occurrence mass in D2–D4 from decaying below D1:
   L_drop = 1/(3|M|) sum_{l=2}^4 sum_{(j,k)} [ m_jk^(1) - m_jk^(l) - 0.05 ]_+^2
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
    """Compute the decoupled 'Acquire -> Preserve' LCB loss terms.

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
        - 'loss_lcb_d1_bind': D1 ownership acquisition loss.
        - 'loss_lcb_late_bind': D2–D4 direct ownership maintenance loss.
        - 'loss_lcb_owner_cons': D1 -> D2–D4 occurrence-level consistency loss.
        - 'loss_lcb_drop': D1 -> D2–D4 anti-washout hinge loss.
    """
    ref_tensor = outputs["pred_logits"] if "pred_logits" in outputs else None
    if ref_tensor is None:
        for att in attentions:
            if att is not None:
                ref_tensor = att
                break
    zero = ref_tensor.sum() * 0.0 if ref_tensor is not None else torch.tensor(0.0)

    empty_result = {
        "loss_lcb_d1_bind": zero,
        "loss_lcb_late_bind": zero,
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

    active_layer_indices = [
        idx for idx in layers if idx < len(attentions) and attentions[idx] is not None
    ]
    if not active_layer_indices:
        return empty_result

    video_mask = outputs.get("video_mask")
    device = attentions[active_layer_indices[0]].device
    dtype = attentions[active_layer_indices[0]].dtype
    eps = torch.finfo(dtype).eps

    norm_attentions: Dict[int, Tensor] = {}
    for idx in active_layer_indices:
        att = attentions[idx]
        if att is not None:
            norm_attentions[idx] = normalize_cross_attention(att, video_mask, eps=eps)

    has_d1 = 0 in norm_attentions
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

        overlap = _overlap_for_targets(
            spans,
            valid_length,
            span_loss_type=span_loss_type,
            dtype=dtype,
            device=device,
        )

        bg_mask = (~overlap.any(dim=0, keepdim=True)).to(dtype)
        masks_with_bg = torch.cat([overlap.to(dtype), bg_mask], dim=0)

        num_matches_sample = len(src_indices)
        num_total_matches += num_matches_sample

        for layer_idx in active_layer_indices:
            att = norm_attentions[layer_idx]
            matched_att = att[batch_index, src_indices, :valid_length]

            dist_raw = torch.matmul(matched_att, masks_with_bg.t())
            dist_norm = dist_raw / dist_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
            p_by_layer[layer_idx].append(dist_norm)

            matched_mass = dist_raw[torch.arange(num_matches_sample, device=device), tgt_indices]
            masses_by_layer[layer_idx].append(matched_mass)

    if num_total_matches == 0:
        return empty_result

    # 1. D1 Ownership Acquisition Loss (L_D1-bind)
    # L_D1-bind = - 1/|M| sum_{(j,k)} log(m_jk^(1) + eps)
    loss_d1_bind = zero
    if has_d1 and masses_by_layer[0]:
        m1_all = torch.cat(masses_by_layer[0], dim=0)
        loss_d1_bind = -m1_all.clamp_min(eps).log().mean()

    # 2. D2–D4 Direct Ownership Maintenance Loss (L_late-bind)
    # L_late-bind = - 1/(3|M|) sum_{l=2}^4 sum_{(j,k)} log(m_jk^(l) + eps)
    loss_late_bind = zero
    subsequent_layers = [idx for idx in active_layer_indices if idx > 0]
    if subsequent_layers:
        late_bind_terms: List[Tensor] = []
        for layer_idx in subsequent_layers:
            if masses_by_layer[layer_idx]:
                ml_all = torch.cat(masses_by_layer[layer_idx], dim=0)
                late_bind_terms.append(-ml_all.clamp_min(eps).log().mean())
        if late_bind_terms:
            loss_late_bind = torch.stack(late_bind_terms).mean()

    # 3. D1 -> D2–D4 Occurrence-Level Consistency Loss (L_owner-cons)
    # JS divergence between stopgrad(p^(1)) and p^(l) for l in {2, 3, 4}
    loss_owner_cons = zero
    if has_d1 and subsequent_layers and p_by_layer[0]:
        cons_layer_terms: List[Tensor] = []
        for layer_idx in subsequent_layers:
            js_sample_list: List[Tensor] = []
            for sample_p1, sample_pl in zip(p_by_layer[0], p_by_layer[layer_idx]):
                p1_anchor = sample_p1.detach()
                js = js_divergence(p1_anchor, sample_pl, eps=eps)
                js_sample_list.append(js)
            if js_sample_list:
                cons_layer_terms.append(torch.cat(js_sample_list, dim=0).mean())
        if cons_layer_terms:
            loss_owner_cons = torch.stack(cons_layer_terms).mean()

    # 4. Anti-Washout Loss (L_drop)
    # [ m^(1)_jk - m^(l)_jk - delta ]_+^2 for l in {2, 3, 4}
    loss_drop = zero
    if has_d1 and subsequent_layers and masses_by_layer[0]:
        m1_all = torch.cat(masses_by_layer[0], dim=0)
        if detach_d1_in_drop:
            m1_all = m1_all.detach()

        drop_layer_terms: List[Tensor] = []
        for layer_idx in subsequent_layers:
            if masses_by_layer[layer_idx]:
                ml_all = torch.cat(masses_by_layer[layer_idx], dim=0)
                diff = m1_all - ml_all - float(drop_margin)
                drop_penalty = F.relu(diff).pow(2)
                drop_layer_terms.append(drop_penalty.mean())
        if drop_layer_terms:
            loss_drop = torch.stack(drop_layer_terms).mean()

    return {
        "loss_lcb_d1_bind": loss_d1_bind,
        "loss_lcb_late_bind": loss_late_bind,
        "loss_lcb_owner_cons": loss_owner_cons,
        "loss_lcb_drop": loss_drop,
    }


def install_layer_consistent_binding_control(
    criterion: nn.Module,
    attention_capture: Any,
    *,
    d1_bind_coef: float = 0.5,
    late_bind_coef: float = 0.1,
    owner_cons_coef: float = 0.1,
    drop_coef: float = 0.1,
    drop_margin: float = 0.05,
    layers: Sequence[int] = (0, 1, 2, 3),
    detach_d1_in_drop: bool = True,
) -> None:
    """Install 'Acquire -> Preserve' Layer-Consistent Binding control on criterion.

    Registers 4 decoupled loss weights:
    - loss_lcb_d1_bind: 0.5 (D1 ownership acquisition)
    - loss_lcb_late_bind: 0.1 (D2–D4 ownership maintenance)
    - loss_lcb_owner_cons: 0.1 (D1 -> D2–D4 consistency)
    - loss_lcb_drop: 0.1 (anti-washout hinge loss)
    """
    if getattr(criterion, "_lcb_control_installed", False):
        criterion.weight_dict["loss_lcb_d1_bind"] = float(d1_bind_coef)
        criterion.weight_dict["loss_lcb_late_bind"] = float(late_bind_coef)
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
            losses["loss_lcb_d1_bind"] = zero
            losses["loss_lcb_late_bind"] = zero
            losses["loss_lcb_owner_cons"] = zero
            losses["loss_lcb_drop"] = zero
            return losses

        # 3. Hungarian matching on final D4 predictions only
        final_outputs = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.matcher(final_outputs, targets)

        # 4. Compute Acquire -> Preserve LCB losses
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
    criterion.weight_dict["loss_lcb_d1_bind"] = float(d1_bind_coef)
    criterion.weight_dict["loss_lcb_late_bind"] = float(late_bind_coef)
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

    for key in (
        "loss_lcb_d1_bind",
        "loss_lcb_late_bind",
        "loss_lcb_owner_cons",
        "loss_lcb_drop",
        "loss_lcb_layer_bind",
    ):
        criterion.weight_dict.pop(key, None)

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
