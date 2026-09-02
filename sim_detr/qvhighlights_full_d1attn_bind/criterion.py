"""Criterion adapter adding only top-level Hungarian D1 Binding Loss."""

from __future__ import annotations

from torch import nn

from .binding_loss import matched_binding_loss


class BindingCriterion(nn.Module):
    def __init__(self, native_criterion: nn.Module, binding_loss_coef: float) -> None:
        super().__init__()
        self.native = native_criterion
        self.weight_dict = native_criterion.weight_dict
        self.binding_loss_coef = float(binding_loss_coef)
        self.weight_dict["loss_binding"] = self.binding_loss_coef

    def forward(self, outputs, targets):
        losses = self.native(outputs, targets)
        final_outputs = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.native.matcher(final_outputs, targets)
        losses["loss_binding"] = matched_binding_loss(
            outputs.get("d1_attention"),
            outputs.get("video_mask"),
            targets,
            indices,
        )
        return losses
