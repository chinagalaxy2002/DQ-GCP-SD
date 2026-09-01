"""Null-safe adapter around the unchanged native Sim-DETR criterion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NullSafeCriterion(nn.Module):
    def __init__(
        self, native_criterion, exist_loss_coef=1.0,
        saliency_loss_coef=1.0, saliency_margin=0.2,
    ):
        super().__init__()
        self.native = native_criterion
        self.weight_dict = native_criterion.weight_dict
        self.weight_dict["loss_exist"] = float(exist_loss_coef)
        self.weight_dict["loss_saliency"] = float(saliency_loss_coef)
        self.saliency_margin = float(saliency_margin)

    @staticmethod
    def _existence_loss(outputs, targets):
        return F.binary_cross_entropy_with_logits(
            outputs["pred_exist_logits"].reshape(-1),
            targets["exist_label"].float().reshape(-1),
        )

    def _saliency_loss(self, outputs, targets):
        scores = outputs["saliency_scores"]
        valid_mask = outputs["video_mask"].bool()
        positive_mask = targets["src_pos_mask"].bool() & valid_mask
        sample_losses = []
        for sample_index in range(len(scores)):
            positives = scores[sample_index][positive_mask[sample_index]]
            negatives = scores[sample_index][valid_mask[sample_index] & ~positive_mask[sample_index]]
            if positives.numel() == 0 or negatives.numel() == 0:
                continue
            pairwise = self.saliency_margin + negatives[:, None] - positives[None, :]
            sample_losses.append(F.relu(pairwise).mean() * 2.0)
        if not sample_losses:
            return scores.sum() * 0.0
        return torch.stack(sample_losses).mean()

    def _all_null_losses(self, outputs, targets):
        indices = self.native.matcher(outputs, targets)
        losses = self.native.loss_labels(outputs, targets, indices, log=False)
        zero = outputs["pred_spans"].sum() * 0.0
        losses.update({"loss_span": zero, "loss_giou": zero, "loss_mask_iou": zero})
        losses.update(self.native.loss_iou_scores(outputs, targets, indices))
        return losses

    def forward(self, outputs, targets):
        if any(len(item["spans"]) for item in targets["span_labels"]):
            losses = self.native(outputs, targets)
            losses["loss_exist"] = self._existence_loss(outputs, targets)
            losses["loss_saliency"] = self._saliency_loss(outputs, targets)
            return losses

        primary = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        losses = self._all_null_losses(primary, targets)
        losses["loss_exist"] = self._existence_loss(outputs, targets)
        losses["loss_saliency"] = self._saliency_loss(outputs, targets)
        for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            layer_losses = self._all_null_losses(auxiliary, targets)
            losses.update({f"{key}_{layer_index}": value for key, value in layer_losses.items()})
        return losses
