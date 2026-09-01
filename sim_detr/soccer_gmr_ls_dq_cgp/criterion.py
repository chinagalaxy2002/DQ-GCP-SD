"""Null-safe Soccer-GMR losses for LS-DQ-CGP, Binding, and Existence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from sim_detr.ls_dq_cgp.loss import binding_loss
from sim_detr.misc import accuracy
from sim_detr.span_utils import span_cxw_to_xx


class SoccerGMRLSDQCGPCriterion(nn.Module):
    """Keep native localization losses while making empty GT sets well-defined."""

    def __init__(
        self,
        native_criterion,
        *,
        binding_loss_coef=0.2,
        exist_loss_coef=1.0,
        saliency_loss_coef=1.0,
        saliency_margin=0.2,
        background_focal_weight=0.1,
        null_background_focal_weight=0.05,
        null_iou_loss_weight=0.05,
    ):
        super().__init__()
        self.native = native_criterion
        # These two losses are supplied here, once, with empty-set handling.
        self.native.losses = [
            name for name in self.native.losses if name not in {"ls_binding", "ls_exist"}
        ]
        self.weight_dict = {
            key: value
            for key, value in self.native.weight_dict.items()
            if key not in {"loss_ls_bind", "loss_ls_exist"}
        }
        self.weight_dict["loss_ls_bind"] = float(binding_loss_coef)
        self.weight_dict["loss_ls_exist"] = float(exist_loss_coef)
        self.weight_dict["loss_saliency"] = float(saliency_loss_coef)
        self.binding_loss_coef = float(binding_loss_coef)
        self.saliency_margin = float(saliency_margin)
        self.background_focal_weight = float(background_focal_weight)
        self.null_background_focal_weight = float(null_background_focal_weight)
        self.null_iou_loss_weight = float(null_iou_loss_weight)

    def _empty_indices(self, outputs):
        device = outputs["pred_logits"].device
        return [
            (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
            )
            for _ in range(outputs["pred_logits"].shape[0])
        ]

    def _candidate_weights(self, outputs, targets, indices):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        weights = outputs["pred_logits"].new_full(
            (batch_size, num_queries), self.background_focal_weight
        )
        exists = targets["exist_label"].bool()
        weights[~exists] = self.null_background_focal_weight
        source = self.native._get_src_permutation_idx(indices)
        if source[0].numel():
            weights[source] = 1.0
        return weights

    def _label_loss(self, outputs, targets, indices, log=True):
        logits = outputs["pred_logits"].float()
        source = self.native._get_src_permutation_idx(indices)
        target = torch.full(
            logits.shape[:2], self.native.background_label,
            dtype=torch.int64, device=logits.device,
        )
        if source[0].numel():
            target[source] = self.native.foreground_label
        target_one_hot = F.one_hot(target, num_classes=2).permute(0, 2, 1).float()
        focal = torchvision.ops.focal_loss.sigmoid_focal_loss(
            logits.transpose(1, 2), target_one_hot,
            alpha=0.25, gamma=2.0, reduction="none",
        ).mean(dim=1)
        losses = {
            "loss_label": (focal * self._candidate_weights(outputs, targets, indices)).mean()
        }
        if log and source[0].numel():
            losses["class_error"] = 100 - accuracy(
                logits[source], self.native.foreground_label
            )[0]
        return losses

    def _iou_loss(self, outputs, targets):
        score_field = outputs.get("native_iou_scores", outputs["iou_scores"])
        predicted_scores = score_field[..., 0].sigmoid()
        predicted_spans = span_cxw_to_xx(outputs["pred_spans"])
        per_sample = []
        for batch_index, item in enumerate(targets["span_labels"]):
            target_cxw = item["spans"]
            if target_cxw.numel() == 0:
                max_iou = predicted_scores[batch_index].new_zeros(
                    predicted_scores.shape[1]
                )
            else:
                target_spans = span_cxw_to_xx(target_cxw)
                predicted = predicted_spans[batch_index]
                area_pred = (predicted[:, 1] - predicted[:, 0]).clamp(min=0)
                area_target = (target_spans[:, 1] - target_spans[:, 0]).clamp(min=0)
                intersection = (
                    torch.minimum(predicted[:, 1, None], target_spans[None, :, 1])
                    - torch.maximum(predicted[:, 0, None], target_spans[None, :, 0])
                ).clamp(min=0)
                union = area_pred[:, None] + area_target[None, :] - intersection
                iou = torch.where(
                    union > 0, intersection / union.clamp_min(1e-6),
                    torch.zeros_like(union),
                )
                max_iou = iou.max(dim=1).values
            per_sample.append(
                F.mse_loss(predicted_scores[batch_index], max_iou.detach())
            )
        values = torch.stack(per_sample)
        sample_weights = torch.where(
            targets["exist_label"].bool(), torch.ones_like(values),
            torch.full_like(values, self.null_iou_loss_weight),
        )
        return (values * sample_weights).mean()

    def _saliency_loss(self, outputs, targets):
        scores = outputs["saliency_scores"]
        valid = outputs["video_mask"].bool()
        positive = targets["src_pos_mask"].bool() & valid
        terms = []
        for sample_index in range(len(scores)):
            pos_scores = scores[sample_index][positive[sample_index]]
            neg_scores = scores[sample_index][valid[sample_index] & ~positive[sample_index]]
            if pos_scores.numel() and neg_scores.numel():
                pairwise = self.saliency_margin + neg_scores[:, None] - pos_scores[None, :]
                terms.append(F.relu(pairwise).mean() * 2.0)
        return torch.stack(terms).mean() if terms else scores.sum() * 0.0

    @staticmethod
    def _existence_loss(outputs, targets):
        return F.binary_cross_entropy_with_logits(
            outputs["pred_exist_logits"].reshape(-1),
            targets["exist_label"].float().reshape(-1),
        )

    def _all_null_native_losses(self, outputs, targets):
        indices = self._empty_indices(outputs)
        losses = self._label_loss(outputs, targets, indices, log=False)
        zero = outputs["pred_spans"].sum() * 0.0
        losses.update({
            "loss_span": zero,
            "loss_giou": zero,
            "loss_mask_iou": zero,
            "loss_iou_scores": self._iou_loss(outputs, targets),
        })
        return losses

    def _replace_null_aware(self, losses, outputs, targets, suffix=""):
        indices = self.native.matcher(outputs, targets)
        labels = self._label_loss(outputs, targets, indices, log=True)
        losses[f"loss_label{suffix}"] = labels["loss_label"]
        if "class_error" in labels:
            losses[f"class_error{suffix}"] = labels["class_error"]
        losses[f"loss_iou_scores{suffix}"] = self._iou_loss(outputs, targets)
        return indices

    def forward(self, outputs, targets):
        has_positive = any(item["spans"].numel() for item in targets["span_labels"])
        if has_positive:
            losses = self.native(outputs, targets)
            primary = {key: value for key, value in outputs.items() if key != "aux_outputs"}
            indices = self._replace_null_aware(losses, primary, targets)
            for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                self._replace_null_aware(
                    losses, auxiliary, targets, suffix=f"_{layer_index}"
                )
        else:
            primary = {key: value for key, value in outputs.items() if key != "aux_outputs"}
            losses = self._all_null_native_losses(primary, targets)
            indices = self._empty_indices(outputs)
            for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                auxiliary_losses = self._all_null_native_losses(auxiliary, targets)
                losses.update({
                    f"{key}_{layer_index}": value
                    for key, value in auxiliary_losses.items()
                })

        losses["loss_saliency"] = self._saliency_loss(outputs, targets)
        losses["loss_ls_exist"] = self._existence_loss(outputs, targets)
        if self.binding_loss_coef > 0:
            losses["loss_ls_bind"] = binding_loss(
                outputs["ls_d1_attention"], outputs["video_mask"], targets, indices,
                span_loss_type=self.native.span_loss_type,
            )
        return losses
