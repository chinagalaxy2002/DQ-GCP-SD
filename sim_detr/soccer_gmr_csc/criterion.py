"""Null-safe adapter around the unchanged native Sim-DETR criterion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from sim_detr.misc import accuracy
from sim_detr.span_utils import span_cxw_to_xx


class NullSafeCriterion(nn.Module):
    def __init__(
        self, native_criterion, exist_loss_coef=1.0,
        saliency_loss_coef=1.0, saliency_margin=0.2,
        background_focal_weight=0.1, null_background_focal_weight=0.05,
        null_iou_loss_weight=0.05,
    ):
        super().__init__()
        self.native = native_criterion
        self.weight_dict = native_criterion.weight_dict
        self.weight_dict["loss_exist"] = float(exist_loss_coef)
        self.weight_dict["loss_saliency"] = float(saliency_loss_coef)
        self.saliency_margin = float(saliency_margin)
        self.background_focal_weight = float(background_focal_weight)
        self.null_background_focal_weight = float(null_background_focal_weight)
        self.null_iou_loss_weight = float(null_iou_loss_weight)

    def _candidate_weights(self, outputs, targets, indices):
        """Downweight unmatched and null-set candidates without changing heads."""
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

    def _null_aware_label_loss(self, outputs, targets, indices, log=True):
        src_logits = outputs["pred_logits"].to(torch.float32)
        source = self.native._get_src_permutation_idx(indices)
        target = torch.full(
            src_logits.shape[:2], self.native.background_label,
            dtype=torch.int64, device=src_logits.device,
        )
        if source[0].numel():
            target[source] = self.native.foreground_label
        target = F.one_hot(target, num_classes=2).permute(0, 2, 1).to(torch.float32)
        focal = torchvision.ops.focal_loss.sigmoid_focal_loss(
            src_logits.transpose(1, 2), target,
            alpha=0.25, gamma=2.0, reduction="none",
        ).mean(dim=1)
        # Divide by the unweighted number of candidates.  Hence an all-null
        # batch contributes only ``null_background_focal_weight`` of its
        # original background gradient instead of being renormalized back up.
        loss = (focal * self._candidate_weights(outputs, targets, indices)).mean()
        losses = {"loss_label": loss}
        if log and source[0].numel():
            losses["class_error"] = 100 - accuracy(src_logits[source], self.native.foreground_label)[0]
        return losses

    def _null_aware_iou_loss(self, outputs, targets):
        predicted_scores = outputs["iou_scores"][..., 0].sigmoid()
        predicted_spans = span_cxw_to_xx(outputs["pred_spans"])
        per_sample = []
        for batch_index, item in enumerate(targets["span_labels"]):
            targets_cxw = item["spans"]
            padded = predicted_spans.new_zeros((predicted_spans.shape[1], 2))
            if targets_cxw.numel():
                padded[:targets_cxw.shape[0]] = targets_cxw
            target_spans = span_cxw_to_xx(padded)
            areas1 = (predicted_spans[batch_index, :, 1] - predicted_spans[batch_index, :, 0]).clamp(min=0)
            areas2 = (target_spans[:, 1] - target_spans[:, 0]).clamp(min=0)
            intersection = (
                torch.minimum(predicted_spans[batch_index, :, 1, None], target_spans[None, :, 1])
                - torch.maximum(predicted_spans[batch_index, :, 0, None], target_spans[None, :, 0])
            ).clamp(min=0)
            union = areas1[:, None] + areas2[None, :] - intersection
            max_iou = torch.where(
                union > 0, intersection / union.clamp_min(1e-6), torch.zeros_like(union)
            ).max(dim=1).values
            per_sample.append(F.mse_loss(predicted_scores[batch_index], max_iou.detach()))
        values = torch.stack(per_sample)
        sample_weights = torch.where(
            targets["exist_label"].bool(), torch.ones_like(values),
            torch.full_like(values, self.null_iou_loss_weight),
        )
        return {"loss_iou_scores": (values * sample_weights).mean()}

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
        losses = self._null_aware_label_loss(outputs, targets, indices, log=False)
        zero = outputs["pred_spans"].sum() * 0.0
        losses.update({"loss_span": zero, "loss_giou": zero, "loss_mask_iou": zero})
        losses.update(self._null_aware_iou_loss(outputs, targets))
        return losses

    def _replace_null_aware_losses(self, losses, outputs, targets, suffix=""):
        indices = self.native.matcher(outputs, targets)
        label_losses = self._null_aware_label_loss(outputs, targets, indices, log=True)
        losses[f"loss_label{suffix}"] = label_losses["loss_label"]
        if "class_error" in label_losses:
            losses[f"class_error{suffix}"] = label_losses["class_error"]
        losses[f"loss_iou_scores{suffix}"] = self._null_aware_iou_loss(outputs, targets)["loss_iou_scores"]

    def forward(self, outputs, targets):
        if any(len(item["spans"]) for item in targets["span_labels"]):
            losses = self.native(outputs, targets)
            primary = {key: value for key, value in outputs.items() if key != "aux_outputs"}
            self._replace_null_aware_losses(losses, primary, targets)
            for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                self._replace_null_aware_losses(losses, auxiliary, targets, suffix=f"_{layer_index}")
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
