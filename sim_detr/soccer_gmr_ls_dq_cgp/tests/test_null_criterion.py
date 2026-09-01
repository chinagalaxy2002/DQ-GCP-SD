from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sim_detr.soccer_gmr_ls_dq_cgp.criterion import SoccerGMRLSDQCGPCriterion


class FakeNativeCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = ["spans", "labels", "ls_binding", "ls_exist"]
        self.weight_dict = {
            "loss_span": 10.0, "loss_giou": 1.0, "loss_label": 4.0,
            "loss_mask_iou": 0.0, "loss_iou_scores": 2.0,
            "loss_ls_bind": 0.2, "loss_ls_exist": 1.0,
        }
        self.background_label = 1
        self.foreground_label = 0
        self.span_loss_type = "l1"
        self.matcher = SimpleNamespace()

    @staticmethod
    def _get_src_permutation_idx(indices):
        return (
            torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)]),
            torch.cat([source for source, _ in indices]),
        )


class TestNullCriterion(unittest.TestCase):
    def test_all_null_batch_is_finite_and_has_zero_localization(self):
        native = FakeNativeCriterion()
        criterion = SoccerGMRLSDQCGPCriterion(native)
        outputs = {
            "pred_logits": torch.randn(2, 3, 2, requires_grad=True),
            "pred_spans": torch.rand(2, 3, 2, requires_grad=True),
            "iou_scores": torch.full((2, 3, 1), 20.0),
            "native_iou_scores": torch.randn(2, 3, 1, requires_grad=True),
            "saliency_scores": torch.randn(2, 5, requires_grad=True),
            "video_mask": torch.ones(2, 5),
            "ls_d1_attention": torch.softmax(torch.randn(2, 3, 5), dim=-1),
            "pred_exist_logits": torch.randn(2, requires_grad=True),
        }
        targets = {
            "span_labels": [
                {"spans": torch.empty(0, 2)}, {"spans": torch.empty(0, 2)}
            ],
            "exist_label": torch.zeros(2),
            "src_pos_mask": torch.zeros(2, 5),
        }
        losses = criterion(outputs, targets)
        self.assertEqual(float(losses["loss_span"]), 0.0)
        self.assertEqual(float(losses["loss_giou"]), 0.0)
        self.assertEqual(float(losses["loss_ls_bind"]), 0.0)
        self.assertTrue(all(
            torch.isfinite(value).all()
            for value in losses.values() if torch.is_tensor(value)
        ))
        weighted = sum(
            value * criterion.weight_dict[key]
            for key, value in losses.items() if key in criterion.weight_dict
        )
        weighted.backward()
        self.assertIsNotNone(outputs["pred_logits"].grad)
        self.assertIsNotNone(outputs["pred_exist_logits"].grad)


if __name__ == "__main__":
    unittest.main()
