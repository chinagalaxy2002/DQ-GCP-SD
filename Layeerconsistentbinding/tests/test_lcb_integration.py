import sys
from pathlib import Path
import unittest

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from Layeerconsistentbinding.controls import (
    install_layer_consistent_binding_control,
    remove_layer_consistent_binding_control,
)
from sim_detr.config import BaseOptions
from sim_detr.model import build_model


class TestLCBIntegration(unittest.TestCase):
    def _create_mock_args(self, device):
        base = BaseOptions()
        base.initialize()
        gpu_id = 0 if device.type == "cuda" else 0
        device_arg = "0" if device.type == "cuda" else "-1"
        args = base.parser.parse_args([
            "--dset_name", "hl",
            "--ctx_mode", "video_tef",
            "--v_feat_dim", "10",
            "--t_feat_dim", "8",
            "--dec_layers", "4",
            "--enc_layers", "1",
            "--dim_feedforward", "256",
            "--hidden_dim", "256",
            "--nheads", "4",
            "--num_queries", "10",
            "--device", device_arg,
            "--gpu_id", str(gpu_id),
        ])
        args.device = device
        args.use_matcher = True
        return args

    def test_full_forward_backward_step(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required because Sim-DETR transformer has hardcoded cuda tensors")

        device = torch.device("cuda:0")
        args = self._create_mock_args(device)
        model, criterion = build_model(args)
        model.to(device)
        criterion.to(device)

        capture = NativeCrossAttentionCapture(model).install()

        install_layer_consistent_binding_control(
            criterion,
            capture,
            layer_bind_coef=0.5,
            owner_cons_coef=0.1,
            drop_coef=0.1,
            drop_margin=0.05,
            layers=(0, 1, 2, 3),
        )

        bsz = 2
        l_vid = 8
        l_txt = 4
        src_vid = torch.randn(bsz, l_vid, args.v_feat_dim, device=device)
        src_vid_mask = torch.ones(bsz, l_vid, dtype=torch.float32, device=device)
        src_txt = torch.randn(bsz, l_txt, args.t_feat_dim, device=device)
        src_txt_mask = torch.ones(bsz, l_txt, dtype=torch.float32, device=device)

        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.2, 0.2], [0.6, 0.3]], device=device)},
                {"spans": torch.tensor([[0.5, 0.4]], device=device)},
            ],
            "mask_labels": [
                torch.zeros(2, l_vid, device=device),
                torch.zeros(1, l_vid, device=device),
            ],
        }

        model.train()
        criterion.train()
        capture.clear()

        outputs = model(
            src_txt=src_txt,
            src_txt_mask=src_txt_mask,
            src_vid=src_vid,
            src_vid_mask=src_vid_mask,
        )

        losses = criterion(outputs, targets)

        # Check that LCB losses are computed and included
        self.assertIn("loss_lcb_layer_bind", losses)
        self.assertIn("loss_lcb_owner_cons", losses)
        self.assertIn("loss_lcb_drop", losses)

        # Weighted loss calculation
        total_loss = sum(
            losses[k] * criterion.weight_dict[k]
            for k in losses
            if k in criterion.weight_dict
        )

        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients exist on decoder weights
        has_grad = False
        for param in model.transformer.decoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "Expected non-zero gradients on decoder parameters")

        optimizer.step()
        capture.remove()
        remove_layer_consistent_binding_control(criterion)


if __name__ == "__main__":
    unittest.main()
