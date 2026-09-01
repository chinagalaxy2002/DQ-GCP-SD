from types import MethodType

import unittest
import torch
from torch import nn

from sim_detr.transformer import Transformer
from sim_detr.ls_dq_cgp.model import SimDETRWithLSDQCGP


class ZeroPosition(nn.Module):
    def forward(self, features, mask):
        return torch.zeros_like(features)


class IdentityFuser(nn.Module):
    def forward(self, video, text, video_mask, text_mask):
        return video


def make_model(use_exist=False):
    transformer = Transformer(
        d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=2,
        dim_feedforward=128, dropout=0.0, return_intermediate_dec=True,
    )
    model = SimDETRWithLSDQCGP(
        transformer=transformer,
        position_embed=ZeroPosition(),
        txt_position_embed=nn.Identity(),
        txt_dim=8,
        vid_dim=10,
        num_queries=5,
        input_dropout=0.0,
        aux_loss=True,
        ls_num_basis=4,
        ls_prompt_length=2,
        ls_router_hidden_dim=32,
        ls_frf_hidden_dim=64,
        use_ls_exist_head=use_exist,
    )
    model.fuser = IdentityFuser()

    def fake_forward(transformer_self, source, mask, query_embed, position,
                     saliency_proj1, video_length=None, sent_feat=None):
        del mask, position, saliency_proj1
        batch = source.shape[0]
        queries = query_embed.shape[0]
        base = sent_feat.unsqueeze(1).expand(batch, queries, 256)
        hs = torch.stack([base, base + 0.1], dim=0)
        refs = query_embed.sigmoid().unsqueeze(0).unsqueeze(0).expand(2, batch, -1, -1)
        memory = source[:, :video_length]
        raw = source.new_zeros(batch, queries, video_length)
        for query in range(queries):
            raw[:, query, query % video_length] = 1.0
        model.d1_capture.attention = raw
        saliency = source.new_zeros(batch, video_length)
        return hs, refs, memory, saliency

    object.__setattr__(model.transformer, "forward", MethodType(fake_forward, model.transformer))
    return model


def inputs():
    return dict(
        src_txt=torch.randn(2, 4, 8),
        src_txt_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1.]], dtype=torch.float32),
        src_vid=torch.randn(2, 6, 10),
        src_vid_mask=torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1.]], dtype=torch.float32),
    )


class TestModelIntegration(unittest.TestCase):
    def test_model_replaces_final_logits_and_preserves_two_layer_heads(self):
        model = make_model().eval()
        output = model(**inputs())
        self.assertEqual(output["pred_logits"].shape, (2, 5, 2))
        self.assertEqual(output["pred_spans"].shape, (2, 5, 2))
        self.assertEqual(output["ls_d1_attention"].shape, (2, 5, 6))
        self.assertEqual(len(output["aux_outputs"]), 1)
        self.assertTrue(torch.all(output["iou_scores"] == 20))
        self.assertEqual(output["native_iou_scores"].shape, (2, 5, 1))
        self.assertFalse(any(key.startswith("d1_capture") for key in model.state_dict()))

    def test_counterfactuals_and_gradients(self):
        model = make_model().eval()
        batch = inputs()
        active = model(**batch)["pred_logits"]
        model.context_roll = True
        rolled = model(**batch)["pred_logits"]
        self.assertFalse(torch.allclose(active, rolled))
        model.context_roll = False
        model(**batch)["pred_logits"].sum().backward()
        self.assertIsNotNone(model.ls_cgp.basis_prompts.grad)

    def test_optional_exist_head_is_video_query_level(self):
        output = make_model(use_exist=True).eval()(**inputs())
        self.assertEqual(output["pred_exist_logits"].shape, (2,))

    def test_non_two_layer_decoder_is_rejected(self):
        transformer = Transformer(
            d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=3,
            dim_feedforward=128, dropout=0.0, return_intermediate_dec=True,
        )
        with self.assertRaisesRegex(ValueError, "dec_layers=2"):
            SimDETRWithLSDQCGP(
                transformer=transformer, position_embed=ZeroPosition(),
                txt_position_embed=nn.Identity(), txt_dim=8, vid_dim=10,
                num_queries=5, input_dropout=0.0,
            )
