"""Model-level tests for DQ-CGP text semantics and native decoder ownership."""

from types import MethodType

import pytest
import torch
from torch import nn

from sim_detr.transformer import Transformer
from sim_detr.dq_cgp.dual_stage.sim_detr_dq_cgp import SimDETRWithDualDQCGP
from sim_detr.dq_cgp.dual_stage.utils import masked_mean_pool


class ZeroPositionEmbedding(nn.Module):
    def forward(self, features, mask):
        return torch.zeros_like(features)


def make_model():
    transformer = Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=4,
        dim_feedforward=128,
        dropout=0.0,
        return_intermediate_dec=True,
        activation="relu",
    )
    model = SimDETRWithDualDQCGP(
        transformer=transformer,
        position_embed=ZeroPositionEmbedding(),
        txt_position_embed=nn.Identity(),
        txt_dim=8,
        vid_dim=10,
        num_queries=5,
        input_dropout=0.0,
        aux_loss=True,
        use_query_cgp=True,
        query_cgp_num_basis=4,
        query_cgp_prompt_length=2,
        query_cgp_router_hidden_dim=32,
        query_cgp_frf_hidden_dim=64,
    ).eval()

    # The semantic branch is exercised through the real model forward.  The
    # expensive Transformer body is replaced because these tests only need to
    # inspect what the model supplied to the already-installed decoder hook.
    def fake_transformer_forward(
        transformer_self,
        src,
        mask,
        query_embed,
        pos_embed,
        saliency_proj1,
        video_length=None,
        sent_feat=None,
    ):
        batch_size = src.shape[0]
        num_queries = query_embed.shape[0]
        num_layers = transformer_self.dec_layers
        hs = src.new_zeros(num_layers, batch_size, num_queries, 256)
        base_reference = query_embed.sigmoid().unsqueeze(0).expand(
            batch_size, -1, -1
        )
        references = base_reference.unsqueeze(0).expand(
            num_layers, -1, -1, -1
        )
        memory = src[:, :video_length]
        saliency = src.new_zeros(batch_size, video_length)
        return hs, references, memory, saliency

    object.__setattr__(
        model.transformer,
        "forward",
        MethodType(fake_transformer_forward, model.transformer),
    )
    return model


def make_forward_inputs():
    generator = torch.Generator().manual_seed(23)
    src_txt = torch.randn(2, 4, 8, generator=generator)
    src_txt[0, 1] = 0  # dropped raw feature row
    src_txt[1, 2:] = 0
    src_txt_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32
    )
    src_vid = torch.randn(2, 3, 10, generator=generator)
    src_vid_mask = torch.ones(2, 3)
    return src_txt, src_txt_mask, src_vid, src_vid_mask


class TestSemanticMask:
    def test_default_mask_excludes_raw_zero_rows_and_padding(self):
        model = make_model()
        src_txt, src_txt_mask, src_vid, src_vid_mask = make_forward_inputs()

        with torch.no_grad():
            projected = model.input_txt_proj(src_txt)
            derived = src_txt_mask.bool() & (src_txt.abs().sum(dim=-1) > 0)
            expected = masked_mean_pool(projected, derived)
            model(src_txt, src_txt_mask, src_vid, src_vid_mask)

        actual = model.transformer.decoder._dq_cgp_adapter_kwargs[
            "query_semantic"
        ]
        assert torch.equal(actual, expected)

    def test_explicit_semantic_mask_is_supported_and_padding_safe(self):
        model = make_model()
        src_txt, src_txt_mask, src_vid, src_vid_mask = make_forward_inputs()
        semantic_mask = torch.tensor(
            [[1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float32
        )

        with torch.no_grad():
            projected = model.input_txt_proj(src_txt)
            effective = semantic_mask.bool() & src_txt_mask.bool()
            expected = masked_mean_pool(projected, effective)
            model(
                src_txt,
                src_txt_mask,
                src_vid,
                src_vid_mask,
                src_txt_semantic_mask=semantic_mask,
            )

        actual = model.transformer.decoder._dq_cgp_adapter_kwargs[
            "query_semantic"
        ]
        assert torch.equal(actual, expected)

    def test_rejects_any_query_with_no_semantic_tokens(self):
        model = make_model()
        src_txt, src_txt_mask, src_vid, src_vid_mask = make_forward_inputs()
        src_txt[1] = 0

        with pytest.raises(ValueError, match="batch indices"):
            model(src_txt, src_txt_mask, src_vid, src_vid_mask)

    def test_rejects_bad_explicit_mask_shape(self):
        model = make_model()
        src_txt, src_txt_mask, src_vid, src_vid_mask = make_forward_inputs()

        with pytest.raises(ValueError, match="src_txt_semantic_mask"):
            model(
                src_txt,
                src_txt_mask,
                src_vid,
                src_vid_mask,
                src_txt_semantic_mask=torch.ones(2, 3),
            )
