"""Integration tests for the state-dict-preserving decoder hook."""

import copy
import os
import tempfile

import pytest
import torch
from torch import nn

from sim_detr.model import MLP, SimDETR
from sim_detr.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from sim_detr.dq_cgp.decoder_wrapper import (
    install_dq_cgp_decoder_hook,
    remove_dq_cgp_decoder_hook,
    set_dq_cgp_adapter_kwargs,
)
from sim_detr.dq_cgp.query_cgp import DETRQueryCGP
from sim_detr.dq_cgp.sim_detr_dq_cgp import SimDETRWithDQCGP


D = 256
NQ = 6
BSZ = 2
TV = 12
N_LAYERS = 4


def make_decoder():
    layer = TransformerDecoderLayer(
        d_model=D,
        nhead=8,
        dim_feedforward=128,
        dropout=0.0,
        activation="relu",
    )
    decoder = TransformerDecoder(
        layer,
        num_layers=N_LAYERS,
        norm=nn.LayerNorm(D),
        return_intermediate=True,
        d_model=D,
        query_dim=2,
        modulate_t_attn=True,
    )
    decoder.class_embed = nn.Linear(D, 2)
    decoder.iou_head = nn.Linear(D, 1)
    decoder.span_embed = MLP(D, D, 2, 3)
    # Make reference refinement content-dependent.  Sim-DETR initializes the
    # final bbox layer to zero, which would make a reference-preservation test
    # pass even if it accidentally consumed the adapted state.
    nn.init.normal_(decoder.bbox_embed.layers[-1].weight, std=0.01)
    nn.init.normal_(decoder.bbox_embed.layers[-1].bias, std=0.01)
    return decoder.eval()


def make_cgp(beta=0.05):
    return DETRQueryCGP(
        hidden_dim=D,
        num_basis=4,
        prompt_length=2,
        router_hidden_dim=32,
        frf_hidden_dim=64,
        beta=beta,
    ).eval()


def make_inputs():
    generator = torch.Generator().manual_seed(17)
    tgt = torch.randn(NQ, BSZ, D, generator=generator)
    memory = torch.randn(TV, BSZ, D, generator=generator)
    memory_mask = torch.zeros(BSZ, TV, dtype=torch.bool)
    pos = torch.randn(TV, BSZ, D, generator=generator)
    refpoints = torch.randn(NQ, BSZ, 2, generator=generator)
    semantic = torch.randn(BSZ, D, generator=generator)
    return tgt, memory, memory_mask, pos, refpoints, semantic


def run_decoder(decoder, inputs):
    tgt, memory, memory_mask, pos, refpoints, _ = inputs
    with torch.no_grad():
        return decoder(
            tgt,
            memory,
            memory_key_padding_mask=memory_mask,
            pos=pos,
            refpoints_unsigmoid=refpoints,
        )


class TestNativeDecoderHook:
    def test_preserves_decoder_identity_public_api_and_state_keys(self):
        decoder = make_decoder()
        baseline_keys = set(decoder.state_dict())
        layers = decoder.layers

        returned = install_dq_cgp_decoder_hook(decoder, make_cgp(), after_layer=0)

        assert returned is decoder
        assert decoder.layers is layers
        assert decoder.num_layers == N_LAYERS
        assert decoder.class_embed is not None
        assert set(decoder.state_dict()) == baseline_keys
        assert not any("query_cgp" in key for key in decoder.state_dict())

    def test_after_layer_validation(self):
        with pytest.raises(ValueError, match="at least two"):
            one_layer = make_decoder()
            one_layer.num_layers = 1
            install_dq_cgp_decoder_hook(one_layer, make_cgp(), after_layer=0)

        for invalid in (-1, N_LAYERS - 1, N_LAYERS):
            with pytest.raises(ValueError, match="after_layer"):
                install_dq_cgp_decoder_hook(
                    make_decoder(), make_cgp(), after_layer=invalid
                )

        with pytest.raises(TypeError, match="integer"):
            install_dq_cgp_decoder_hook(
                make_decoder(), make_cgp(), after_layer=0.5
            )

    def test_remove_restores_native_forward(self):
        decoder = make_decoder()
        native_func = decoder.forward.__func__
        install_dq_cgp_decoder_hook(decoder, make_cgp(), after_layer=0)
        assert decoder.forward.__func__ is not native_func
        remove_dq_cgp_decoder_hook(decoder)
        assert decoder.forward.__func__ is native_func

    def test_adapter_runs_exactly_once(self):
        class CountingIdentityAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, decoder_state, **kwargs):
                self.calls += 1
                return decoder_state

        decoder = make_decoder()
        adapter = CountingIdentityAdapter()
        inputs = make_inputs()
        install_dq_cgp_decoder_hook(decoder, adapter, after_layer=0)
        set_dq_cgp_adapter_kwargs(decoder, query_semantic=inputs[-1])
        run_decoder(decoder, inputs)
        assert adapter.calls == 1

    def test_adapter_must_preserve_decoder_shape(self):
        class BadShapeAdapter(nn.Module):
            def forward(self, decoder_state, **kwargs):
                del kwargs
                return decoder_state[:-1]

        decoder = make_decoder()
        inputs = make_inputs()
        install_dq_cgp_decoder_hook(
            decoder, BadShapeAdapter(), after_layer=0
        )
        set_dq_cgp_adapter_kwargs(decoder, query_semantic=inputs[-1])
        with pytest.raises(ValueError, match="must preserve shape"):
            run_decoder(decoder, inputs)

    def test_active_hook_keeps_layer1_aux_and_r1_native(self):
        baseline = make_decoder()
        hooked = copy.deepcopy(baseline)
        inputs = make_inputs()
        cgp = make_cgp(beta=0.05)
        install_dq_cgp_decoder_hook(hooked, cgp, after_layer=0)
        set_dq_cgp_adapter_kwargs(hooked, query_semantic=inputs[-1])

        baseline_hs, baseline_refs = run_decoder(baseline, inputs)
        hooked_hs, hooked_refs = run_decoder(hooked, inputs)

        assert torch.equal(hooked_hs[0], baseline_hs[0])
        assert torch.equal(hooked_refs[1], baseline_refs[1])
        assert not torch.equal(hooked_hs[1:], baseline_hs[1:])
        assert not torch.equal(hooked_refs[2:], baseline_refs[2:])
        assert cgp.last_output is not None

    def test_beta_zero_is_exact_native_decoder_identity(self):
        baseline = make_decoder()
        hooked = copy.deepcopy(baseline)
        inputs = make_inputs()
        cgp = make_cgp(beta=0.0)
        install_dq_cgp_decoder_hook(hooked, cgp, after_layer=0)
        set_dq_cgp_adapter_kwargs(hooked, query_semantic=inputs[-1])

        baseline_hs, baseline_refs = run_decoder(baseline, inputs)
        hooked_hs, hooked_refs = run_decoder(hooked, inputs)

        assert torch.equal(hooked_hs, baseline_hs)
        assert torch.equal(hooked_refs, baseline_refs)
        assert cgp.last_output is None


def make_transformer():
    return Transformer(
        d_model=D,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=N_LAYERS,
        dim_feedforward=128,
        dropout=0.0,
        return_intermediate_dec=True,
        activation="relu",
    )


def common_model_kwargs():
    return dict(
        position_embed=nn.Identity(),
        txt_position_embed=nn.Identity(),
        txt_dim=16,
        vid_dim=32,
        num_queries=NQ,
        input_dropout=0.0,
        aux_loss=True,
    )


class TestCheckpointCompatibility:
    def test_baseline_keys_load_with_only_query_cgp_missing(self):
        baseline = SimDETR(transformer=make_transformer(), **common_model_kwargs())
        dq_model = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )

        missing, unexpected = dq_model.load_state_dict(
            baseline.state_dict(), strict=False
        )
        assert unexpected == []
        assert missing
        assert all(key.startswith("query_cgp.") for key in missing)
        assert not any("decoder.decoder" in key for key in dq_model.state_dict())

    def test_dq_checkpoint_strict_round_trip(self):
        source = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )
        target = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )
        target.load_state_dict(source.state_dict(), strict=True)

    def test_baseline_checkpoint_helper(self):
        baseline = SimDETR(transformer=make_transformer(), **common_model_kwargs())
        dq_model = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "baseline.ckpt")
            torch.save({"model": baseline.state_dict()}, checkpoint_path)
            SimDETRWithDQCGP.load_from_baseline_checkpoint(
                checkpoint_path, dq_model, device="cpu"
            )

        loaded = dq_model.state_dict()
        for key, value in baseline.state_dict().items():
            assert torch.equal(loaded[key], value), key

    def test_baseline_checkpoint_helper_accepts_module_prefix(self):
        baseline = SimDETR(transformer=make_transformer(), **common_model_kwargs())
        dq_model = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )
        prefixed = {
            f"module.{key}": value for key, value in baseline.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "baseline_dp.ckpt")
            torch.save({"model": prefixed}, checkpoint_path)
            SimDETRWithDQCGP.load_from_baseline_checkpoint(
                checkpoint_path, dq_model, device="cpu"
            )

        loaded = dq_model.state_dict()
        for key, value in baseline.state_dict().items():
            assert torch.equal(loaded[key], value), key

    def test_init_from_rejects_dq_checkpoint(self):
        source = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )
        target = SimDETRWithDQCGP(
            transformer=make_transformer(),
            use_query_cgp=True,
            **common_model_kwargs(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "dq.ckpt")
            torch.save({"model": source.state_dict()}, checkpoint_path)
            with pytest.raises(RuntimeError, match="--init_from expects"):
                SimDETRWithDQCGP.load_from_baseline_checkpoint(
                    checkpoint_path, target, device="cpu"
                )
