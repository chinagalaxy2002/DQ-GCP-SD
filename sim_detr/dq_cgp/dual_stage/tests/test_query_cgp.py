"""Tests for DETRQueryCGP (query_cgp.py).

Run from the Sim-DETR project root:
    PYTHONPATH=. python -m pytest sim_detr/dq_cgp/tests/test_query_cgp.py -v
"""

import pytest
import torch
from sim_detr.dq_cgp.dual_stage.query_cgp import DETRQueryCGP, DETRQueryCGPOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_QUERIES = 10
BATCH_SIZE  = 2
HIDDEN_DIM  = 256
VIDEO_LEN   = 75   # full valid frames
SOURCE_LEN  = 75   # Sim-DETR: memory is already pure video


def make_module(**kwargs):
    defaults = dict(hidden_dim=HIDDEN_DIM, num_basis=16, prompt_length=6,
                    router_hidden_dim=256, frf_hidden_dim=512,
                    temperature=1.0, beta=0.05)
    defaults.update(kwargs)
    return DETRQueryCGP(**defaults)


def make_inputs(source_len=SOURCE_LEN, pad_frames=0):
    """Return (decoder_state, memory, memory_key_padding_mask, query_semantic)."""
    decoder_state = torch.randn(NUM_QUERIES, BATCH_SIZE, HIDDEN_DIM)
    memory        = torch.randn(source_len, BATCH_SIZE, HIDDEN_DIM)
    # Padding mask: True = padding
    mask = torch.zeros(BATCH_SIZE, source_len, dtype=torch.bool)
    if pad_frames > 0:
        mask[:, source_len - pad_frames:] = True
    query_semantic = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    return decoder_state, memory, mask, query_semantic


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_adapted_state_shape_matches_input(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert adapted.shape == ds.shape, \
            f"Expected {ds.shape}, got {adapted.shape}"

    def test_temporal_attention_shape(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        attn = m.last_output.temporal_attention
        assert attn.shape == (BATCH_SIZE, NUM_QUERIES, VIDEO_LEN)

    def test_basis_weights_shape(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        bw = m.last_output.basis_weights
        assert bw.shape == (BATCH_SIZE, NUM_QUERIES, 16)

    def test_video_length_none_uses_full_memory(self):
        """When video_length=None the entire memory is treated as video."""
        m = make_module()
        ds, mem, mask, sem = make_inputs(source_len=75)
        adapted = m(ds, mem, mask, sem, video_length=None)
        assert adapted.shape == ds.shape
        assert m.last_output.temporal_attention.shape == (BATCH_SIZE, NUM_QUERIES, 75)


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

class TestNumerical:
    def test_attention_sums_to_one_over_valid_frames(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        attn = m.last_output.temporal_attention
        row_sum = attn.sum(dim=-1)  # [B, Q]
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5), \
            f"Attention row sums: min={row_sum.min():.6f} max={row_sum.max():.6f}"

    def test_attention_zero_on_padding_frames(self):
        PAD = 20
        m = make_module()
        ds, mem, mask, sem = make_inputs(pad_frames=PAD)
        m(ds, mem, mask, sem, video_length=SOURCE_LEN)
        attn = m.last_output.temporal_attention
        pad_mass = attn[:, :, SOURCE_LEN - PAD:].abs().max()
        assert pad_mass < 1e-5, f"Attention on padding frames should be 0, got {pad_mass}"

    def test_basis_weights_sum_to_one(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        bw = m.last_output.basis_weights
        row_sum = bw.sum(dim=-1)
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)

    def test_no_nan_or_inf(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert not torch.isnan(adapted).any(), "NaN in adapted state"
        assert not torch.isinf(adapted).any(), "Inf in adapted state"


# ---------------------------------------------------------------------------
# Beta=0 identity test (critical ablation)
# ---------------------------------------------------------------------------

class TestBetaZero:
    def test_beta_zero_returns_identity(self):
        m = make_module(beta=0.0)
        ds, mem, mask, sem = make_inputs()
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert adapted is ds, \
            "beta=0 must return the input tensor object unchanged (no computation)"
        assert m.last_output is None

    def test_beta_zero_set_after_init(self):
        m = make_module(beta=0.05)
        m.set_beta(0.0)
        ds, mem, mask, sem = make_inputs()
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert adapted is ds


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradients:
    def test_all_params_receive_gradients(self):
        m = make_module()
        m.train()
        ds, mem, mask, sem = make_inputs()
        ds  = ds.requires_grad_(False)
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        loss = adapted.sum()
        loss.backward()
        for name, p in m.named_parameters():
            if name == "beta":
                continue   # beta is a buffer, no grad
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"Zero/None gradient for parameter: {name}"

    def test_beta_buffer_has_no_grad(self):
        m = make_module()
        m.train()
        ds, mem, mask, sem = make_inputs()
        adapted = m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        adapted.sum().backward()
        assert m.beta.grad is None, "beta is a buffer and must not accumulate gradients"


# ---------------------------------------------------------------------------
# Diagnostic API
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_last_output_is_named_tuple(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert isinstance(m.last_output, DETRQueryCGPOutput)

    def test_clear_diagnostics_resets_to_none(self):
        m = make_module()
        ds, mem, mask, sem = make_inputs()
        m(ds, mem, mask, sem, video_length=VIDEO_LEN)
        assert m.last_output is not None
        m.clear_diagnostics()
        assert m.last_output is None
