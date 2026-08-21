"""Checkpoint identity tests for the tied all-interlayer experiment."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sim_detr.dq_cgp.tied_all_stage import inference_dq_cgp
from sim_detr.dq_cgp.tied_all_stage.sim_detr_dq_cgp import ARCHITECTURE_ID
from sim_detr.dq_cgp.tied_all_stage.checkpoint_contract import (
    CHECKPOINT_CONFIG_KEYS,
    validate_tied_checkpoint,
)
from sim_detr.dq_cgp.tied_all_stage.train_dq_cgp import _checkpoint_payload


EXPECTED_ARCHITECTURE_ID = "sim_detr_tied_all_stage_dq_cgp_v1"


def _contract_config(**overrides):
    values = {
        "use_query_cgp": True,
        "dec_layers": 4,
        "hidden_dim": 256,
        "num_queries": 10,
        "query_cgp_num_basis": 16,
        "query_cgp_prompt_length": 6,
        "query_cgp_router_hidden_dim": 256,
        "query_cgp_frf_hidden_dim": 512,
        "query_cgp_temperature": 1.0,
        "query_cgp_beta": 0.05,
        "query_cgp_after_layer": 0,
        "query_cgp_middle_after_layer": 1,
        "query_cgp_late_after_layer": 2,
        "query_cgp_binding_loss_coef": 0.2,
        "query_cgp_route_loss_coef": 0.01,
    }
    values.update(overrides)
    assert set(values) == set(CHECKPOINT_CONFIG_KEYS)
    return values


class TestCheckpointArchitectureContract:
    def test_architecture_id_is_stable(self):
        assert ARCHITECTURE_ID == EXPECTED_ARCHITECTURE_ID

    def test_saved_checkpoint_payload_contains_architecture_id(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        opt = SimpleNamespace(experiment="tied-test")

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            opt=opt,
            epoch=3,
            best_metric=0.4,
            es_cnt=2,
        )

        assert payload["architecture_id"] == EXPECTED_ARCHITECTURE_ID
        assert payload["model"].keys() == model.state_dict().keys()

    def test_contract_accepts_exact_config_and_rejects_semantic_mismatch(self):
        saved = _contract_config()
        checkpoint = {
            "architecture_id": EXPECTED_ARCHITECTURE_ID,
            "model_cfg": saved,
        }
        validate_tied_checkpoint(checkpoint, SimpleNamespace(**saved), "test")

        current = _contract_config(query_cgp_middle_after_layer=2)
        with pytest.raises(RuntimeError, match="middle_after_layer"):
            validate_tied_checkpoint(checkpoint, current, "test")

    def test_contract_rejects_missing_model_config(self):
        checkpoint = {"architecture_id": EXPECTED_ARCHITECTURE_ID}
        with pytest.raises(RuntimeError, match="model_cfg"):
            validate_tied_checkpoint(checkpoint, _contract_config(), "test")

    def test_inference_rejects_missing_or_single_stage_architecture_id(self):
        model = nn.Linear(3, 2)
        criterion = nn.Identity()
        opt = SimpleNamespace(
            use_query_cgp=True,
            device=torch.device("cpu"),
            resume="unused.ckpt",
            eval_disable_query_cgp_early=False,
            eval_disable_query_cgp_middle=False,
            eval_disable_query_cgp_late=False,
        )

        for bad_id in (None, "sim_detr_single_stage_dq_cgp_v3"):
            checkpoint = {"model": model.state_dict()}
            if bad_id is not None:
                checkpoint["architecture_id"] = bad_id
            with patch.object(
                inference_dq_cgp,
                "build_model_dq_cgp",
                return_value=(model, criterion),
            ), patch.object(
                inference_dq_cgp.torch,
                "load",
                return_value=checkpoint,
            ):
                with pytest.raises(RuntimeError, match="architecture_id"):
                    inference_dq_cgp.setup_model_dq_cgp_inference(opt)
