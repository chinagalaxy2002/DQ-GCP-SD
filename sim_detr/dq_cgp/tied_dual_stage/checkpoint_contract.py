"""Checkpoint identity and configuration contract for the tied experiment."""

from __future__ import annotations

from collections.abc import Mapping

from .sim_detr_dq_cgp import ARCHITECTURE_ID


# These options either change where/how the shared module is applied or change
# DQ-CGP training semantics without necessarily changing state-dict shapes.
CHECKPOINT_CONFIG_KEYS = (
    "use_query_cgp",
    "dec_layers",
    "hidden_dim",
    "num_queries",
    "query_cgp_num_basis",
    "query_cgp_prompt_length",
    "query_cgp_router_hidden_dim",
    "query_cgp_frf_hidden_dim",
    "query_cgp_temperature",
    "query_cgp_beta",
    "query_cgp_after_layer",
    "query_cgp_late_after_layer",
    "query_cgp_binding_loss_coef",
    "query_cgp_route_loss_coef",
)


def _as_mapping(config):
    if isinstance(config, Mapping):
        return config
    try:
        return vars(config)
    except TypeError as exc:  # pragma: no cover - defensive API guard
        raise TypeError("current_config must be a mapping or option namespace") from exc


def validate_tied_checkpoint(checkpoint, current_config, operation: str) -> None:
    """Reject checkpoints whose tied architecture semantics do not match.

    Parameter tying intentionally makes the state keys identical to the
    single-stage model.  Strict ``load_state_dict`` therefore needs this
    explicit metadata/configuration check before it can be considered safe.
    """
    checkpoint_architecture = checkpoint.get("architecture_id")
    if checkpoint_architecture != ARCHITECTURE_ID:
        raise RuntimeError(
            f"{operation} requires a tied dual-stage DQ-CGP checkpoint with "
            f"architecture_id={ARCHITECTURE_ID!r}; got "
            f"{checkpoint_architecture!r}."
        )

    saved_config = checkpoint.get("model_cfg")
    if not isinstance(saved_config, Mapping):
        raise RuntimeError(
            f"{operation} requires checkpoint model_cfg metadata so tied "
            "insertion and DQ-CGP semantics can be verified."
        )
    current = _as_mapping(current_config)

    missing = [key for key in CHECKPOINT_CONFIG_KEYS if key not in saved_config]
    current_missing = [key for key in CHECKPOINT_CONFIG_KEYS if key not in current]
    if missing or current_missing:
        raise RuntimeError(
            f"{operation} cannot verify the tied configuration; "
            f"checkpoint missing={missing}, current options missing={current_missing}."
        )

    mismatches = {
        key: (saved_config[key], current[key])
        for key in CHECKPOINT_CONFIG_KEYS
        if saved_config[key] != current[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, current={active!r}"
            for key, (saved, active) in mismatches.items()
        )
        raise RuntimeError(
            f"{operation} tied DQ-CGP configuration mismatch: {details}"
        )


__all__ = ["CHECKPOINT_CONFIG_KEYS", "validate_tied_checkpoint"]
