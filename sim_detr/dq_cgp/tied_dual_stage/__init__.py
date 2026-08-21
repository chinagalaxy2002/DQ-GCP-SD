"""Parameter-tied dual-stage DQ-CGP experiment for Sim-DETR.

This subpackage integrates DQ-CGP into Sim-DETR as an inter-decoder-layer
adapter.  **No original Sim-DETR files are modified.**

Architecture
------------
::

    Sim-DETR Encoder
         │
         ▼
    video memory Mv
         │
         ▼
    Decoder Layer 1  (native Sim-DETR)
         │
         ├── reference refinement R1
         ├── aux supervision on native H1
         │
         ▼
    ┌──────────────┐
    │   DQ-CGP     │  Temporal Binding → RCG → BPS → FRF → β residual
    └──────────────┘
         │
         ▼
    adapted H1~
         │
         ▼
    Decoder Layer 2 → 3
         │
         ▼
    DQ-CGP (same parameters reused)
         │
         ▼
    Decoder Layer 4
         │
         ▼
    class / span / iou / mask heads

Usage
-----
From a training script::

    from sim_detr.dq_cgp.tied_dual_stage.model_builder import (
        add_dq_cgp_args, build_model_dq_cgp,
    )

    add_dq_cgp_args(parser)          # extends Sim-DETR's BaseOptions parser
    opt = parser.parse_args()
    model, criterion = build_model_dq_cgp(opt)

Or directly::

    from sim_detr.dq_cgp.tied_dual_stage import (
        SimDETRWithTiedDualDQCGP, DETRQueryCGP,
    )
"""

from .query_cgp import DETRQueryCGP, DETRQueryCGPOutput, DQCGP
from .decoder_wrapper import (
    DQCGPDecoderWrapper,
    install_dq_cgp_decoder_hook,
    install_dq_cgp_decoder_hooks,
    remove_dq_cgp_decoder_hook,
    set_dq_cgp_adapter_kwargs,
    set_dq_cgp_disabled_layers,
    get_dq_cgp_stage_outputs,
)
from .sim_detr_dq_cgp import ARCHITECTURE_ID, SimDETRWithTiedDualDQCGP
from .checkpoint_contract import CHECKPOINT_CONFIG_KEYS, validate_tied_checkpoint
from .utils import masked_mean_pool
from .loss import compute_binding_loss, compute_routing_loss, loss_query_cgp
from .model_builder import add_dq_cgp_args, build_model_dq_cgp, DQCGPSetCriterion

__all__ = [
    # Core module
    "DETRQueryCGP",
    "DETRQueryCGPOutput",
    "DQCGP",
    # Architecture
    "DQCGPDecoderWrapper",
    "install_dq_cgp_decoder_hook",
    "install_dq_cgp_decoder_hooks",
    "remove_dq_cgp_decoder_hook",
    "set_dq_cgp_adapter_kwargs",
    "set_dq_cgp_disabled_layers",
    "get_dq_cgp_stage_outputs",
    "ARCHITECTURE_ID",
    "CHECKPOINT_CONFIG_KEYS",
    "validate_tied_checkpoint",
    "SimDETRWithTiedDualDQCGP",
    # Loss
    "compute_binding_loss",
    "compute_routing_loss",
    "loss_query_cgp",
    "DQCGPSetCriterion",
    # Builder
    "add_dq_cgp_args",
    "build_model_dq_cgp",
    # Utils
    "masked_mean_pool",
]
