"""DQ-CGP: DETR-Query Conditioned Compositional Generalization Prompting.

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
    Decoder Layer 2 → 3 → 4
         │
         ▼
    class / span / iou / mask heads

Usage
-----
From a training script::

    from sim_detr.dq_cgp.model_builder import add_dq_cgp_args, build_model_dq_cgp

    add_dq_cgp_args(parser)          # extends Sim-DETR's BaseOptions parser
    opt = parser.parse_args()
    model, criterion = build_model_dq_cgp(opt)

Or directly::

    from sim_detr.dq_cgp import SimDETRWithDQCGP, DETRQueryCGP
"""

from .query_cgp import DETRQueryCGP, DETRQueryCGPOutput, DQCGP
from .decoder_wrapper import (
    DQCGPDecoderWrapper,
    install_dq_cgp_decoder_hook,
    remove_dq_cgp_decoder_hook,
    set_dq_cgp_adapter_kwargs,
)
from .sim_detr_dq_cgp import SimDETRWithDQCGP
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
    "remove_dq_cgp_decoder_hook",
    "set_dq_cgp_adapter_kwargs",
    "SimDETRWithDQCGP",
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
