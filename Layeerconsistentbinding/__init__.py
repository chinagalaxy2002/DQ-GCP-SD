"""Layer-Consistent Binding (LCB-Full) for Sim-DETR.

This package provides:
- Training-only loss regularizer on native Sim-DETR decoder cross-attention (D1–D4).
- Ownership consistency and anti-washout supervision across decoder layers.
- Full layerwise ownership persistence and multi-occurrence retrieval metrics.
"""

from Layeerconsistentbinding.controls import (
    compute_layer_consistent_binding_losses,
    install_layer_consistent_binding_control,
    js_divergence,
    normalize_cross_attention,
    remove_layer_consistent_binding_control,
)
from Layeerconsistentbinding.metrics import (
    compute_layerwise_ownership_summary,
    compute_multi_occurrence_stratified_metrics,
)

__all__ = [
    "compute_layer_consistent_binding_losses",
    "compute_layerwise_ownership_summary",
    "compute_multi_occurrence_stratified_metrics",
    "install_layer_consistent_binding_control",
    "js_divergence",
    "normalize_cross_attention",
    "remove_layer_consistent_binding_control",
]
