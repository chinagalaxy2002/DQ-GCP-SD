"""Late-Semantic DQ-CGP for the two-layer Sim-DETR decoder.

Model-level symbols are loaded lazily so the standalone CGP component does not
require all optional Sim-DETR dependencies at import time.
"""

from .cgp_module import LateSemanticCGP, LSDQCGPOutput

__all__ = [
    "LateSemanticCGP",
    "LSDQCGPOutput",
    "SimDETRWithLSDQCGP",
    "LSDQCGPSetCriterion",
    "add_ls_dq_cgp_args",
    "build_model_ls_dq_cgp",
]


def __getattr__(name):
    if name == "SimDETRWithLSDQCGP":
        from .model import SimDETRWithLSDQCGP
        return SimDETRWithLSDQCGP
    if name in {"LSDQCGPSetCriterion", "add_ls_dq_cgp_args", "build_model_ls_dq_cgp"}:
        from . import model_builder
        return getattr(model_builder, name)
    raise AttributeError(name)
