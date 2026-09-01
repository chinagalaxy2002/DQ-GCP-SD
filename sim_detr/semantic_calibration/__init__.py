"""Candidate-conditioned semantic calibration for the native Sim-DETR model."""

from .semantic_calibrator import (
    CandidateSemanticCalibrator,
    SemanticCalibrationOutput,
    normalize_evidence_weights,
    pool_video_evidence,
)
from .semantic_model import SimDETRWithSemanticCalibration

__all__ = [
    "CandidateSemanticCalibrator",
    "SemanticCalibrationOutput",
    "SimDETRWithSemanticCalibration",
    "normalize_evidence_weights",
    "pool_video_evidence",
]
