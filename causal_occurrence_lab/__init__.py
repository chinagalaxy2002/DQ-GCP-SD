"""Isolated occurrence-binding experiments for Sim-DETR.

The package deliberately lives outside :mod:`sim_detr`.  It imports the
production model, matcher, dataset and DQ-CGP implementation at runtime, but
keeps all experiment-only controls and diagnostics here.
"""

__version__ = "0.1.0"
