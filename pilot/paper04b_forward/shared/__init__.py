"""Shared modules for Paper-04b MVP pipeline."""

from .config import OpticalParams, MVPConfig, OPTICAL
from .sources import SourceSpec
from .forward_closed import forward_closed_source
from .metrics import compute_all_metrics, metrics_summary, ncc, scale_factor_k, rmse
from .preflight import assert_voxel_consistency, preflight_check

__all__ = [
    "OpticalParams",
    "MVPConfig",
    "OPTICAL",
    "SourceSpec",
    "forward_closed_source",
    "compute_all_metrics",
    "metrics_summary",
    "ncc",
    "scale_factor_k",
    "rmse",
    "assert_voxel_consistency",
    "preflight_check",
]
