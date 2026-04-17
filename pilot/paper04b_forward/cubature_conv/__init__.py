"""Cubature convergence study (Part C, §4.A extension).

Comparison of cubature schemes vs closed-form ground truth for:
- Ball source (uniform intensity sphere)
- Gaussian source (3D Gaussian splat)

Deliverables:
- Figure 4-A-1: NCC heatmap (scheme × r/δ)
- Table 4-A-2/4-A-3: R × N-points, σ × N-points lookup
"""

from .config import OPTICAL, OpticalParams
from .green import G_inf, G_bar_angle_averaged
from .closed_form_ball import closed_form_ball_response, run_ball_sweep as run_ball_gt
from .closed_form_gaussian_fft import (
    closed_form_gaussian_point,
    run_gaussian_sweep as run_gaussian_gt,
)
from .cubature_schemes import get_scheme, SCHEME_FUNCTIONS
from .run_ball_sweep import run_ball_sweep
from .run_gaussian_sweep import run_gaussian_sweep
from .plot_ncc_heatmap import (
    plot_ball_heatmap,
    plot_gaussian_heatmap,
    plot_critical_r_over_delta,
)

__all__ = [
    "OPTICAL",
    "OpticalParams",
    "G_inf",
    "G_bar_angle_averaged",
    "closed_form_ball_response",
    "closed_form_gaussian_point",
    "get_scheme",
    "SCHEME_FUNCTIONS",
    "run_ball_sweep",
    "run_gaussian_sweep",
    "run_ball_gt",
    "run_gaussian_gt",
    "plot_ball_heatmap",
    "plot_gaussian_heatmap",
    "plot_critical_r_over_delta",
]
