"""Green's function for infinite homogeneous medium (§3.2.1).

Standalone copy for MVP pipeline to avoid import issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import OpticalParams

from .config import OPTICAL


def G_inf(r: np.ndarray, optical: OpticalParams = None) -> np.ndarray:
    if optical is None:
        optical = OPTICAL

    r = np.asarray(r, dtype=np.float64)
    scalar_input = r.ndim == 0
    r = np.atleast_1d(r)

    result = np.zeros_like(r)
    mask = r > 1e-10
    result[mask] = np.exp(-r[mask] / optical.delta) / (4 * np.pi * optical.D * r[mask])

    if scalar_input:
        return result[0]
    return result


def G_bar_angle_averaged(
    d: float, rp: np.ndarray, optical: OpticalParams = None
) -> np.ndarray:
    if optical is None:
        optical = OPTICAL

    rp = np.asarray(rp, dtype=np.float64)
    delta = optical.delta
    D = optical.D

    result = np.zeros_like(rp)
    mask = rp > 1e-10

    term1 = np.exp(-np.abs(d - rp[mask]) / delta)
    term2 = np.exp(-(d + rp[mask]) / delta)

    result[mask] = (
        (1.0 / (4 * np.pi * D)) * (delta / (2 * d * rp[mask])) * (term1 - term2)
    )

    return result
