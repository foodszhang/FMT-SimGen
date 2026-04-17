"""Green's function for infinite homogeneous medium (§3.2.1)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import OpticalParams

from .config import OPTICAL


def G_inf(r: np.ndarray, optical: OpticalParams = None) -> np.ndarray:
    """Point-source Green's function in infinite medium.

    G(r) = exp(-r/δ) / (4πDr)

    Parameters
    ----------
    r : array
        Distance(s) from source.
    optical : OpticalParams
        Optical parameters.

    Returns
    -------
    array
        Green's function values.
    """
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
    """Angle-averaged Green's function (§3.3.2).

    For a point at distance r' from ball center, observed at distance d
    from the ball center on the z-axis:

    Ḡ_∞(d, r') = 1/(4πD) · δ/(2dr') · [exp(-|d-r'|/δ) - exp(-(d+r')/δ)]

    Parameters
    ----------
    d : float
        Distance from ball center to observation point.
    rp : array
        Radial distances within the ball.
    optical : OpticalParams

    Returns
    -------
    array
        Angle-averaged Green's function values.
    """
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
