"""Closed-form ball integral using angle-averaged Green's function (§3.3.2).

Ground truth for ball source comparison.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import OPTICAL, BALL_RADII_MM, BALL_D_DELTAS
from .green import G_bar_angle_averaged


def closed_form_ball_response(
    d: float,
    R: float,
    optical=None,
    n_quad: int = 10,
) -> float:
    """Closed-form integral of Green's function over a uniform ball source.

    Uses angle-averaged Green's function and 1D radial Gauss-Legendre quadrature.

    I(d) = ∫_0^R 4πr'² · Ḡ_∞(d, r') dr'

    Parameters
    ----------
    d : float
        Distance from ball center to observation point (mm).
    R : float
        Ball radius (mm).
    optical : OpticalParams
        Optical parameters.
    n_quad : int
        Number of Gauss-Legendre quadrature points.

    Returns
    -------
    float
        Integrated response.
    """
    if optical is None:
        optical = OPTICAL

    nodes, weights = leggauss(n_quad)
    rp = 0.5 * R * (nodes + 1.0)
    wp = 0.5 * R * weights

    Gbar = G_bar_angle_averaged(d, rp, optical)

    integrand = 4 * np.pi * rp**2 * Gbar

    return np.sum(wp * integrand)


def closed_form_ball_surface(
    R: float,
    d_delta: int,
    grid_size: int = 256,
    window_delta: float = 10.0,
    optical=None,
    n_quad: int = 10,
) -> np.ndarray:
    """Compute closed-form ball response on a surface grid.

    Parameters
    ----------
    R : float
        Ball radius (mm).
    d_delta : int
        Observation distance in units of δ (3, 5, or 10).
    grid_size : int
        Grid resolution.
    window_delta : float
        Window size in units of δ.
    optical : OpticalParams
    n_quad : int

    Returns
    -------
    np.ndarray
        Surface response [grid_size, grid_size].
    """
    if optical is None:
        optical = OPTICAL

    d = d_delta * optical.delta
    window = window_delta * optical.delta

    x = np.linspace(-window, window, grid_size)
    y = np.linspace(-window, window, grid_size)
    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2)
    dist = np.sqrt(d**2 + rho**2)

    response = np.zeros_like(dist)
    for i in range(grid_size):
        for j in range(grid_size):
            if dist[i, j] > 1e-10:
                response[i, j] = closed_form_ball_response(
                    dist[i, j], R, optical, n_quad
                )

    return response


def run_ball_sweep(
    radii: list = None,
    d_deltas: list = None,
    optical=None,
    n_quad: int = 10,
) -> dict:
    """Run sweep over R × d combinations.

    Parameters
    ----------
    radii : list
        Ball radii in mm.
    d_deltas : list
        Observation distances in units of δ.
    optical : OpticalParams
    n_quad : int

    Returns
    -------
    dict
        Results with keys: radii, d_deltas, responses.
    """
    if radii is None:
        radii = BALL_RADII_MM
    if d_deltas is None:
        d_deltas = BALL_D_DELTAS
    if optical is None:
        optical = OPTICAL

    results = {
        "radii_mm": radii,
        "d_deltas": d_deltas,
        "delta_mm": optical.delta,
        "responses": {},
    }

    for R in radii:
        for d_delta in d_deltas:
            d = d_delta * optical.delta
            I = closed_form_ball_response(d, R, optical, n_quad)
            key = f"R{R:.1f}_d{d_delta}delta"
            results["responses"][key] = {
                "R_mm": R,
                "d_mm": d,
                "d_delta": d_delta,
                "response": I,
            }

    return results
