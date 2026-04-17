"""Closed-form Gaussian source using analytic slice + 2D FFT (§3.3.3).

Ground truth for Gaussian splat comparison.
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

from .config import (
    OPTICAL,
    GAUSSIAN_SIGMAS_MM,
    GAUSSIAN_D_DELTAS,
    GRID_SIZE,
    WINDOW_DELTA,
)
from .green import G_inf


def gaussian_2d(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """2D Gaussian.

    Parameters
    ----------
    x, y : arrays
        Coordinates.
    sigma : float
        Standard deviation.

    Returns
    -------
    array
        Gaussian values.
    """
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def closed_form_gaussian_response_fft(
    sigma: float,
    d: float,
    grid_size: int = GRID_SIZE,
    window_delta: float = WINDOW_DELTA,
    optical=None,
    n_z_slices: int = 64,
) -> np.ndarray:
    """Closed-form Gaussian source response using layered FFT approach.

    For each z-slice of the Gaussian, compute marginal 2D Gaussian and
    convolve with depth-dependent Green's kernel via FFT.

    Parameters
    ----------
    sigma : float
        Gaussian sigma (mm).
    d : float
        Observation distance (mm).
    grid_size : int
        Grid resolution.
    window_delta : float
        Window size in units of δ.
    optical : OpticalParams
    n_z_slices : int
        Number of z-slices for integration.

    Returns
    -------
    np.ndarray
        Surface response [grid_size, grid_size].
    """
    if optical is None:
        optical = OPTICAL

    window = window_delta * optical.delta
    dx = 2 * window / grid_size
    x = np.linspace(-window, window, grid_size)
    y = np.linspace(-window, window, grid_size)
    X, Y = np.meshgrid(x, y)

    z_max = 4 * sigma
    z = np.linspace(-z_max, z_max, n_z_slices)
    dz = z[1] - z[0]

    kx = fftfreq(grid_size, dx) * 2 * np.pi
    ky = fftfreq(grid_size, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    response = np.zeros((grid_size, grid_size))

    for z_k in z:
        sigma_z = sigma
        gaussian_2d_k = np.exp(-0.5 * sigma_z**2 * K2)

        r_k = np.sqrt(d**2 + z_k**2)
        G_k = G_inf(r_k, optical)

        kernel_k = np.exp(-0.5 * r_k**2 * K2 / optical.delta**2) / (
            4 * np.pi * optical.D
        )
        kernel_k_fft = fft2(kernel_k)

        slice_fft = fft2(gaussian_2d(X, Y, sigma_z)) * kernel_k_fft
        slice_response = np.real(ifft2(slice_fft))

        weight = np.exp(-(z_k**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        response += weight * slice_response * dz

    return response


def gaussian_source_amplitude_z(z: float, sigma: float) -> float:
    """Marginal amplitude of 3D Gaussian at height z.

    The 3D Gaussian N(x,y,z) = exp(-(x²+y²+z²)/(2σ²)) / (2πσ²)^(3/2)
    has marginal in z: ∫∫ N dx dy = exp(-z²/(2σ²)) / √(2πσ²)

    Parameters
    ----------
    z : float
        Height.
    sigma : float
        Standard deviation.

    Returns
    -------
    float
        Marginal amplitude.
    """
    return np.exp(-(z**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def closed_form_gaussian_point(
    d: float,
    sigma: float,
    optical=None,
    n_z_quad: int = 64,
) -> float:
    """Point-wise Gaussian response at distance d on axis.

    Uses 1D Gauss-Legendre quadrature over z.

    Parameters
    ----------
    d : float
        Distance from source center (mm).
    sigma : float
        Gaussian sigma (mm).
    optical : OpticalParams
    n_z_quad : int
        Number of z quadrature points.

    Returns
    -------
    float
        Integrated response at distance d.
    """
    if optical is None:
        optical = OPTICAL

    from numpy.polynomial.legendre import leggauss

    z_max = 5 * sigma
    nodes, weights = leggauss(n_z_quad)
    z = 0.5 * z_max * (nodes + 1.0)
    z = np.concatenate([-z[::-1], z])
    w = 0.5 * z_max * np.concatenate([weights[::-1], weights])

    response = 0.0
    for zi, wi in zip(z, w):
        r = np.sqrt(d**2 + zi**2)
        G = G_inf(r, optical)
        amp = gaussian_source_amplitude_z(zi, sigma)
        response += wi * amp * G

    return response


def run_gaussian_sweep(
    sigmas: list = None,
    d_deltas: list = None,
    optical=None,
    n_z_slices: int = 64,
) -> dict:
    """Run sweep over σ × d combinations.

    Parameters
    ----------
    sigmas : list
        Gaussian sigmas in mm.
    d_deltas : list
        Observation distances in units of δ.
    optical : OpticalParams
    n_z_slices : int

    Returns
    -------
    dict
        Results.
    """
    if sigmas is None:
        sigmas = GAUSSIAN_SIGMAS_MM
    if d_deltas is None:
        d_deltas = GAUSSIAN_D_DELTAS
    if optical is None:
        optical = OPTICAL

    results = {
        "sigmas_mm": sigmas,
        "d_deltas": d_deltas,
        "delta_mm": optical.delta,
        "responses": {},
    }

    for sigma in sigmas:
        for d_delta in d_deltas:
            d = d_delta * optical.delta
            I = closed_form_gaussian_point(d, sigma, optical, n_z_slices)
            key = f"sigma{sigma:.1f}_d{d_delta}delta"
            results["responses"][key] = {
                "sigma_mm": sigma,
                "d_mm": d,
                "d_delta": d_delta,
                "response": I,
            }

    return results
