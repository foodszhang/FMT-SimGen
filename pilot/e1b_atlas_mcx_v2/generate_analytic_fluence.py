"""Generate analytic Green's function 3D fluence volume.

For comparison with MCX simulation output.
Uses semi-infinite medium Green's function with extrapolated boundary condition.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_diffusion_params(tissue_params: dict) -> dict:
    """Compute diffusion parameters from tissue properties.

    Parameters
    ----------
    tissue_params : dict
        {mua_mm, mus_prime_mm, g, n}

    Returns
    -------
    dict
        {D, mu_eff, z_b, R_eff, A}
    """
    mua = tissue_params["mua_mm"]
    mus_prime = tissue_params["mus_prime_mm"]
    n = tissue_params.get("n", 1.37)

    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(mua / D)

    R_eff = 0.493  # for n=1.37 tissue/air interface
    A = (1 + R_eff) / (1 - R_eff)
    z_b = 2 * A * D

    return {
        "D": D,
        "mu_eff": mu_eff,
        "z_b": z_b,
        "R_eff": R_eff,
        "A": A,
    }


def green_semi_infinite_point_source(
    source_pos_mm: np.ndarray,
    volume_shape_xyz: Tuple[int, int, int],
    voxel_size_mm: float,
    origin_mm: np.ndarray,
    tissue_params: dict,
    surface_z_mm: float = 0.0,
) -> np.ndarray:
    """Generate 3D fluence volume from point source using semi-infinite Green's function.

    Uses extrapolated boundary condition (EBC) with image source.
    Surface is at z = surface_z_mm (default 0).

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position [x, y, z] in mm.
    volume_shape_xyz : tuple
        Volume shape (nx, ny, nz).
    voxel_size_mm : float
        Voxel size in mm.
    origin_mm : np.ndarray
        Volume origin [x0, y0, z0] in mm.
    tissue_params : dict
        Tissue parameters {mua_mm, mus_prime_mm, g, n}.
    surface_z_mm : float
        Z coordinate of the surface (default 0).

    Returns
    -------
    np.ndarray
        3D fluence volume [nx, ny, nz].
    """
    diff_params = compute_diffusion_params(tissue_params)
    D = diff_params["D"]
    mu_eff = diff_params["mu_eff"]
    z_b = diff_params["z_b"]

    nx, ny, nz = volume_shape_xyz

    x = np.arange(nx) * voxel_size_mm + origin_mm[0]
    y = np.arange(ny) * voxel_size_mm + origin_mm[1]
    z = np.arange(nz) * voxel_size_mm + origin_mm[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    dx = xx - source_pos_mm[0]
    dy = yy - source_pos_mm[1]

    d = source_pos_mm[2] - surface_z_mm  # source depth from surface

    r1 = np.sqrt(dx**2 + dy**2 + (zz - source_pos_mm[2]) ** 2)
    r2 = np.sqrt(dx**2 + dy**2 + (zz - (source_pos_mm[2] + 2 * z_b)) ** 2)

    r1 = np.maximum(r1, voxel_size_mm)
    r2 = np.maximum(r2, voxel_size_mm)

    G = (np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * r2) / r2) / (4 * np.pi * D)
    G = np.maximum(G, 0).astype(np.float32)

    return G


def green_infinite_point_source(
    source_pos_mm: np.ndarray,
    volume_shape_xyz: Tuple[int, int, int],
    voxel_size_mm: float,
    origin_mm: np.ndarray,
    tissue_params: dict,
) -> np.ndarray:
    """Generate 3D fluence volume from point source using infinite medium Green's function.

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position [x, y, z] in mm.
    volume_shape_xyz : tuple
        Volume shape (nx, ny, nz).
    voxel_size_mm : float
        Voxel size in mm.
    origin_mm : np.ndarray
        Volume origin [x0, y0, z0] in mm.
    tissue_params : dict
        Tissue parameters {mua_mm, mus_prime_mm, g, n}.

    Returns
    -------
    np.ndarray
        3D fluence volume [nx, ny, nz].
    """
    mua = tissue_params["mua_mm"]
    mus_prime = tissue_params["mus_prime_mm"]

    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(mua / D)

    nx, ny, nz = volume_shape_xyz

    x = np.arange(nx) * voxel_size_mm + origin_mm[0]
    y = np.arange(ny) * voxel_size_mm + origin_mm[1]
    z = np.arange(nz) * voxel_size_mm + origin_mm[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    r = np.sqrt(
        (xx - source_pos_mm[0]) ** 2
        + (yy - source_pos_mm[1]) ** 2
        + (zz - source_pos_mm[2]) ** 2
    )
    r = np.maximum(r, voxel_size_mm)

    fluence = np.exp(-mu_eff * r) / (4.0 * np.pi * D * r)
    fluence = fluence.astype(np.float32)

    return fluence


def green_halfspace_multi_point(
    source_points_mm: np.ndarray,
    source_weights: np.ndarray,
    volume_shape_xyz: Tuple[int, int, int],
    voxel_size_mm: float,
    origin_mm: np.ndarray,
    tissue_params: dict,
) -> np.ndarray:
    """Generate 3D fluence volume from multiple point sources.

    Parameters
    ----------
    source_points_mm : np.ndarray
        Source positions [N, 3] in mm.
    source_weights : np.ndarray
        Source weights [N].
    volume_shape_xyz : tuple
        Volume shape (nx, ny, nz).
    voxel_size_mm : float
        Voxel size in mm.
    origin_mm : np.ndarray
        Volume origin [x0, y0, z0] in mm.
    tissue_params : dict
        Tissue parameters {mua_mm, mus_prime_mm, g, n}.

    Returns
    -------
    np.ndarray
        3D fluence volume [nx, ny, nz].
    """
    mua = tissue_params["mua_mm"]
    mus_prime = tissue_params["mus_prime_mm"]
    g = tissue_params.get("g", 0.9)

    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(3.0 * mua * (mua + mus_prime))

    nx, ny, nz = volume_shape_xyz

    x = np.arange(nx) * voxel_size_mm + origin_mm[0]
    y = np.arange(ny) * voxel_size_mm + origin_mm[1]
    z = np.arange(nz) * voxel_size_mm + origin_mm[2]

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    fluence = np.zeros((nx, ny, nz), dtype=np.float32)

    for i, (pt, w) in enumerate(zip(source_points_mm, source_weights)):
        r = np.sqrt((xx - pt[0]) ** 2 + (yy - pt[1]) ** 2 + (zz - pt[2]) ** 2)
        r = np.maximum(r, 1e-6)
        fluence += (w * np.exp(-mu_eff * r) / (4.0 * np.pi * D * r)).astype(np.float32)

        if (i + 1) % 10 == 0:
            logger.debug(f"Processed {i + 1}/{len(source_points_mm)} points")

    return fluence


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare (will be flattened).

    Returns
    -------
    float
        NCC value in [-1, 1].
    """
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    a_mean = a_flat.mean()
    b_mean = b_flat.mean()

    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean

    num = np.dot(a_centered, b_centered)
    denom = np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered))

    if denom < 1e-10:
        return 0.0

    return float(num / denom)


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute root mean squared error.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))
