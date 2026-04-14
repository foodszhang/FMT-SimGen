#!/usr/bin/env python3
"""Green function kernels for E1d.

Extends E1c kernels to support finite-size sources via superposition.
"""

import numpy as np
from typing import Tuple, Optional


def diffusion_params(tissue_params: dict) -> dict:
    """Compute diffusion parameters from tissue optical properties."""
    mua = tissue_params["mua_mm"]
    mus = tissue_params["mus_mm"]
    g = tissue_params.get("g", 0.9)
    n = tissue_params.get("n", 1.37)

    mus_prime = mus * (1 - g)
    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(3.0 * mua * (mua + mus_prime))
    l_star = 1.0 / (mua + mus_prime)

    R_eff = 0.493
    A = (1 + R_eff) / (1 - R_eff)
    zb = 2 * A * D

    return {
        "D": D,
        "mu_eff": mu_eff,
        "mus_prime": mus_prime,
        "l_star": l_star,
        "A": A,
        "zb": zb,
        "n": n,
    }


def compute_source_surface_distances(
    surface_points_mm: np.ndarray,
    source_points_mm: np.ndarray,
    z_surface: float = 10.0,
    surface_z_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distances from surface points to source points.

    Args:
        surface_points_mm: [N_surf, 2] or [H, W, 2] surface XY coordinates
        source_points_mm: [N_src, 3] source point positions
        z_surface: z-coordinate of surface plane (used if surface_z_values is None)
        surface_z_values: [N_surf] actual z values at each surface point (for non-planar surface)

    Returns:
        r: [N_surf, N_src] total 3D distances
        rho: [N_surf, N_src] in-plane radial distances
        depth: [N_src] depth of each source point from surface
    """
    if surface_points_mm.ndim == 3:
        surface_xy = surface_points_mm.reshape(-1, 2)
    else:
        surface_xy = surface_points_mm

    n_surf = len(surface_xy)
    n_src = len(source_points_mm)

    surface_xy = surface_xy[:, np.newaxis, :]
    source_xy = source_points_mm[np.newaxis, :, :2]

    dx = surface_xy[:, :, 0] - source_xy[:, :, 0]
    dy = surface_xy[:, :, 1] - source_xy[:, :, 1]
    rho = np.sqrt(dx**2 + dy**2)

    if surface_z_values is not None:
        if surface_z_values.ndim > 1:
            surface_z_flat = surface_z_values.reshape(-1)
        else:
            surface_z_flat = surface_z_values
        depth = surface_z_flat[:, np.newaxis] - source_points_mm[np.newaxis, :, 2]
    else:
        depth = z_surface - source_points_mm[:, 2]
        depth = depth[np.newaxis, :]

    r = np.sqrt(rho**2 + depth**2)

    if surface_z_values is None:
        depth = depth[0, :]

    return r, rho, depth


def green_infinite_finite_source(
    surface_points_mm: np.ndarray,
    source_points_mm: np.ndarray,
    source_weights: np.ndarray,
    tissue_params: dict,
    z_surface: float = 10.0,
    surface_z_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Infinite medium Green's function for finite-size source.

    Superposition of point source responses weighted by source_weights.

    G_total(s) = sum_i w_i * G_inf(r_i)

    Args:
        surface_points_mm: [N_surf, 2] XY coordinates on surface
        source_points_mm: [N_src, 3] sampled source point positions
        source_weights: [N_src] weight for each sample point
        tissue_params: optical properties
        z_surface: z-coordinate of surface plane (used if surface_z_values is None)
        surface_z_values: [N_surf] actual z values at each surface point

    Returns:
        response: [N_surf] total surface response
    """
    diff = diffusion_params(tissue_params)
    D = diff["D"]
    mu_eff = diff["mu_eff"]

    r, _, _ = compute_source_surface_distances(
        surface_points_mm, source_points_mm, z_surface, surface_z_values
    )

    r = np.maximum(r, 1e-6)

    G = np.exp(-mu_eff * r) / (4 * np.pi * D * r)

    response = np.sum(G * source_weights[np.newaxis, :], axis=1)

    return response


def green_halfspace_finite_source(
    surface_points_mm: np.ndarray,
    source_points_mm: np.ndarray,
    source_weights: np.ndarray,
    tissue_params: dict,
    z_surface: float = 10.0,
    boundary_condition: str = "extrapolated",
    surface_z_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Half-space Green's function for finite-size source.

    Uses image source method for each sampled point.

    G_halfspace = sum_i w_i * [G(r1_i) - G(r2_i)]

    Args:
        surface_points_mm: [N_surf, 2] XY coordinates on surface
        source_points_mm: [N_src, 3] sampled source point positions
        source_weights: [N_src] weight for each sample point
        tissue_params: optical properties
        z_surface: z-coordinate of surface plane (used if surface_z_values is None)
        boundary_condition: "extrapolated" or "zero"
        surface_z_values: [N_surf] actual z values at each surface point

    Returns:
        response: [N_surf] total surface response
    """
    diff = diffusion_params(tissue_params)
    D = diff["D"]
    mu_eff = diff["mu_eff"]
    zb = diff["zb"]

    r, rho, depth = compute_source_surface_distances(
        surface_points_mm, source_points_mm, z_surface, surface_z_values
    )

    r = np.maximum(r, 1e-6)
    G1 = np.exp(-mu_eff * r) / (4 * np.pi * D * r)

    if boundary_condition == "extrapolated":
        r2 = np.sqrt(rho**2 + (depth + 2 * zb) ** 2)
    elif boundary_condition == "zero":
        r2 = np.sqrt(rho**2 + (2 * depth) ** 2)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")

    r2 = np.maximum(r2, 1e-6)
    G2 = np.exp(-mu_eff * r2) / (4 * np.pi * D * r2)

    G_halfspace = G1 - G2
    G_halfspace = np.maximum(G_halfspace, 0.0)

    response = np.sum(G_halfspace * source_weights[np.newaxis, :], axis=1)

    return response


def gaussian_psf_finite_source(
    surface_points_mm: np.ndarray,
    source_points_mm: np.ndarray,
    source_weights: np.ndarray,
    tissue_params: dict,
    sigma_mm: float = 2.0,
    z_surface: float = 10.0,
) -> np.ndarray:
    """Gaussian 2D PSF baseline for finite-size source.

    Simply projects all source points to surface and applies 2D Gaussian.

    Args:
        surface_points_mm: [N_surf, 2] XY coordinates on surface
        source_points_mm: [N_src, 3] sampled source point positions
        source_weights: [N_src] weight for each sample point
        tissue_params: optical properties (unused, kept for API consistency)
        sigma_mm: Gaussian width
        z_surface: unused, kept for API consistency

    Returns:
        response: [N_surf] total surface response
    """
    if surface_points_mm.ndim == 3:
        surface_xy = surface_points_mm.reshape(-1, 2)
    else:
        surface_xy = surface_points_mm

    n_surf = len(surface_xy)
    n_src = len(source_points_mm)

    surface_xy = surface_xy[:, np.newaxis, :]
    source_xy = source_points_mm[np.newaxis, :, :2]

    dx = surface_xy[:, :, 0] - source_xy[:, :, 0]
    dy = surface_xy[:, :, 1] - source_xy[:, :, 1]
    rho_sq = dx**2 + dy**2

    G = np.exp(-0.5 * rho_sq / sigma_mm**2) / (2 * np.pi * sigma_mm**2)

    response = np.sum(G * source_weights[np.newaxis, :], axis=1)

    return response


def get_kernel_function(kernel_name: str):
    """Get kernel function by name.

    Args:
        kernel_name: "gaussian_psf", "green_infinite", or "green_halfspace"

    Returns:
        kernel function for finite-size sources
    """
    kernels = {
        "gaussian_psf": gaussian_psf_finite_source,
        "green_infinite": green_infinite_finite_source,
        "green_halfspace": green_halfspace_finite_source,
    }
    if kernel_name not in kernels:
        raise ValueError(
            f"Unknown kernel: {kernel_name}. Available: {list(kernels.keys())}"
        )
    return kernels[kernel_name]
