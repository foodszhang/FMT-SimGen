#!/usr/bin/env python3
"""Green function kernels for E1c.

Implements three forward models for surface response:
1. gaussian_2d: Empirical 2D Gaussian baseline
2. green_infinite: Infinite medium Green's function
3. green_halfspace: Half-space Green's function with image source
"""

import numpy as np
from typing import Tuple


def diffusion_params(tissue_params: dict) -> dict:
    """Compute diffusion parameters from tissue optical properties.

    Args:
        tissue_params: dict with mua_mm, mus_mm, g, n

    Returns:
        dict with D, mu_eff, l_star, A, zb
    """
    mua = tissue_params["mua_mm"]  # absorption [1/mm]
    mus = tissue_params["mus_mm"]  # scattering [1/mm]
    g = tissue_params.get("g", 0.9)
    n = tissue_params.get("n", 1.37)

    mus_prime = mus * (1 - g)  # reduced scattering [1/mm]
    D = 1.0 / (3.0 * (mua + mus_prime))  # diffusion coefficient [mm]
    mu_eff = np.sqrt(3.0 * mua * (mua + mus_prime))  # effective attenuation [1/mm]
    l_star = 1.0 / (mua + mus_prime)  # transport mean free path [mm]

    R_eff = 0.493 if abs(n - 1.37) < 0.1 else 0.493
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


def source_depth_from_surface(
    source_world: np.ndarray, z_surface: float = 10.0
) -> float:
    """Compute source depth from surface plane.

    Args:
        source_world: [x, y, z] world coordinates of source
        z_surface: z-coordinate of surface plane (default 10.0 for dorsal)

    Returns:
        depth in mm (positive = below surface)
    """
    return z_surface - source_world[2]


def radial_distance_to_source_projection(
    x_grid_mm: np.ndarray,
    y_grid_mm: np.ndarray,
    source_world: np.ndarray,
) -> np.ndarray:
    """Compute radial distance from each surface point to source XY projection.

    This gives the in-plane distance for Gaussian 2D kernel.

    Args:
        x_grid_mm: 2D array of x coordinates on surface [H, W]
        y_grid_mm: 2D array of y coordinates on surface [H, W]
        source_world: [x, y, z] world coordinates of source

    Returns:
        rho: radial distance in mm [H, W]
    """
    dx = x_grid_mm - source_world[0]
    dy = y_grid_mm - source_world[1]
    rho = np.sqrt(dx**2 + dy**2)
    return rho


def gaussian_2d_surface(
    x_grid_mm: np.ndarray,
    y_grid_mm: np.ndarray,
    source_world: np.ndarray,
    tissue_params: dict,
    calib_params: dict | None = None,
) -> np.ndarray:
    """2D Gaussian kernel for surface response.

    This is the empirical baseline currently used in E0/E1.

    Args:
        x_grid_mm: 2D array of x coordinates [H, W]
        y_grid_mm: 2D array of y coordinates [H, W]
        source_world: [x, y, z] source position in world coords
        tissue_params: optical properties
        calib_params: calibration parameters (sigma_mm, etc.)

    Returns:
        surface_response: 2D array [H, W]
    """
    if calib_params is None:
        calib_params = {}

    sigma_mm = calib_params.get("sigma_mm", 2.0)

    rho = radial_distance_to_source_projection(x_grid_mm, y_grid_mm, source_world)

    response = np.exp(-0.5 * (rho / sigma_mm) ** 2)

    response = response / (2 * np.pi * sigma_mm**2)

    return response


def green_infinite_surface(
    x_grid_mm: np.ndarray,
    y_grid_mm: np.ndarray,
    source_world: np.ndarray,
    tissue_params: dict,
    z_surface: float = 10.0,
) -> np.ndarray:
    """Infinite medium Green's function evaluated on surface plane.

    G_inf(r) = exp(-mu_eff * r) / (4 * pi * D * r)

    where r is the 3D distance from source to surface point.

    Args:
        x_grid_mm: 2D array of x coordinates [H, W]
        y_grid_mm: 2D array of y coordinates [H, W]
        source_world: [x, y, z] source position in world coords
        tissue_params: optical properties
        z_surface: z-coordinate of surface plane

    Returns:
        surface_response: 2D array [H, W]
    """
    diff = diffusion_params(tissue_params)
    D = diff["D"]
    mu_eff = diff["mu_eff"]

    dx = x_grid_mm - source_world[0]
    dy = y_grid_mm - source_world[1]
    dz = z_surface - source_world[2]  # = depth from dorsal

    r = np.sqrt(dx**2 + dy**2 + dz**2)

    r = np.maximum(r, 1e-6)

    response = np.exp(-mu_eff * r) / (4 * np.pi * D * r)

    return response


def green_halfspace_surface(
    x_grid_mm: np.ndarray,
    y_grid_mm: np.ndarray,
    source_world: np.ndarray,
    tissue_params: dict,
    z_surface: float = 10.0,
    boundary_condition: str = "extrapolated",
) -> np.ndarray:
    """Half-space Green's function with boundary condition.

    Uses image source method to satisfy boundary condition at surface.

    G_halfspace = G(r1) - G(r2)

    where:
    - r1 = distance from surface point to real source
    - r2 = distance from surface point to image source

    For extrapolated boundary:
    - image source is at z = z_surface + zb + 2*depth = z_surface + 2*(zb + depth)
    - real source is at z = z_surface - depth

    For zero-boundary (simpler):
    - image source is at z = z_surface + depth (mirror across surface)

    Args:
        x_grid_mm: 2D array of x coordinates [H, W]
        y_grid_mm: 2D array of y coordinates [H, W]
        source_world: [x, y, z] source position in world coords
        tissue_params: optical properties
        z_surface: z-coordinate of surface plane
        boundary_condition: "extrapolated" or "zero"

    Returns:
        surface_response: 2D array [H, W]
    """
    diff = diffusion_params(tissue_params)
    D = diff["D"]
    mu_eff = diff["mu_eff"]
    zb = diff["zb"]

    depth = z_surface - source_world[2]

    dx = x_grid_mm - source_world[0]
    dy = y_grid_mm - source_world[1]

    r1 = np.sqrt(dx**2 + dy**2 + depth**2)

    if boundary_condition == "extrapolated":
        z_image = z_surface + depth + 2 * zb
        r2 = np.sqrt(dx**2 + dy**2 + (depth + 2 * zb) ** 2)
    elif boundary_condition == "zero":
        z_image = z_surface + depth
        r2 = np.sqrt(dx**2 + dy**2 + (2 * depth) ** 2)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")

    r1 = np.maximum(r1, 1e-6)
    r2 = np.maximum(r2, 1e-6)

    G1 = np.exp(-mu_eff * r1) / (4 * np.pi * D * r1)
    G2 = np.exp(-mu_eff * r2) / (4 * np.pi * D * r2)

    response = G1 - G2

    response = np.maximum(response, 0.0)

    return response


def get_kernel_function(kernel_name: str):
    """Get kernel function by name.

    Args:
        kernel_name: "gaussian_2d", "green_infinite", or "green_halfspace"

    Returns:
        kernel function
    """
    kernels = {
        "gaussian_2d": gaussian_2d_surface,
        "green_infinite": green_infinite_surface,
        "green_halfspace": green_halfspace_surface,
    }
    if kernel_name not in kernels:
        raise ValueError(
            f"Unknown kernel: {kernel_name}. Available: {list(kernels.keys())}"
        )
    return kernels[kernel_name]
