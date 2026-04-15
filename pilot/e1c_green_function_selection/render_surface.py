#!/usr/bin/env python3
"""Surface rendering utilities for E1c.

Provides unified grid construction and surface image rendering
for all kernel types.
"""

import numpy as np
from typing import Tuple, Callable
from kernels import get_kernel_function


def build_surface_grid(
    image_size: int,
    pixel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 2D coordinate grid on surface plane.

    The grid is centered at origin (0, 0) in world coordinates.

    Args:
        image_size: number of pixels along each dimension
        pixel_size_mm: pixel size in mm

    Returns:
        x_grid_mm: 2D array [H, W] of x coordinates
        y_grid_mm: 2D array [H, W] of y coordinates
    """
    fov_mm = image_size * pixel_size_mm
    coords_mm = (np.arange(image_size) - image_size / 2 + 0.5) * pixel_size_mm

    x_grid_mm, y_grid_mm = np.meshgrid(coords_mm, coords_mm)

    return x_grid_mm, y_grid_mm


def render_surface_image(
    kernel_name: str,
    source_world: np.ndarray,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    calib_params: dict | None = None,
    z_surface: float = 10.0,
) -> np.ndarray:
    """Render surface response image using specified kernel.

    Args:
        kernel_name: "gaussian_2d", "green_infinite", or "green_halfspace"
        source_world: [x, y, z] source position in world coordinates
        tissue_params: dict with mua_mm, mus_mm, g, n
        image_size: image dimension in pixels
        pixel_size_mm: pixel size in mm
        calib_params: calibration parameters for Gaussian kernel
        z_surface: z-coordinate of surface plane

    Returns:
        surface_image: 2D array [H, W]
    """
    x_grid_mm, y_grid_mm = build_surface_grid(image_size, pixel_size_mm)

    kernel_func = get_kernel_function(kernel_name)

    if kernel_name == "gaussian_2d":
        image = kernel_func(
            x_grid_mm, y_grid_mm, source_world, tissue_params, calib_params
        )
    elif kernel_name == "green_infinite":
        image = kernel_func(
            x_grid_mm, y_grid_mm, source_world, tissue_params, z_surface
        )
    elif kernel_name == "green_halfspace":
        image = kernel_func(
            x_grid_mm, y_grid_mm, source_world, tissue_params, z_surface
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    return image


def render_all_kernels(
    source_world: np.ndarray,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    calib_params: dict | None = None,
    z_surface: float = 10.0,
) -> dict:
    """Render surface images using all three kernels.

    Args:
        source_world: [x, y, z] source position
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        calib_params: Gaussian calibration
        z_surface: surface z coordinate

    Returns:
        dict with keys "gaussian_2d", "green_infinite", "green_halfspace"
    """
    kernels = ["gaussian_2d", "green_infinite", "green_halfspace"]
    results = {}

    for kernel_name in kernels:
        results[kernel_name] = render_surface_image(
            kernel_name=kernel_name,
            source_world=source_world,
            tissue_params=tissue_params,
            image_size=image_size,
            pixel_size_mm=pixel_size_mm,
            calib_params=calib_params,
            z_surface=z_surface,
        )

    return results
