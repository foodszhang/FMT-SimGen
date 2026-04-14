#!/usr/bin/env python3
"""Local surface renderer for E1d.

Implements efficient ROI-based rendering for finite-size sources.
"""

import numpy as np
from typing import Tuple, Optional
from source_models import BaseSource
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


def estimate_roi_from_source(
    source_center_mm: np.ndarray,
    source_extent_mm: float,
    tissue_params: dict,
    pixel_size_mm: float,
    image_size: int,
    kernel_type: str = "green_halfspace",
    cutoff_ratio: float = 1e-3,
    safety_margin_mm: float = 5.0,
) -> Tuple[int, int, int, int]:
    """Estimate ROI bounds based on source and kernel properties.

    Uses the fact that Green's function response decays exponentially.

    Args:
        source_center_mm: [x, y, z] source center position
        source_extent_mm: approximate source extent in mm
        tissue_params: optical properties
        pixel_size_mm: pixel size in mm
        image_size: full image size
        kernel_type: kernel type for estimating spread
        cutoff_ratio: cutoff threshold relative to peak
        safety_margin_mm: additional margin around estimated ROI

    Returns:
        (x0, y0, x1, y1): ROI bounds in pixel coordinates
    """
    import sys

    sys.path.insert(0, "/home/foods/pro/FMT-SimGen/pilot/e1c_green_function_selection")
    from kernels import diffusion_params

    diff = diffusion_params(tissue_params)
    mu_eff = diff["mu_eff"]

    depth = abs(10.0 - source_center_mm[2])

    typical_distance = max(depth, source_extent_mm)

    if cutoff_ratio > 0:
        max_radius_mm = -np.log(cutoff_ratio) / mu_eff + typical_distance
    else:
        max_radius_mm = image_size * pixel_size_mm / 2

    max_radius_mm += safety_margin_mm

    center_x_px = int(image_size / 2 + source_center_mm[0] / pixel_size_mm)
    center_y_px = int(image_size / 2 + source_center_mm[1] / pixel_size_mm)

    radius_px = int(np.ceil(max_radius_mm / pixel_size_mm))

    x0 = max(0, center_x_px - radius_px)
    y0 = max(0, center_y_px - radius_px)
    x1 = min(image_size, center_x_px + radius_px)
    y1 = min(image_size, center_y_px + radius_px)

    return x0, y0, x1, y1


def render_local_surface_response(
    source: BaseSource,
    kernel_type: str,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    sampling_level: str = "7-point",
    cutoff_ratio: float = 1e-3,
    z_surface: float = 10.0,
    use_roi: bool = True,
) -> np.ndarray:
    """Render surface response with local ROI optimization.

    Args:
        source: source model instance
        kernel_type: "gaussian_psf", "green_infinite", or "green_halfspace"
        tissue_params: optical properties
        image_size: image dimension in pixels
        pixel_size_mm: pixel size in mm
        sampling_level: source sampling level
        cutoff_ratio: ROI cutoff threshold
        z_surface: z-coordinate of surface plane
        use_roi: whether to use ROI optimization

    Returns:
        surface_image: 2D array [H, W]
    """
    source_points, source_weights = source.sample_points(sampling_level)

    x_grid_mm, y_grid_mm = build_surface_grid(image_size, pixel_size_mm)

    kernel_func = get_kernel_function(kernel_type)

    if use_roi:
        roi = estimate_roi_from_source(
            source_center_mm=source.center,
            source_extent_mm=source.get_extent_mm(),
            tissue_params=tissue_params,
            pixel_size_mm=pixel_size_mm,
            image_size=image_size,
            kernel_type=kernel_type,
            cutoff_ratio=cutoff_ratio,
        )
        x0, y0, x1, y1 = roi
    else:
        x0, y0, x1, y1 = 0, 0, image_size, image_size

    surface_image = np.zeros((image_size, image_size), dtype=np.float32)

    if x1 > x0 and y1 > y0:
        roi_x_grid = x_grid_mm[y0:y1, x0:x1]
        roi_y_grid = y_grid_mm[y0:y1, x0:x1]
        roi_surface_points = np.stack([roi_x_grid, roi_y_grid], axis=-1)

        if kernel_type == "gaussian_psf":
            roi_response = kernel_func(
                roi_surface_points,
                source_points,
                source_weights,
                tissue_params,
            )
        elif kernel_type == "green_infinite":
            roi_response = kernel_func(
                roi_surface_points,
                source_points,
                source_weights,
                tissue_params,
                z_surface,
            )
        elif kernel_type == "green_halfspace":
            roi_response = kernel_func(
                roi_surface_points,
                source_points,
                source_weights,
                tissue_params,
                z_surface,
            )
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        surface_image[y0:y1, x0:x1] = roi_response.reshape(roi_x_grid.shape)

    return surface_image


def render_full_surface_response(
    source: BaseSource,
    kernel_type: str,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    sampling_level: str = "7-point",
    z_surface: float = 10.0,
) -> np.ndarray:
    """Render surface response without ROI optimization.

    Args:
        source: source model instance
        kernel_type: kernel type
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        sampling_level: source sampling level
        z_surface: z-coordinate of surface plane

    Returns:
        surface_image: 2D array [H, W]
    """
    return render_local_surface_response(
        source=source,
        kernel_type=kernel_type,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level=sampling_level,
        z_surface=z_surface,
        use_roi=False,
    )
