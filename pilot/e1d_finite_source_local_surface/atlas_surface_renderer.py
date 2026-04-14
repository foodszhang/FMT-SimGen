#!/usr/bin/env python3
"""Atlas-aware surface renderer for E1d-R2.

Renders response directly on atlas surface nodes instead of flat 2D plane.
Supports local_depth and local_plane geometry modes.
"""

from typing import Tuple, Optional
import numpy as np

from source_quadrature import sample_gaussian, sample_uniform
from surface_data import AtlasSurfaceData, compute_local_frame_distances
from kernels import diffusion_params


def render_atlas_surface_response(
    source_type: str,
    source_center: np.ndarray,
    source_params: dict,
    tissue_params: dict,
    surface_coords_mm: np.ndarray,
    surface_normals_mm: Optional[np.ndarray] = None,
    sampling_scheme: str = "sr-6",
    kernel_type: str = "green_halfspace",
    geometry_mode: str = "local_depth",
    source_alpha: float = 1.0,
) -> np.ndarray:
    """Render response on atlas surface nodes.

    Args:
        source_type: "point", "gaussian", or "uniform"
        source_center: [3] source center position in mm
        source_params: source-specific parameters (sigmas or axes)
        tissue_params: optical properties
        surface_coords_mm: [N, 3] surface node coordinates
        surface_normals_mm: [N, 3] surface normals (for local_plane mode)
        sampling_scheme: quadrature scheme name
        kernel_type: "green_infinite" or "green_halfspace"
        geometry_mode: "local_depth" or "local_plane"
        source_alpha: source intensity

    Returns:
        response: [N] surface response at each node
    """
    if source_type == "point":
        source_points = source_center.reshape(1, 3).astype(np.float32)
        source_weights = np.array([source_alpha], dtype=np.float32)

    elif source_type == "gaussian":
        sigmas = np.array(
            source_params.get("sigmas", [1.0, 1.0, 1.0]), dtype=np.float32
        )
        source_points, source_weights = sample_gaussian(
            center=source_center.astype(np.float32),
            sigmas=sigmas,
            alpha=source_alpha,
            scheme=sampling_scheme,
        )

    elif source_type == "uniform":
        axes = np.array(source_params.get("axes", [1.0, 1.0, 1.0]), dtype=np.float32)
        source_points, source_weights = sample_uniform(
            center=source_center.astype(np.float32),
            axes=axes,
            alpha=source_alpha,
            scheme=sampling_scheme,
        )

    else:
        raise ValueError(f"Unknown source type: {source_type}")

    diff = diffusion_params(tissue_params)
    D = diff["D"]
    mu_eff = diff["mu_eff"]
    zb = diff["zb"]

    depth, rho = compute_local_frame_distances(
        surface_coords_mm, source_points, surface_normals_mm, geometry_mode
    )

    n_surf, n_src = depth.shape
    response = np.zeros(n_surf, dtype=np.float32)

    for i in range(n_src):
        depth_i = depth[:, i]
        rho_i = rho[:, i]

        r1_sq = rho_i**2 + depth_i**2
        r1 = np.sqrt(np.maximum(r1_sq, 1e-12))

        if kernel_type == "green_infinite":
            G = np.exp(-mu_eff * r1) / (4 * np.pi * D * r1)

        elif kernel_type == "green_halfspace":
            r2_sq = rho_i**2 + (depth_i + 2 * zb) ** 2
            r2 = np.sqrt(np.maximum(r2_sq, 1e-12))

            G1 = np.exp(-mu_eff * r1) / (4 * np.pi * D * r1)
            G2 = np.exp(-mu_eff * r2) / (4 * np.pi * D * r2)

            G = np.maximum(G1 - G2, 0.0)

        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

        response += G * source_weights[i]

    return response


def render_atlas_surface_flat(
    source_type: str,
    source_center: np.ndarray,
    source_params: dict,
    tissue_params: dict,
    surface_coords_mm: np.ndarray,
    z_surface: float,
    sampling_scheme: str = "7-point",
    kernel_type: str = "green_halfspace",
    source_alpha: float = 1.0,
) -> np.ndarray:
    """Render response with flat plane assumption.

    Args:
        source_type: "point", "gaussian", or "uniform"
        source_center: [3] source center
        source_params: source parameters
        tissue_params: optical properties
        surface_coords_mm: [N, 3] surface coordinates
        z_surface: assumed flat surface Z
        sampling_scheme: quadrature scheme
        kernel_type: kernel type
        source_alpha: source intensity

    Returns:
        response: [N] surface response
    """
    surface_xy = surface_coords_mm[:, :2]
    surface_z_values = np.full(len(surface_coords_mm), z_surface, dtype=np.float32)

    from kernels import green_halfspace_finite_source, green_infinite_finite_source

    if source_type == "point":
        source_points = source_center.reshape(1, 3).astype(np.float32)
        source_weights = np.array([source_alpha], dtype=np.float32)

    elif source_type == "gaussian":
        sigmas = np.array(
            source_params.get("sigmas", [1.0, 1.0, 1.0]), dtype=np.float32
        )
        source_points, source_weights = sample_gaussian(
            center=source_center.astype(np.float32),
            sigmas=sigmas,
            alpha=source_alpha,
            scheme=sampling_scheme,
        )

    elif source_type == "uniform":
        axes = np.array(source_params.get("axes", [1.0, 1.0, 1.0]), dtype=np.float32)
        source_points, source_weights = sample_uniform(
            center=source_center.astype(np.float32),
            axes=axes,
            alpha=source_alpha,
            scheme=sampling_scheme,
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    if kernel_type == "green_halfspace":
        response = green_halfspace_finite_source(
            surface_points_mm=surface_xy,
            source_points_mm=source_points,
            source_weights=source_weights,
            tissue_params=tissue_params,
            z_surface=z_surface,
            surface_z_values=None,
        )
    elif kernel_type == "green_infinite":
        response = green_infinite_finite_source(
            surface_points_mm=surface_xy,
            source_points_mm=source_points,
            source_weights=source_weights,
            tissue_params=tissue_params,
            z_surface=z_surface,
            surface_z_values=None,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")

    return response


def render_atlas_surface_local_depth(
    source_type: str,
    source_center: np.ndarray,
    source_params: dict,
    tissue_params: dict,
    surface_coords_mm: np.ndarray,
    sampling_scheme: str = "sr-6",
    kernel_type: str = "green_halfspace",
    source_alpha: float = 1.0,
) -> np.ndarray:
    """Render response with local depth approximation.

    Args:
        source_type: source type
        source_center: [3] source center
        source_params: source parameters
        tissue_params: optical properties
        surface_coords_mm: [N, 3] surface coordinates (with actual Z values)
        sampling_scheme: quadrature scheme
        kernel_type: kernel type
        source_alpha: source intensity

    Returns:
        response: [N] surface response
    """
    return render_atlas_surface_response(
        source_type=source_type,
        source_center=source_center,
        source_params=source_params,
        tissue_params=tissue_params,
        surface_coords_mm=surface_coords_mm,
        surface_normals_mm=None,
        sampling_scheme=sampling_scheme,
        kernel_type=kernel_type,
        geometry_mode="local_depth",
        source_alpha=source_alpha,
    )


def render_atlas_surface_local_plane(
    source_type: str,
    source_center: np.ndarray,
    source_params: dict,
    tissue_params: dict,
    surface_coords_mm: np.ndarray,
    surface_normals_mm: np.ndarray,
    sampling_scheme: str = "sr-6",
    kernel_type: str = "green_halfspace",
    source_alpha: float = 1.0,
) -> np.ndarray:
    """Render response with local plane approximation.

    Uses surface normals for more accurate local frame.

    Args:
        source_type: source type
        source_center: [3] source center
        source_params: source parameters
        tissue_params: optical properties
        surface_coords_mm: [N, 3] surface coordinates
        surface_normals_mm: [N, 3] surface normals
        sampling_scheme: quadrature scheme
        kernel_type: kernel type
        source_alpha: source intensity

    Returns:
        response: [N] surface response
    """
    return render_atlas_surface_response(
        source_type=source_type,
        source_center=source_center,
        source_params=source_params,
        tissue_params=tissue_params,
        surface_coords_mm=surface_coords_mm,
        surface_normals_mm=surface_normals_mm,
        sampling_scheme=sampling_scheme,
        kernel_type=kernel_type,
        geometry_mode="local_plane",
        source_alpha=source_alpha,
    )
