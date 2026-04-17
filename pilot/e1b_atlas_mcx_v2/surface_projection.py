"""Surface-aware Green's function projection.

Directly computes Green's function response on the visible surface,
without generating an intermediate 3D volume.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ]
    )


@njit(cache=True, fastmath=True)
def _project_voxels_to_surface_numba(
    cam_x: np.ndarray,
    cam_y: np.ndarray,
    depths: np.ndarray,
    original_coords: np.ndarray,
    width: int,
    height: int,
    half_w: float,
    half_h: float,
    px_size_x: float,
    px_size_y: float,
    half_voxel: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated voxel-to-surface projection.

    For each voxel, fills all detector pixels it covers, keeping the shallowest.

    Returns
    -------
    surface_coords : np.ndarray
        (H, W, 3) array of surface voxel coordinates in mm.
    depth_map : np.ndarray
        (H, W) depth map.
    """
    surface_coords = np.zeros((height, width, 3), dtype=np.float32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    n_voxels = len(cam_x)

    for idx in range(n_voxels):
        px = cam_x[idx]
        py = cam_y[idx]
        d = depths[idx]

        # Check if within FOV and in front of camera
        if abs(px) > half_w or abs(py) > half_h or d < 0:
            continue

        # Calculate pixel range covered by this voxel's physical extent
        u_start = int((px - half_voxel + half_w) / px_size_x)
        u_end = int((px + half_voxel + half_w) / px_size_x)
        v_start = int((py - half_voxel + half_h) / px_size_y)
        v_end = int((py + half_voxel + half_h) / px_size_y)

        # Clamp to valid range
        u_start = max(0, u_start)
        u_end = min(width - 1, u_end)
        v_start = max(0, v_start)
        v_end = min(height - 1, v_end)

        # Fill all pixels covered by this voxel
        for pu in range(u_start, u_end + 1):
            for pv in range(v_start, v_end + 1):
                if d < depth_map[pv, pu]:
                    depth_map[pv, pu] = d
                    surface_coords[pv, pu, 0] = original_coords[idx, 0]
                    surface_coords[pv, pu, 1] = original_coords[idx, 1]
                    surface_coords[pv, pu, 2] = original_coords[idx, 2]

    return surface_coords, depth_map


def project_get_surface_coords(
    volume_mask: np.ndarray,
    angle_deg: float,
    camera_distance_mm: float,
    fov_mm: float,
    detector_resolution: Tuple[int, int],
    voxel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project binary volume and return surface voxel 3D coordinates per pixel.

    For each detector pixel, finds the nearest (shallowest) non-zero voxel
    along the viewing direction and returns its ORIGINAL (pre-rotation) 3D coordinate.

    Parameters
    ----------
    volume_mask : np.ndarray
        Binary mask (X, Y, Z) where >0 indicates tissue.
    angle_deg : float
        Rotation angle around Y axis in degrees.
    camera_distance_mm : float
        Distance from camera to rotation center.
    fov_mm : float
        Field of view in mm.
    detector_resolution : tuple
        (width, height) in pixels.
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    surface_coords : np.ndarray
        (H, W, 3) array of surface voxel coordinates in mm (original frame).
        Invalid pixels (no surface) are [0, 0, 0].
    valid_mask : np.ndarray
        (H, W) boolean mask indicating valid surface pixels.
    """
    width, height = detector_resolution
    nx, ny, nz = volume_mask.shape

    # Get non-zero voxel indices
    nonzero = np.argwhere(volume_mask > 0)  # (N, 3) in [ix, iy, iz]

    if len(nonzero) == 0:
        logger.warning("Empty volume mask")
        return np.zeros((height, width, 3), dtype=np.float32), np.zeros(
            (height, width), dtype=bool
        )

    # Original 3D coordinates in mm (centered at volume center)
    # volume_mask is (X, Y, Z) = (nx, ny, nz)
    center = np.array([nx / 2, ny / 2, nz / 2])
    original_coords = (nonzero.astype(np.float32) - center + 0.5) * voxel_size_mm

    # Apply rotation for projection
    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        rotated_coords = original_coords @ R.T
    else:
        rotated_coords = original_coords.copy()

    # Camera coordinates (looking along -Z direction)
    # rotated_coords: [x, y, z] where +z is towards camera
    cam_x = rotated_coords[:, 0]
    cam_y = rotated_coords[:, 1]
    # Depth from camera (positive = in front of camera)
    depths = camera_distance_mm - rotated_coords[:, 2]

    # Projection bounds
    half_w = fov_mm / 2
    half_h = fov_mm / 2
    px_size_x = fov_mm / width
    px_size_y = fov_mm / height

    # Use numba-accelerated projection
    half_voxel = voxel_size_mm / 2

    surface_coords, depth_map = _project_voxels_to_surface_numba(
        cam_x.astype(np.float32),
        cam_y.astype(np.float32),
        depths.astype(np.float32),
        original_coords.astype(np.float32),
        width,
        height,
        half_w,
        half_h,
        px_size_x,
        px_size_y,
        half_voxel,
    )

    valid_mask = depth_map < np.inf
    n_valid = np.sum(valid_mask)
    logger.debug(
        "Angle %s°: %d/%d valid surface pixels", angle_deg, n_valid, width * height
    )

    return surface_coords, valid_mask


def compute_diffusion_params(tissue_params: dict) -> dict:
    """Compute diffusion parameters from tissue properties."""
    mua = tissue_params["mua_mm"]
    mus_prime = tissue_params["mus_prime_mm"]

    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(3.0 * mua * (mua + mus_prime))

    return {"D": D, "mu_eff": mu_eff}


def green_infinite_point_source_on_surface(
    source_pos_mm: np.ndarray,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    tissue_params: dict,
) -> np.ndarray:
    """Compute Green's function response on surface points.

    Uses infinite medium Green's function (Eq. 7 from the paper).
    G(r) = exp(-mu_eff * r) / (4 * pi * D * r)

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position [3] in mm.
    surface_coords : np.ndarray
        (H, W, 3) surface voxel coordinates in mm.
    valid_mask : np.ndarray
        (H, W) boolean mask for valid pixels.
    tissue_params : dict
        {mua_mm, mus_prime_mm} tissue parameters.

    Returns
    -------
    projection : np.ndarray
        (H, W) Green's function response on surface.
    """
    diff_params = compute_diffusion_params(tissue_params)
    D = diff_params["D"]
    mu_eff = diff_params["mu_eff"]

    # Compute distance from source to each surface point
    dx = surface_coords[:, :, 0] - source_pos_mm[0]
    dy = surface_coords[:, :, 1] - source_pos_mm[1]
    dz = surface_coords[:, :, 2] - source_pos_mm[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)  # Avoid division by zero

    # Green's function (infinite medium)
    G = np.exp(-mu_eff * r) / (4.0 * np.pi * D * r)

    # Only valid pixels
    projection = np.zeros_like(G, dtype=np.float32)
    projection[valid_mask] = G[valid_mask]

    return projection


def green_semi_infinite_point_source_on_surface(
    source_pos_mm: np.ndarray,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    tissue_params: dict,
    surface_z_mm: float = 0.0,
) -> np.ndarray:
    """Compute semi-infinite Green's function response on surface points.

    Uses extrapolated boundary condition with image source (Eq. 14).
    Accounts for tissue-air interface.

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position [3] in mm.
    surface_coords : np.ndarray
        (H, W, 3) surface voxel coordinates in mm.
    valid_mask : np.ndarray
        (H, W) boolean mask for valid pixels.
    tissue_params : dict
        {mua_mm, mus_prime_mm, g, n} tissue parameters.
    surface_z_mm : float
        Z coordinate of the surface plane.

    Returns
    -------
    projection : np.ndarray
        (H, W) Green's function response on surface.
    """
    mua = tissue_params["mua_mm"]
    mus_prime = tissue_params["mus_prime_mm"]
    n = tissue_params.get("n", 1.37)

    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = np.sqrt(mua / D)

    # Extrapolated boundary parameters
    R_eff = 0.493  # for n=1.37 tissue/air interface
    A = (1 + R_eff) / (1 - R_eff)
    z_b = 2 * A * D

    # Source depth from surface
    d = source_pos_mm[2] - surface_z_mm

    # Distance to real source
    dx = surface_coords[:, :, 0] - source_pos_mm[0]
    dy = surface_coords[:, :, 1] - source_pos_mm[1]
    dz_real = surface_coords[:, :, 2] - source_pos_mm[2]
    r1 = np.sqrt(dx**2 + dy**2 + dz_real**2)

    # Distance to image source (reflected across extrapolated boundary)
    # Image source at z = source_z + 2*(z_b + d) = source_z + 2*z_b + 2*d
    # But surface is at z = surface_z_mm, so image source is at:
    # z_image = source_pos_mm[2] + 2 * z_b
    dz_image = surface_coords[:, :, 2] - (source_pos_mm[2] + 2 * z_b)
    r2 = np.sqrt(dx**2 + dy**2 + dz_image**2)

    r1 = np.maximum(r1, 0.01)
    r2 = np.maximum(r2, 0.01)

    # Semi-infinite Green's function
    G = (np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * r2) / r2) / (4 * np.pi * D)
    G = np.maximum(G, 0)

    projection = np.zeros_like(G, dtype=np.float32)
    projection[valid_mask] = G[valid_mask]

    return projection


def render_green_surface_projection(
    source_pos_mm: np.ndarray,
    atlas_volume_binary: np.ndarray,
    angle_deg: float,
    camera_distance_mm: float,
    fov_mm: float,
    detector_resolution: Tuple[int, int],
    tissue_params: dict,
    voxel_size_mm: float = 0.2,
    green_type: str = "infinite",
    surface_z_mm: float = 0.0,
) -> np.ndarray:
    """Render Green's function projection for a point source.

    Main entry point for surface-aware Green projection.

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position [3] in mm.
    atlas_volume_binary : np.ndarray
        Binary atlas mask (X, Y, Z).
    angle_deg : float
        Viewing angle.
    camera_distance_mm : float
        Camera distance.
    fov_mm : float
        Field of view.
    detector_resolution : tuple
        (width, height) in pixels.
    tissue_params : dict
        Tissue optical parameters.
    voxel_size_mm : float
        Voxel size in mm.
    green_type : str
        "infinite" or "semi_infinite".
    surface_z_mm : float
        Surface Z coordinate (for semi-infinite).

    Returns
    -------
    projection : np.ndarray
        (H, W) 2D projection image.
    """
    # Step 1: Get surface coordinates for this angle
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_volume_binary,
        angle_deg,
        camera_distance_mm,
        fov_mm,
        detector_resolution,
        voxel_size_mm,
    )

    # Step 2: Compute Green's function on surface
    if green_type == "infinite":
        projection = green_infinite_point_source_on_surface(
            source_pos_mm, surface_coords, valid_mask, tissue_params
        )
    elif green_type == "semi_infinite":
        projection = green_semi_infinite_point_source_on_surface(
            source_pos_mm, surface_coords, valid_mask, tissue_params, surface_z_mm
        )
    else:
        raise ValueError(f"Unknown green_type: {green_type}")

    return projection


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation."""
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
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))
