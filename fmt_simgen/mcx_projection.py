"""
MCX projection module for FMT-SimGen.

Projects MCX 3D fluence volumes (.jnii) to multi-angle 2D detector
projections (proj.npz) using the reference's proven projection algorithm.

Coordinate system:
- MCX volume is stored in ZYX order (shape: Z×Y×X = 104×200×190)
- JNII → XYZ: transpose(2, 1, 0) → shape (X=190, Y=200, Z=104)
- Reference projection places volume center at world origin, camera at [0,0,D]
  looking toward origin along -Z, with rotation around Y axis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import jdata as jd
import numpy as np

from fmt_simgen.view_config import TurntableCamera

logger = logging.getLogger(__name__)


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """Build Y-axis rotation matrix (row-major, column-vector convention).

    Rotates points by angle_deg degrees around the Y axis.
    For angle > 0: positive rotation (counterclockwise when viewed from +Y).
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a],
    ], dtype=np.float32)


def project_volume_reference(
    volume_3d: np.ndarray,
    angle_deg: float,
    camera_distance: float,
    fov_mm: float,
    detector_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reference orthographic projection (adapted from gen_mul_projection.py).

    Places volume center at origin, camera at [0,0,D] looking toward origin.
    All rays are parallel to -Z. Keeps nearest (shallowest) non-zero voxel.

    Parameters
    ----------
    volume_3d : np.ndarray
        3D fluence volume [X×Y×Z].
    angle_deg : float
        Rotation angle in degrees.
    camera_distance : float
        Camera z-position in mm.
    fov_mm : float
        Field of view in mm (square detector).
    detector_resolution : tuple
        (width, height) in pixels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (projection [H×W], depth_map [H×W])
    """
    width_pixels, height_pixels = detector_resolution
    width_phys, height_phys = fov_mm, fov_mm

    projection = np.zeros((height_pixels, width_pixels), dtype=np.float32)
    depth_map = np.full((height_pixels, width_pixels), np.inf, dtype=np.float32)

    pixel_to_phys_x = width_phys / width_pixels
    pixel_to_phys_y = height_phys / height_pixels

    nx, ny, nz = volume_3d.shape

    # Collect non-zero voxels with their world coordinates
    nonzero_mask = volume_3d > 0
    nonzero_indices = np.argwhere(nonzero_mask)  # [N, 3] in (ix, iy, iz)
    if len(nonzero_indices) == 0:
        return projection, depth_map

    # Voxel center in world coordinates, centered at origin
    # reference: x = i - nx/2 + 0.5
    points = np.zeros((len(nonzero_indices), 3), dtype=np.float32)
    for idx, (i, j, k) in enumerate(nonzero_indices):
        points[idx, 0] = i - nx / 2 + 0.5
        points[idx, 1] = j - ny / 2 + 0.5
        points[idx, 2] = k - nz / 2 + 0.5

    values = volume_3d[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

    # Rotate volume around Y axis by angle_deg
    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        # points = R @ points.T .T
        points = points @ R.T

    # Project: for orthographic, all rays parallel to -Z
    # Camera at (0, 0, camera_distance), looking toward origin
    # Transform points to camera frame: subtract camera pos, rotate by R^-1 = R^T
    # Since R is orthogonal, R^-1 = R^T
    # camera_coords = R @ (point - [0,0,D]) = R @ point + [0,0,-D]
    # For orthographic projection, we only care about the rotated (x, z) in camera frame
    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        cam_x = points[:, 0]
        cam_y = points[:, 1]  # world Y = detector vertical
        cam_z = points[:, 2]  # world Z → depth
    else:
        cam_x = points[:, 0]
        cam_y = points[:, 1]
        cam_z = points[:, 2]

    # depth = camera_distance - cam_z (positive = in front of detector)
    # camera at (0, 0, camera_distance), looking toward -Z
    # point with cam_z < D is in front of camera (along -Z direction)
    depths = camera_distance - cam_z

    for idx in range(len(points)):
        px = cam_x[idx]
        py = cam_y[idx]
        depth_val = depths[idx]

        if abs(px) > width_phys / 2 or abs(py) > height_phys / 2:
            continue
        if depth_val < 0:
            continue

        pixel_u = int((px + width_phys / 2) / pixel_to_phys_x)
        pixel_v = int((py + height_phys / 2) / pixel_to_phys_y)

        if 0 <= pixel_u < width_pixels and 0 <= pixel_v < height_pixels:
            if depth_val < depth_map[pixel_v, pixel_u]:
                depth_map[pixel_v, pixel_u] = depth_val
                projection[pixel_v, pixel_u] = values[idx]

    return projection, depth_map


def load_jnii_volume(jnii_path: Path) -> np.ndarray:
    """Load a .jnii MCX fluence file and return volume in XYZ order.

    Parameters
    ----------
    jnii_path : Path
        Path to .jnii file.

    Returns
    -------
    np.ndarray
        Fluence volume in XYZ order [X×Y×Z], float32.
        Shape: (190, 200, 104) for standard FMT-SimGen MCX volume.
    """
    data = jd.loadjd(str(jnii_path))
    nifti = data["NIFTIData"] if isinstance(data, dict) else data
    if nifti.ndim == 5:
        nifti = nifti[:, :, :, 0, 0]
    # ZYX → XYZ: transpose(2, 1, 0)
    return nifti.transpose(2, 1, 0).astype(np.float32)


def project_mcx_fluence(
    fluence_xyz: np.ndarray,
    camera: TurntableCamera,
) -> Dict[str, np.ndarray]:
    """Project MCX fluence volume to multi-angle 2D projections.

    Uses the reference's proven projection algorithm adapted from
    gen_mul_projection.py. Places volume center at world origin,
    camera at [0, 0, D] looking toward origin, with rotation around Y.

    Parameters
    ----------
    fluence_xyz : np.ndarray
        3D fluence volume in XYZ order [X×Y×Z].
    camera : TurntableCamera
        Camera model with configured angles and detector parameters.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from angle string to projection image [H×W].
        Keys: "-90", "-60", "-30", "0", "30", "60", "90"
    """
    if fluence_xyz.sum() == 0:
        logger.warning("Fluence volume is all-zero")

    results: Dict[str, np.ndarray] = {}
    for angle in camera.angles:
        proj, _ = project_volume_reference(
            fluence_xyz,
            angle,
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
        )
        results[str(angle)] = proj.astype(np.float32)

    return results


def project_sample(
    sample_dir: Path,
    camera: TurntableCamera,
    skip_existing: bool = True,
) -> Path:
    """Project a single sample's MCX fluence to proj.npz.

    Parameters
    ----------
    sample_dir : Path
        Sample directory containing {id}.jnii files.
    camera : TurntableCamera
        Camera model for projection.
    skip_existing : bool
        If True (default), skip samples with existing proj.npz.

    Returns
    -------
    Path
        Path to the generated proj.npz file.

    Raises
    ------
    FileNotFoundError
        If no .jnii file exists in the sample directory.
    """
    jnii_files = list(sample_dir.glob("*.jnii"))
    if not jnii_files:
        raise FileNotFoundError(f"No .jnii found in {sample_dir}")

    jnii_path = jnii_files[0]
    proj_path = sample_dir / "proj.npz"

    if skip_existing and proj_path.exists():
        logger.debug("Skipping %s: proj.npz exists", sample_dir.name)
        return proj_path

    fluence = load_jnii_volume(jnii_path)
    projections = project_mcx_fluence(fluence, camera)

    for angle_str, proj in projections.items():
        if proj.sum() == 0:
            logger.warning(
                "Sample %s angle %s: all-zero projection",
                sample_dir.name,
                angle_str,
            )

    np.savez(proj_path, **projections)
    logger.debug("Saved projections: %s", proj_path)

    return proj_path
