"""
MCX source pattern generation for pattern3d format.

Converts FMT-SimGen tumor_params.json (analytic Gaussian tumors in physical mm)
into MCX pattern3d source files (float32 binary + origin for JSON Pos field).
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def tumor_params_to_mcx_pattern(
    tumor_params: Dict,
    mcx_config: Dict,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Convert tumor_params.json to MCX pattern3d binary and origin.

    Parameters
    ----------
    tumor_params : Dict
        Tumor parameters dict with keys:
        - num_foci: int
        - source_type: "gaussian" or "uniform"
        - foci: List[Dict] with keys:
            - center: [px, py, pz] in physical mm [X, Y, Z]
            - shape: "sphere" or "ellipsoid"
            - radius: float (mm) for sphere
            - rx, ry, rz: float (mm) for ellipsoid
    mcx_config : Dict
        MCX configuration section from default.yaml with keys:
        - trunk_offset_mm: [ox, oy, oz] physical mm offset
        - voxel_size_mm: float (e.g., 0.2)
        - volume_shape: [nz, ny, nx] (ZYX order)

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, int]]
        - pattern: float32 3D array (nx, ny, nz) in XYZ order
        - origin: (x0, y0, z0) in volume voxel coordinates (XYZ order for JSON Pos)

    Raises
    ------
    ValueError
        If no foci produce non-zero voxels or focus is outside volume bounds.
    """
    trunk_offset_mm = np.array(mcx_config["trunk_offset_mm"])  # [ox, oy, oz]
    voxel_size_mm = mcx_config["voxel_size_mm"]
    volume_shape = tuple(mcx_config["volume_shape"])  # (nz, ny, nx)

    foci = tumor_params.get("foci", [])
    if not foci:
        raise ValueError("tumor_params has no foci")

    # Compute global pattern bbox from all foci
    global_min_voxel = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max_voxel = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for focus in foci:
        focus_min, focus_max = _focus_bbox_voxels(focus, trunk_offset_mm, voxel_size_mm)
        global_min_voxel = np.minimum(global_min_voxel, focus_min)
        global_max_voxel = np.maximum(global_max_voxel, focus_max)

    # Apply 2-voxel padding
    pad = 2
    global_min_voxel = np.floor(global_min_voxel).astype(int) - pad
    global_max_voxel = np.ceil(global_max_voxel).astype(int) + pad

    # Clamp to volume bounds
    nx, ny, nz = volume_shape[2], volume_shape[1], volume_shape[0]
    x0 = max(0, int(global_min_voxel[0]))
    y0 = max(0, int(global_min_voxel[1]))
    z0 = max(0, int(global_min_voxel[2]))
    x1 = min(nx, int(global_max_voxel[0]))
    y1 = min(ny, int(global_max_voxel[1]))
    z1 = min(nz, int(global_max_voxel[2]))

    # Validate non-empty bbox
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        raise ValueError(
            f"Pattern bbox is empty after clamping: "
            f"(x0={x0}, x1={x1}, y0={y0}, y1={y1}, z0={z0}, z1={z1}). "
            f"Tumor focus may be outside MCX volume bounds."
        )

    pattern_shape = (x1 - x0, y1 - y0, z1 - z0)
    logger.debug(
        f"Pattern bbox: ({x0},{y0},{z0}) to ({x1},{y1},{z1}), "
        f"shape {pattern_shape}"
    )

    # Evaluate Gaussian at voxel centers
    pattern = _evaluate_pattern_voxels(
        (x0, y0, z0), (x1, y1, z1), foci, trunk_offset_mm, voxel_size_mm
    )

    # Apply 1% threshold
    pattern_max = pattern.max()
    if pattern_max > 0:
        threshold = 0.01 * pattern_max
        pattern = np.where(pattern < threshold, 0.0, pattern)

    nonzero_count = np.count_nonzero(pattern)
    if nonzero_count == 0:
        raise ValueError(
            f"Pattern is all zeros after 1% threshold. "
            f"Check tumor center positions and sigma values."
        )

    logger.debug(f"Pattern nonzero voxels: {nonzero_count}/{pattern.size}")

    return pattern.astype(np.float32), (x0, y0, z0)


def _focus_bbox_voxels(
    focus: Dict,
    trunk_offset_mm: np.ndarray,
    voxel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3-sigma bounding box for a focus in voxel coordinates.

    Parameters
    ----------
    focus : Dict
        Focus dict with center, shape, radius/rx/ry/rz
    trunk_offset_mm : np.ndarray
        [ox, oy, oz] physical offset
    voxel_size_mm : float
        Voxel size in mm

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (min_voxel, max_voxel) in voxel coordinates [ix, iy, iz]
    """
    center = np.array(focus["center"], dtype=np.float64)  # [px, py, pz]

    if focus["shape"] == "sphere":
        sigma_mm = float(focus["radius"])
    elif focus["shape"] == "ellipsoid":
        sigma_mm = max(float(focus["rx"]), float(focus["ry"]), float(focus["rz"]))
    else:
        raise ValueError(f"Unknown shape type: {focus['shape']}")

    cutoff_mm = 3.0 * sigma_mm

    # Bbox in physical mm
    bbox_min_mm = center - cutoff_mm
    bbox_max_mm = center + cutoff_mm

    # Convert to voxel coordinates (floor for min, ceil for max)
    voxel_min = (bbox_min_mm - trunk_offset_mm) / voxel_size_mm
    voxel_max = (bbox_max_mm - trunk_offset_mm) / voxel_size_mm

    return voxel_min, voxel_max


def _evaluate_pattern_voxels(
    bbox_min: Tuple[int, int, int],
    bbox_max: Tuple[int, int, int],
    foci: List[Dict],
    trunk_offset_mm: np.ndarray,
    voxel_size_mm: float,
) -> np.ndarray:
    """Evaluate multi-focal Gaussian at pattern voxel centers.

    Parameters
    ----------
    bbox_min : Tuple[int, int, int]
        (x0, y0, z0) in volume voxel coords
    bbox_max : Tuple[int, int, int]
        (x1, y1, z1) in volume voxel coords (exclusive)
    foci : List[Dict]
        Focus definitions
    trunk_offset_mm : np.ndarray
        Physical offset [ox, oy, oz]
    voxel_size_mm : float
        Voxel size mm

    Returns
    -------
    np.ndarray
        float32 pattern (nx, ny, nz) in XYZ order
    """
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    nx, ny, nz = x1 - x0, y1 - y0, z1 - z0

    # Generate voxel center coordinates in physical mm
    # voxel index i -> physical coord = (x0 + i) * voxel_size_mm + trunk_offset
    gx = np.arange(nx) * voxel_size_mm + (x0 * voxel_size_mm + trunk_offset_mm[0]) + voxel_size_mm / 2
    gy = np.arange(ny) * voxel_size_mm + (y0 * voxel_size_mm + trunk_offset_mm[1]) + voxel_size_mm / 2
    gz = np.arange(nz) * voxel_size_mm + (z0 * voxel_size_mm + trunk_offset_mm[2]) + voxel_size_mm / 2

    gx3d, gy3d, gz3d = np.meshgrid(gx, gy, gz, indexing="ij")
    coords = np.column_stack([gx3d.ravel(), gy3d.ravel(), gz3d.ravel()])

    # Evaluate each focus and take max
    values = np.zeros(len(coords), dtype=np.float32)
    for focus in foci:
        focus_values = _evaluate_single_focus_gaussian(coords, focus)
        values = np.maximum(values, focus_values)

    return values.reshape(nx, ny, nz)


def _evaluate_single_focus_gaussian(coords: np.ndarray, focus: Dict) -> np.ndarray:
    """Evaluate single focus Gaussian at coordinates with 3-sigma truncation.

    Parameters
    ----------
    coords : np.ndarray
        [N, 3] coordinates in physical mm [X, Y, Z]
    focus : Dict
        Focus dict with center, shape, radius/rx/ry/rz

    Returns
    -------
    np.ndarray
        [N] Gaussian values (float32)
    """
    center = np.array(focus["center"], dtype=np.float64)  # [px, py, pz]
    diff = coords - center  # [N, 3]

    if focus["shape"] == "sphere":
        sigma = float(focus["radius"])
        distances = np.linalg.norm(diff, axis=1)
        values = np.exp(-0.5 * (distances / sigma) ** 2)
        cutoff = 3.0 * sigma
        values = np.where(distances <= cutoff, values, 0.0)
    elif focus["shape"] == "ellipsoid":
        rx = float(focus["rx"])
        ry = float(focus["ry"])
        rz = float(focus["rz"])
        sigma = np.array([rx, ry, rz], dtype=np.float64)
        normalized = diff / sigma
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        values = np.exp(-0.5 * np.sum(normalized ** 2, axis=1))
        cutoff = 3.0 * max(rx, ry, rz)
        values = np.where(distances <= cutoff, values, 0.0)
    else:
        raise ValueError(f"Unknown shape type: {focus['shape']}")

    return values.astype(np.float32)
