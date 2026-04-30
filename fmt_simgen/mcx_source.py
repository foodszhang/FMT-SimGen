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
        - voxel_size_mm: float (e.g., 0.2)
        - volume_shape: [nz, ny, nx] (ZYX order)

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, int]]
        - pattern: float32 3D array (nx, ny, nz) where:
          - nx = volume Z range (pattern x-index -> volume Z)
          - ny = volume Y range (pattern y-index -> volume Y)
          - nz = volume X range (pattern z-index -> volume X)
          MCX interprets pattern[x,y,z] as volume[Pos_z+x, Pos_y+y, Pos_x+z]
        - origin: (x0, y0, z0) in volume voxel coordinates (XYZ order)

    Raises
    ------
    ValueError
        If no foci produce non-zero voxels or focus is outside volume bounds.
    """
    # Compute global pattern bbox from all foci
    global_min_voxel = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max_voxel = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for focus in foci:
        focus_min, focus_max = _focus_bbox_voxels(focus, voxel_size_mm, source_type)
        global_min_voxel = np.minimum(global_min_voxel, focus_min)
        global_max_voxel = np.maximum(global_max_voxel, focus_max)

    # Apply 2-voxel padding
    pad = 2
    global_min_voxel = np.floor(global_min_voxel).astype(int) - pad
    global_max_voxel = np.ceil(global_max_voxel).astype(int) + pad

    # Clamp to volume bounds
    # volume_shape = (nz, ny, nx) in ZYX order
    nz_vol, ny_vol, nx_vol = volume_shape[0], volume_shape[1], volume_shape[2]
    x0 = max(0, int(global_min_voxel[0]))
    y0 = max(0, int(global_min_voxel[1]))
    z0 = max(0, int(global_min_voxel[2]))
    x1 = min(nx_vol, int(global_max_voxel[0]))
    y1 = min(ny_vol, int(global_max_voxel[1]))
    z1 = min(nz_vol, int(global_max_voxel[2]))

    # Validate non-empty bbox
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        raise ValueError(
            f"Pattern bbox is empty after clamping: "
            f"(x0={x0}, x1={x1}, y0={y0}, y1={y1}, z0={z0}, z1={z1}). "
            f"Tumor focus may be outside MCX volume bounds."
        )

    # MCX mapping: pattern[x,y,z] -> volume[Pos_z+x, Pos_y+y, Pos_x+z]
    # Pattern dimensions:
    # - nx = volume Z range (pattern x-index -> volume Z)
    # - ny = volume Y range (pattern y-index -> volume Y)
    # - nz = volume X range (pattern z-index -> volume X)
    pattern_shape = (z1 - z0, y1 - y0, x1 - x0)
    logger.debug(
        f"Pattern bbox: ({x0},{y0},{z0}) to ({x1},{y1},{z1}), "
        f"shape (Nx,Ny,Nz)={pattern_shape}"
    )

    # Evaluate pattern at voxel centers
    pattern = _evaluate_pattern_voxels(
        (x0, y0, z0), (x1, y1, z1), foci, voxel_size_mm, source_type
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
    voxel_size_mm: float,
    source_type: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box for a focus in voxel coordinates.

    Parameters
    ----------
    focus : Dict
        Focus dict with center, shape, radius/rx/ry/rz
    voxel_size_mm : float
        Voxel size in mm
    source_type : str
        "gaussian" or "uniform" — determines cutoff distance

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (min_voxel, max_voxel) in voxel coordinates [ix, iy, iz]
    """
    center = np.array(focus["center"], dtype=np.float64)  # [px, py, pz]

    if focus["shape"] == "sphere":
        radius_mm = float(focus["radius"])
    elif focus["shape"] == "ellipsoid":
        radius_mm = max(float(focus["rx"]), float(focus["ry"]), float(focus["rz"]))
    else:
        raise ValueError(f"Unknown shape type: {focus['shape']}")

    if source_type == "uniform":
        cutoff_mm = radius_mm
    else:
        cutoff_mm = 3.0 * radius_mm  # 3-sigma for Gaussian

    # Bbox in physical mm
    bbox_min_mm = center - cutoff_mm
    bbox_max_mm = center + cutoff_mm

    # Convert to voxel coordinates (floor for min, ceil for max)
    # center is already in MCX volume world coordinates (trunk-local frame),
    # no offset subtraction needed.
    voxel_min = bbox_min_mm / voxel_size_mm
    voxel_max = bbox_max_mm / voxel_size_mm

    return voxel_min, voxel_max


def _evaluate_pattern_voxels(
    bbox_min: Tuple[int, int, int],
    bbox_max: Tuple[int, int, int],
    foci: List[Dict],
    voxel_size_mm: float,
    source_type: str = "gaussian",
) -> np.ndarray:
    """Evaluate multi-focal source pattern at voxel centers.

    Parameters
    ----------
    bbox_min : Tuple[int, int, int]
        (x0, y0, z0) in volume voxel coords (XYZ order)
    bbox_max : Tuple[int, int, int]
        (x1, y1, z1) in volume voxel coords (exclusive, XYZ order)
    foci : List[Dict]
        Focus definitions
    voxel_size_mm : float
        Voxel size mm
    source_type : str
        "gaussian" or "uniform"

    Returns
    -------
    np.ndarray
        float32 pattern (nx, ny, nz) where:
        - nx = volume Z range (pattern x-index -> volume Z)
        - ny = volume Y range (pattern y-index -> volume Y)
        - nz = volume X range (pattern z-index -> volume X)
        MCX interprets pattern[x,y,z] as volume[Pos_z+x, Pos_y+y, Pos_x+z]
    """
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    # MCX mapping: pattern[x,y,z] -> volume[Pos_z+x, Pos_y+y, Pos_x+z]
    # Pattern dimensions:
    # - nx = volume Z range (pattern x-index -> volume Z)
    # - ny = volume Y range (pattern y-index -> volume Y)
    # - nz = volume X range (pattern z-index -> volume X)
    nx = z1 - z0  # volume Z range -> pattern x-index range
    ny = y1 - y0  # volume Y range -> pattern y-index range
    nz = x1 - x0  # volume X range -> pattern z-index range

    # Generate physical coordinates (volume XYZ in mm)
    # gx[i] = volume Z coordinate for pattern x-index i
    gx = np.arange(nx) * voxel_size_mm + z0 * voxel_size_mm + voxel_size_mm / 2
    # gy[j] = volume Y coordinate for pattern y-index j
    gy = np.arange(ny) * voxel_size_mm + y0 * voxel_size_mm + voxel_size_mm / 2
    # gz[k] = volume X coordinate for pattern z-index k
    gz = np.arange(nz) * voxel_size_mm + x0 * voxel_size_mm + voxel_size_mm / 2

    gx3d, gy3d, gz3d = np.meshgrid(gx, gy, gz, indexing="ij")
    # coords in volume XYZ order: [X, Y, Z] = [gz3d, gy3d, gx3d]
    coords = np.column_stack([gz3d.ravel(), gy3d.ravel(), gx3d.ravel()])

    # Evaluate each focus and take max
    values = np.zeros(len(coords), dtype=np.float32)
    for focus in foci:
        if source_type == "uniform":
            focus_values = _evaluate_single_focus_uniform(coords, focus)
        else:
            focus_values = _evaluate_single_focus_gaussian(coords, focus)
        values = np.maximum(values, focus_values)

    # Reshape to (nx, ny, nz) matching pattern indexing
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


def _evaluate_single_focus_uniform(coords: np.ndarray, focus: Dict) -> np.ndarray:
    """Evaluate single focus uniform (binary) at coordinates within radius.

    Parameters
    ----------
    coords : np.ndarray
        [N, 3] coordinates in physical mm [X, Y, Z]
    focus : Dict
        Focus dict with center, shape, radius/rx/ry/rz

    Returns
    -------
    np.ndarray
        [N] Binary values: 1.0 inside, 0.0 outside (float32)
    """
    center = np.array(focus["center"], dtype=np.float64)  # [px, py, pz]
    diff = coords - center  # [N, 3]

    if focus["shape"] == "sphere":
        radius = float(focus["radius"])
        distances = np.linalg.norm(diff, axis=1)
        values = np.where(distances <= radius, 1.0, 0.0)
    elif focus["shape"] == "ellipsoid":
        rx = float(focus["rx"])
        ry = float(focus["ry"])
        rz = float(focus["rz"])
        # Normalized distance: sum((d/r)^2) <= 1 means inside
        normalized = diff / np.array([rx, ry, rz])
        inside = np.sum(normalized ** 2, axis=1) <= 1.0
        values = np.where(inside, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown shape type: {focus['shape']}")

    return values.astype(np.float32)
