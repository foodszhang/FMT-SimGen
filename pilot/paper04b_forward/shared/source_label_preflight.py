"""Source label preflight check.

Validates that source positions are in valid tissue before running MCX.
Opt-in logging, strict=False by default (log warning but don't fail).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_atlas_volume(volume_path: Path) -> np.ndarray:
    """Load atlas volume from binary file."""
    volume = np.fromfile(volume_path, dtype=np.uint8)
    return volume


def load_material_labels(material_path: Path) -> Dict[int, str]:
    """Load material label mapping from YAML."""
    with open(material_path) as f:
        material = yaml.safe_load(f)

    label_map = {}
    for name, info in material.get("materials", {}).items():
        label = info.get("label", 0)
        label_map[label] = name
    return label_map


def xyz_to_ijk(
    xyz_mm: np.ndarray,
    volume_shape: Tuple[int, int, int],
    voxel_size_mm: float = 0.4,
) -> Tuple[int, int, int]:
    """Convert XYZ coordinates in mm to voxel indices.

    Volume is stored as (Z, Y, X) or (X, Y, Z) depending on format.
    Assumes volume center at (0, 0, 0).

    Parameters
    ----------
    xyz_mm : np.ndarray
        (X, Y, Z) coordinates in mm.
    volume_shape : tuple
        Shape of volume (NX, NY, NZ).
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    tuple of (ix, iy, iz) voxel indices.
    """
    nx, ny, nz = volume_shape

    ix = int(round((xyz_mm[0] / voxel_size_mm) + nx / 2))
    iy = int(round((xyz_mm[1] / voxel_size_mm) + ny / 2))
    iz = int(round((xyz_mm[2] / voxel_size_mm) + nz / 2))

    return (ix, iy, iz)


def check_source_label(
    xyz_mm: np.ndarray,
    volume: np.ndarray,
    volume_shape: Tuple[int, int, int],
    label_map: Dict[int, str],
    voxel_size_mm: float = 0.4,
) -> Tuple[int, str, bool]:
    """Check if source position is in valid tissue.

    Parameters
    ----------
    xyz_mm : np.ndarray
        Source position (X, Y, Z) in mm.
    volume : np.ndarray
        Atlas volume with tissue labels.
    volume_shape : tuple
        (NX, NY, NZ) shape.
    label_map : dict
        Label number -> name mapping.
    voxel_size_mm : float
        Voxel size.

    Returns
    -------
    tuple of (label, name, is_valid)
        label: int label value
        name: str tissue name
        is_valid: bool (True if in soft_tissue, False if in organ/air)
    """
    ix, iy, iz = xyz_to_ijk(xyz_mm, volume_shape, voxel_size_mm)

    nx, ny, nz = volume_shape
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return (0, "out_of_bounds", False)

    volume_3d = volume.reshape(volume_shape)
    label = int(volume_3d[ix, iy, iz])

    name = label_map.get(label, f"unknown_{label}")

    valid_tissues = {"soft_tissue", "muscle", "skin"}
    is_valid = name in valid_tissues

    return (label, name, is_valid)


def preflight_check(
    xyz_mm: np.ndarray,
    position_name: str,
    volume_path: Path,
    material_path: Path,
    volume_shape: Tuple[int, int, int] = (95, 100, 52),
    voxel_size_mm: float = 0.4,
    strict: bool = False,
    log_path: Optional[Path] = None,
) -> bool:
    """Run preflight check for a source position.

    Parameters
    ----------
    xyz_mm : np.ndarray
        Source position (X, Y, Z) in mm.
    position_name : str
        Name of position for logging.
    volume_path : Path
        Path to atlas volume binary file.
    material_path : Path
        Path to material YAML file.
    volume_shape : tuple
        Shape of volume (NX, NY, NZ). Default is downsampled 2x.
    voxel_size_mm : float
        Voxel size in mm.
    strict : bool
        If True, raise error on invalid position. If False, log warning.
    log_path : Path, optional
        Path to write preflight log.

    Returns
    -------
    bool
        True if position is valid, False otherwise.
    """
    volume = load_atlas_volume(volume_path)
    label_map = load_material_labels(material_path)

    label, name, is_valid = check_source_label(
        xyz_mm, volume, volume_shape, label_map, voxel_size_mm
    )

    log_entry = {
        "position": position_name,
        "xyz_mm": xyz_mm.tolist(),
        "label": label,
        "tissue": name,
        "is_valid": is_valid,
    }

    if log_path:
        import json

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    if not is_valid:
        msg = (
            f"Source position {position_name} at {xyz_mm} is in '{name}' (label={label}). "
            f"Expected soft_tissue/muscle/skin."
        )
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return is_valid


def preflight_all_positions(
    positions: Dict[str, "SurfacePosition"],
    volume_path: Path,
    material_path: Path,
    volume_shape: Tuple[int, int, int] = (95, 100, 52),
    voxel_size_mm: float = 0.4,
    strict: bool = False,
    log_path: Optional[Path] = None,
) -> Dict[str, bool]:
    """Run preflight check for all positions.

    Returns
    -------
    dict mapping position name to is_valid.
    """
    results = {}
    for key, pos in positions.items():
        results[key] = preflight_check(
            pos.xyz_mm,
            pos.name,
            volume_path,
            material_path,
            volume_shape,
            voxel_size_mm,
            strict,
            log_path,
        )
    return results
