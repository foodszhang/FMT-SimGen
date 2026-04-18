"""MCX simulation runner for MVP pipeline.

Wraps fmt_simgen MCX utilities with MVP-specific configurations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MCX_EXE = "/mnt/f/win-pro/bin/mcx.exe"
VOLUME_PATH = Path("output/shared/mcx_volume_trunk.bin")
MATERIAL_PATH = Path("output/shared/mcx_material.yaml")
VOLUME_PATH_HOMOGENEOUS = Path("output/shared/mcx_volume_homogeneous.bin")
MATERIAL_PATH_HOMOGENEOUS = Path("output/shared/mcx_material_homogeneous.yaml")
VOLUME_PATH_ARCHIVED = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOXEL_SIZE_MM = 0.2
VOXEL_SIZE_MM_ARCHIVED = 0.4
VOLUME_SHAPE_ZYX = (104, 200, 190)
VOLUME_SHAPE_ZYX_ARCHIVED = (52, 100, 95)


def create_mcx_json_config(
    source_pattern: np.ndarray,
    source_origin: tuple[int, int, int],
    n_photons: int,
    output_dir: Path,
    config_id: str = "mvp_m1",
    homogeneous: bool = False,
) -> Path:
    """Create MCX JSON configuration file.

    Parameters
    ----------
    source_pattern : np.ndarray
        3D source pattern (NX, NY, NZ), float32.
    source_origin : tuple
        (ox, oy, oz) origin voxel indices for pattern placement.
    n_photons : int
        Number of photons.
    output_dir : Path
        Directory to save config and pattern files.
    config_id : str
        Configuration identifier.
    homogeneous : bool
        If True, use homogeneous volume (soft_tissue only).

    Returns
    -------
    Path
        Path to JSON config file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save source pattern as binary
    pattern_path = output_dir / f"source_{config_id}.bin"
    source_pattern.astype(np.float32).tofile(pattern_path)

    nx, ny, nz = source_pattern.shape
    ox, oy, oz = source_origin

    # Select volume and material paths
    volume_path = VOLUME_PATH_HOMOGENEOUS if homogeneous else VOLUME_PATH
    material_path = MATERIAL_PATH_HOMOGENEOUS if homogeneous else MATERIAL_PATH

    config = {
        "Domain": {
            "VolumeFile": str(volume_path.resolve()),
            "Dim": list(VOLUME_SHAPE_ZYX),
            "OriginType": 1,
            "LengthUnit": VOXEL_SIZE_MM,
            "Media": _load_material_list(material_path),
        },
        "Session": {
            "Photons": n_photons,
            "RNGSeed": 42,
            "ID": config_id,
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5e-8,
            "DT": 5e-8,
        },
        "Optode": {
            "Source": {
                "Pos": [ox, oy, oz],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {
                    "Nx": nx,
                    "Ny": ny,
                    "Nz": nz,
                    "Data": str(pattern_path.name),
                },
                "Param1": [nx, ny, nz],
            }
        },
    }

    json_path = output_dir / f"{config_id}.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Created MCX config: {json_path}")
    logger.info(f"Volume: {volume_path.name}, Homogeneous: {homogeneous}")
    return json_path


def _load_material_list(material_path: Path = MATERIAL_PATH) -> list:
    """Load material list from YAML."""
    import yaml

    with open(material_path) as f:
        materials = yaml.safe_load(f)

    # Materials already have correct format, just return them
    return materials


def run_mcx_simulation(
    json_path: Path,
    timeout_seconds: int = 600,
) -> Path:
    """Run MCX simulation.

    Parameters
    ----------
    json_path : Path
        Path to MCX JSON config.
    timeout_seconds : int
        Timeout in seconds.

    Returns
    -------
    Path
        Path to output .jnii file.
    """
    json_path = Path(json_path)
    work_dir = json_path.parent

    cmd = [MCX_EXE, "-f", json_path.name]

    logger.info(f"Running MCX: {' '.join(cmd)}")
    logger.info(f"Work dir: {work_dir}")

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    if result.returncode != 0:
        logger.error(f"MCX failed with code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"MCX failed: {result.stderr}")

    # Find output .jnii file
    config_id = json_path.stem
    jnii_path = work_dir / f"{config_id}.jnii"

    if not jnii_path.exists():
        raise FileNotFoundError(f"MCX output not found: {jnii_path}")

    logger.info(f"MCX output: {jnii_path}")
    return jnii_path


def load_mcx_fluence(jnii_path: Path) -> np.ndarray:
    """Load MCX fluence from .jnii file.

    Parameters
    ----------
    jnii_path : Path
        Path to .jnii file.

    Returns
    -------
    np.ndarray
        Fluence volume in XYZ order [X, Y, Z].
    """
    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    nifti = data["NIFTIData"] if isinstance(data, dict) else data

    if nifti.ndim == 5:
        nifti = nifti[:, :, :, 0, 0]

    # ZYX -> XYZ
    fluence_xyz = nifti.transpose(2, 1, 0).astype(np.float32)
    return fluence_xyz


def source_position_to_pattern_origin(
    source_center_mm: np.ndarray,
    pattern_shape: tuple[int, int, int],
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> tuple[int, int, int]:
    """Convert source center position to pattern origin voxel.

    Pattern origin is the corner of the pattern bounding box.

    Parameters
    ----------
    source_center_mm : np.ndarray
        Source center [x, y, z] in mm (centered coordinates).
    pattern_shape : tuple
        Pattern shape (nx, ny, nz).
    voxel_size_mm : float
        Voxel size.

    Returns
    -------
    tuple
        (ox, oy, oz) pattern origin in volume voxel indices.
    """
    nx, ny, nz = pattern_shape
    half_nx = nx // 2
    half_ny = ny // 2
    half_nz = nz // 2

    # Convert mm to voxel indices (volume is centered at origin)
    cx = source_center_mm[0] / voxel_size_mm
    cy = source_center_mm[1] / voxel_size_mm
    cz = source_center_mm[2] / voxel_size_mm

    # Volume center is at (nx/2, ny/2, nz/2) in voxel indices
    vol_center = np.array(VOLUME_SHAPE_ZYX[::-1]) / 2  # XYZ order
    vol_cx, vol_cy, vol_cz = vol_center

    # Pattern origin in volume coordinates
    ox = int(round(vol_cx + cx - half_nx))
    oy = int(round(vol_cy + cy - half_ny))
    oz = int(round(vol_cz + cz - half_nz))

    return (ox, oy, oz)


def run_mcx_for_source(
    source_pattern: np.ndarray,
    source_center_mm: np.ndarray,
    n_photons: int,
    output_dir: Path,
    config_id: str = "mvp_m1",
    homogeneous: bool = False,
) -> np.ndarray:
    """Run MCX simulation for a given source pattern.

    Parameters
    ----------
    source_pattern : np.ndarray
        3D source pattern (NX, NY, NZ).
    source_center_mm : np.ndarray
        Source center in mm.
    n_photons : int
        Number of photons.
    output_dir : Path
        Output directory.
    config_id : str
        Config identifier.
    homogeneous : bool
        If True, use homogeneous volume (soft_tissue only).

    Returns
    -------
    np.ndarray
        Fluence volume in XYZ order.
    """
    # Get pattern origin
    pattern_shape = source_pattern.shape
    origin = source_position_to_pattern_origin(
        source_center_mm, pattern_shape, VOXEL_SIZE_MM
    )
    logger.info(f"Pattern shape: {pattern_shape}, origin: {origin}")

    # Create config
    json_path = create_mcx_json_config(
        source_pattern, origin, n_photons, output_dir, config_id, homogeneous
    )

    # Run MCX
    jnii_path = run_mcx_simulation(json_path)

    # Load fluence
    fluence = load_mcx_fluence(jnii_path)

    return fluence
