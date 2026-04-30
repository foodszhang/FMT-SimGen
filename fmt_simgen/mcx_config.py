"""
MCX JSON configuration generator for pattern3d sources.

Generates MCX JSON configuration files and source pattern binaries
from FMT-SimGen tumor_params.json.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml

from fmt_simgen.frame_contract import TRUNK_GRID_SHAPE
from fmt_simgen.mcx_source import tumor_params_to_mcx_pattern

logger = logging.getLogger(__name__)


def generate_mcx_config(
    sample_id: str,
    tumor_params: Dict,
    mcx_config: Dict,
    output_dir: Union[str, Path],
) -> str:
    """Generate MCX JSON config file and source binary for a tumor sample.

    Parameters
    ----------
    sample_id : str
        Sample identifier (e.g., "0000" or "sample_0000")
    tumor_params : Dict
        Tumor parameters dict from tumor_params.json
    mcx_config : Dict
        MCX configuration section from default.yaml
    output_dir : str | Path
        Directory to write output files

    Returns
    -------
    str
        Absolute path to generated JSON config file

    Output files (written to output_dir):
        - {sample_id}.json: MCX JSON configuration
        - source-{sample_id}.bin: float32 pattern binary (ZYX order)

    Raises
    ------
    ValueError
        If pattern has zero non-zero voxels after thresholding.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pattern and origin
    pattern, origin = tumor_params_to_mcx_pattern(tumor_params, mcx_config)

    # Validate non-zero count
    nonzero_count = np.count_nonzero(pattern)
    if nonzero_count == 0:
        raise ValueError(f"Sample {sample_id}: pattern has zero non-zero voxels")

    # Save binary file: transpose from (nx, ny, nz) to (nz, ny, nx) for MCX
    pattern_t = pattern.transpose(2, 1, 0)
    source_bin_path = output_dir / f"source-{sample_id}.bin"
    pattern_t.tofile(source_bin_path)
    logger.debug(f"Saved source binary: {source_bin_path} (shape {pattern_t.shape})")

    # Load media list from material YAML
    material_path = Path(
        mcx_config.get("material_path", "output/shared/mcx_material.yaml")
    )
    if material_path.exists():
        with open(material_path, "r") as f:
            media_list = yaml.safe_load(f)
    else:
        logger.warning(
            f"Material YAML not found: {material_path}, using empty Media list"
        )
        media_list = []

    # Build JSON config
    # Pattern shape is (nx, ny, nz) where:
    # - nx = volume Z range (pattern x-index -> volume Z)
    # - ny = volume Y range (pattern y-index -> volume Y)
    # - nz = volume X range (pattern z-index -> volume X)
    # MCX: pattern[x,y,z] -> volume[Pos_z+x, Pos_y+y, Pos_x+z]
    pnx, pny, pnz = pattern.shape
    x0, y0, z0 = origin  # (x0, y0, z0) in XYZ voxel order (volume coordinates)

    # Determine volume file path.
    # VolumeFile in MCX JSON is relative to the directory where MCX runs (sample dir).
    # Compute the relative path from output_dir (sample directory) to the volume file
    # by going up from output_dir to the project root, then down to the volume.
    project_root = Path(__file__).parent.parent
    volume_file_rel = mcx_config.get(
        "volume_path", "output/shared/mcx_volume_trunk.bin"
    )
    volume_file_abs = (project_root / volume_file_rel).resolve()

    # Compute relative path from sample output_dir to project_root
    try:
        rel_to_project = Path(volume_file_abs).relative_to(project_root.resolve())
        # Compute number of levels to go up from output_dir to project_root
        out_parts = output_dir.resolve().parts
        proj_parts = project_root.resolve().parts
        common = len([a for a, b in zip(out_parts, proj_parts) if a == b])
        levels_up = len(out_parts) - common
        volume_file = Path("/".join([".."] * levels_up)) / rel_to_project
    except ValueError:
        # Fall back to absolute path
        volume_file = volume_file_abs

    # Session ID: use sample_id without any prefix for MCX output naming
    session_id = sample_id

    # RNG seed: deterministic but varied
    rng_seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % (2**31)

    config_dict = {
        "Domain": {
            "VolumeFile": str(volume_file),
            "Dim": list(
                mcx_config["volume_shape"]
            ),  # [Z, Y, X] = TRUNK_GRID_SHAPE[::-1]
            "OriginType": 1,
            "LengthUnit": float(mcx_config["voxel_size_mm"]),
            "Media": media_list,
        },
        "Session": {
            "Photons": int(1e7),
            "RNGSeed": rng_seed,
            "ID": session_id,
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5.0e-08,
            "DT": 5.0e-08,
        },
        "Optode": {
            "Source": {
                # Pos must be in ZYX order to match Dim=[Z,Y,X]
                # MCX: pattern[x,y,z] -> volume[Pos_z+x, Pos_y+y, Pos_x+z]
                # x0,y0,z0 are volume XYZ coordinates from tumor_params_to_mcx_pattern
                # Pos_z = z0 (volume Z offset), Pos_y = y0, Pos_x = x0
                "Pos": [int(z0), int(y0), int(x0)],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {
                    # Nx, Ny, Nz match pattern shape (pnx, pny, pnz)
                    "Nx": int(pnx),
                    "Ny": int(pny),
                    "Nz": int(pnz),
                    "Data": f"source-{sample_id}.bin",
                },
                "Param1": [int(pnx), int(pny), int(pnz)],
            }
        },
    }

    # Save JSON
    json_path = output_dir / f"{sample_id}.json"
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved MCX config: {json_path}")

    return str(json_path)
