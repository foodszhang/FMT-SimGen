#!/usr/bin/env python3
"""Generate MCX surface ground truth for E1c.

Generates top surface fluence images for homogeneous medium
using MCX Monte Carlo simulation.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilot.e0_psf_validation.mcx_point_source import (
    create_homogeneous_volume,
    generate_mcx_config_json,
    run_mcx_simulation,
    load_mcx_fluence,
)


def config_to_world(source_center: list) -> np.ndarray:
    """Convert config source_center to world coordinates.

    Args:
        source_center: [x_mm, y_mm, depth_from_dorsal_mm]

    Returns:
        world coordinates [x, y, z]
    """
    x_mm, y_mm, depth = source_center
    z_world = 10.0 - depth
    return np.array([x_mm, y_mm, z_world])


def world_to_mcx_source_pos(
    source_world: np.ndarray,
    voxel_size_mm: float = 0.1,
) -> Tuple[float, float, float]:
    """Convert world source position to MCX voxel coordinates.

    MCX coordinate system:
    - x_mcx = x_world + 15.0 (in mm)
    - y_mcx = y_world + 15.0 (in mm)
    - z_mcx = depth_from_dorsal (in mm)

    MCX source position is in voxel coordinates.

    Args:
        source_world: [x, y, z] in world coordinates (mm)
        voxel_size_mm: voxel size

    Returns:
        (x_vox, y_vox, z_vox) in MCX voxel coordinates
    """
    x_world, y_world, z_world = source_world
    depth = 10.0 - z_world

    x_mcx_mm = x_world + 15.0
    y_mcx_mm = y_world + 15.0
    z_mcx_mm = depth

    x_vox = x_mcx_mm / voxel_size_mm
    y_vox = y_mcx_mm / voxel_size_mm
    z_vox = z_mcx_mm / voxel_size_mm

    return x_vox, y_vox, z_vox


def generate_surface_gt_for_config(
    config_id: str,
    cfg: dict,
    mcx_config: dict,
    output_dir: Path,
    mcx_exec: str = "mcx",
) -> Dict:
    """Generate surface GT for a single configuration.

    Args:
        config_id: configuration ID (e.g., "M01")
        cfg: configuration dict from config.yaml
        mcx_config: MCX parameters from config.yaml
        output_dir: output directory

    Returns:
        dict with metadata and paths
    """
    source_center = cfg["source_center"]
    source_world = config_to_world(source_center)

    tissue_type = cfg["tissue_type"]
    mua_mm = cfg["mua_mm"]
    mus_mm = cfg["mus_mm"]
    g = cfg["g"]
    n = cfg["n"]

    vol_size_mm = (30.0, 30.0, 20.0)
    voxel_size_mm = mcx_config.get("voxel_size_mm", 0.1)
    n_photons = mcx_config.get("n_sim", 20) * 1_000_000

    sample_dir = output_dir / config_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating GT for {config_id}: {tissue_type} @ depth={source_center[2]}mm"
    )
    logger.info(f"  World: {source_world}")

    volume_dict = create_homogeneous_volume(
        tissue_mu_a=mua_mm,
        tissue_mu_sp=mus_mm * (1 - g),
        tissue_g=g,
        tissue_n=n,
        vol_size_mm=vol_size_mm,
        voxel_size_mm=voxel_size_mm,
    )

    depth_mm = source_center[2]

    source_pos_xy = (source_world[0] + 15.0, source_world[1] + 15.0)

    json_path, volume_bin_path = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=depth_mm,
        source_pos_xy=source_pos_xy,
        n_photons=n_photons,
        session_id=config_id,
        output_dir=sample_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec=mcx_exec)

    fluence = load_mcx_fluence(jnii_path)
    logger.info(f"  Fluence shape: {fluence.shape}")

    nx, ny, nz = fluence.shape
    top_z_idx = 0
    surface_fluence = fluence[:, :, top_z_idx]

    logger.info(f"  Surface shape: {surface_fluence.shape}")
    logger.info(f"  Surface sum: {surface_fluence.sum():.4e}")

    image_size = mcx_config.get("image_size", 256)
    pixel_size_mm = mcx_config.get("pixel_size_mm", 0.1)

    gt_image = mcx_surface_to_gt_image(
        surface_fluence=surface_fluence,
        source_world=source_world,
        voxel_size_mm=voxel_size_mm,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
    )

    x_coords_mm, y_coords_mm = build_gt_coords(image_size, pixel_size_mm)

    npz_path = sample_dir / f"{config_id}_surface_gt.npz"
    np.savez(
        npz_path,
        surface_image=gt_image,
        x_coords_mm=x_coords_mm,
        y_coords_mm=y_coords_mm,
        source_world=source_world,
        source_depth_from_dorsal_mm=source_center[2],
        tissue_type=tissue_type,
        mua_mm=mua_mm,
        mus_mm=mus_mm,
        g=g,
        n=n,
    )

    meta = {
        "config_id": config_id,
        "source_center_config": source_center,
        "source_world": source_world.tolist(),
        "source_depth_from_dorsal_mm": source_center[2],
        "tissue_type": tissue_type,
        "mua_mm": mua_mm,
        "mus_mm": mus_mm,
        "g": g,
        "n": n,
        "image_size": image_size,
        "pixel_size_mm": pixel_size_mm,
        "mcx_fluence_shape": list(fluence.shape),
        "mcx_top_z_idx": top_z_idx,
        "npz_path": str(npz_path),
    }

    meta_path = sample_dir / f"{config_id}_surface_gt_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  Saved: {npz_path}")

    return meta


def mcx_surface_to_gt_image(
    surface_fluence: np.ndarray,
    source_world: np.ndarray,
    voxel_size_mm: float,
    image_size: int,
    pixel_size_mm: float,
) -> np.ndarray:
    """Convert MCX surface fluence to GT image format.

    MCX outputs surface fluence in MCX coordinates [x_mcx, y_mcx].
    Need to map to world coordinates and then to image grid.

    Args:
        surface_fluence: MCX surface fluence [nx, ny] in MCX coords
        source_world: source position in world coords
        voxel_size_mm: voxel size
        image_size: target image size
        pixel_size_mm: target pixel size

    Returns:
        gt_image: [H, W] array in world coordinate order
    """
    nx, ny = surface_fluence.shape

    x_mcx_mm = np.arange(nx) * voxel_size_mm
    y_mcx_mm = np.arange(ny) * voxel_size_mm

    x_world_mm = x_mcx_mm - 15.0
    y_world_mm = y_mcx_mm - 15.0

    fov_mm = image_size * pixel_size_mm
    gt_image = np.zeros((image_size, image_size), dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            x_w = x_world_mm[i]
            y_w = y_world_mm[j]

            if abs(x_w) > fov_mm / 2 or abs(y_w) > fov_mm / 2:
                continue

            col = int((x_w + fov_mm / 2) / pixel_size_mm)
            row = int((y_w + fov_mm / 2) / pixel_size_mm)

            if 0 <= col < image_size and 0 <= row < image_size:
                gt_image[row, col] += surface_fluence[i, j]

    return gt_image


def build_gt_coords(
    image_size: int, pixel_size_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build coordinate arrays for GT image.

    Args:
        image_size: image dimension
        pixel_size_mm: pixel size

    Returns:
        (x_coords_mm, y_coords_mm): 1D arrays of coordinates
    """
    fov_mm = image_size * pixel_size_mm
    coords_mm = (np.arange(image_size) - image_size / 2 + 0.5) * pixel_size_mm
    return coords_mm, coords_mm


def main():
    parser = argparse.ArgumentParser(description="Generate MCX surface GT for E1c")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--mcx",
        default="mcx",
        help="MCX executable",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config IDs to run (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / config["output"]["gt_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_ids = args.configs if args.configs else list(config["configs"].keys())

    results = {}
    for config_id in config_ids:
        if config_id not in config["configs"]:
            logger.warning(f"Config {config_id} not found, skipping")
            continue

        cfg = config["configs"][config_id]
        mcx_config = config["mcx"]

        try:
            meta = generate_surface_gt_for_config(
                config_id=config_id,
                cfg=cfg,
                mcx_config=mcx_config,
                output_dir=output_dir,
                mcx_exec=args.mcx,
            )
            results[config_id] = meta
        except Exception as e:
            logger.error(f"Failed {config_id}: {e}")
            import traceback

            traceback.print_exc()
            results[config_id] = {"error": str(e)}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Generated GT for {len(results)} configs")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
