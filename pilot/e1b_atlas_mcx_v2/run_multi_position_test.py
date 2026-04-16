#!/usr/bin/env python3
"""Test Green function at multiple source positions."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import (
    project_get_surface_coords,
    render_green_surface_projection,
    compute_ncc,
    compute_rmse,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_atlas_homogeneous_volume(atlas_bin_path: Path) -> np.ndarray:
    """Create homogeneous MCX volume preserving atlas shape."""
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8)
    volume = volume.reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ
    homogeneous = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)
    return homogeneous


def get_atlas_center(atlas_binary_xyz: np.ndarray, voxel_size_mm: float) -> tuple:
    """Find the center of mass and dorsal surface."""
    tissue_mask = atlas_binary_xyz > 0
    tissue_voxels = np.argwhere(tissue_mask)

    if len(tissue_voxels) == 0:
        raise ValueError("No tissue found in atlas")

    nx, ny, nz = atlas_binary_xyz.shape
    center = np.array([nx / 2, ny / 2, nz / 2])
    tissue_mm = (tissue_voxels - center + 0.5) * voxel_size_mm

    center_x = float(tissue_mm[:, 0].mean())
    center_y = float(tissue_mm[:, 1].mean())
    dorsal_z = float(tissue_mm[:, 2].max())

    return center_x, center_y, dorsal_z


def generate_source_positions(
    atlas_binary_xyz: np.ndarray,
    voxel_size_mm: float,
    n_positions: int = 5,
    depth_mm: float = 4.0,
) -> List[Tuple[str, np.ndarray]]:
    """Generate multiple source positions within the atlas.

    Returns list of (position_id, source_pos_mm) tuples.
    """
    center_x, center_y, dorsal_z = get_atlas_center(atlas_binary_xyz, voxel_size_mm)

    # Get tissue bounds
    tissue_mask = atlas_binary_xyz > 0
    tissue_voxels = np.argwhere(tissue_mask)
    nx, ny, nz = atlas_binary_xyz.shape
    center = np.array([nx / 2, ny / 2, nz / 2])
    tissue_mm = (tissue_voxels - center + 0.5) * voxel_size_mm

    x_min, x_max = tissue_mm[:, 0].min(), tissue_mm[:, 0].max()
    y_min, y_max = tissue_mm[:, 1].min(), tissue_mm[:, 1].max()

    positions = []

    # Position 1: Center
    source_z = dorsal_z - depth_mm
    positions.append(("center", np.array([center_x, center_y, source_z])))

    # Position 2: Left side (closer to left boundary)
    left_x = center_x + (x_max - center_x) * 0.5  # 50% towards left edge
    positions.append(("left", np.array([left_x, center_y, source_z])))

    # Position 3: Right side
    right_x = center_x + (x_min - center_x) * 0.5  # 50% towards right edge
    positions.append(("right", np.array([right_x, center_y, source_z])))

    # Position 4: Anterior (towards front)
    ant_y = center_y + (y_max - center_y) * 0.3
    positions.append(("anterior", np.array([center_x, ant_y, source_z])))

    # Position 5: Posterior (towards back)
    post_y = center_y + (y_min - center_y) * 0.3
    positions.append(("posterior", np.array([center_x, post_y, source_z])))

    return positions


def generate_mcx_config(
    source_pos_mm: np.ndarray,
    atlas_volume_zyx: np.ndarray,
    voxel_size_mm: float,
    tissue_params: dict,
    n_photons: int,
    config_id: str,
    output_dir: Path,
) -> Path:
    """Generate MCX JSON config."""
    nz, ny, nx = atlas_volume_zyx.shape

    # Convert centered coords to voxel indices
    center_x, center_y, center_z = nx / 2, ny / 2, nz / 2
    source_x_vox = source_pos_mm[0] / voxel_size_mm + center_x
    source_y_vox = source_pos_mm[1] / voxel_size_mm + center_y
    source_z_vox = source_pos_mm[2] / voxel_size_mm + center_z

    volume_bin_path = output_dir / "volume.bin"
    atlas_volume_zyx.tofile(volume_bin_path)

    mus_prime = tissue_params["mus_prime_mm"]
    g = tissue_params["g"]
    mus_total = mus_prime / (1.0 - g) if g < 1.0 else 0.0

    media = [
        {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},
        {
            "mua": tissue_params["mua_mm"],
            "mus": mus_total,
            "g": g,
            "n": tissue_params["n"],
        },
    ]

    config = {
        "Domain": {
            "VolumeFile": "volume.bin",
            "Dim": [nx, ny, nz],
            "OriginType": 1,
            "LengthUnit": voxel_size_mm,
            "Media": media,
        },
        "Session": {
            "Photons": n_photons,
            "RNGSeed": 42,
            "ID": config_id,
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5.0e-9,
            "DT": 5.0e-9,
        },
        "Optode": {
            "Source": {
                "Pos": [float(source_x_vox), float(source_y_vox), float(source_z_vox)],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "isotropic",
            }
        },
    }

    json_path = output_dir / f"{config_id}.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    return json_path


def run_mcx_simulation(
    json_path: Path, mcx_exec: str = "mcx.exe", timeout: int = 600
) -> Path:
    work_dir = json_path.parent
    session_id = json_path.stem
    output_jnii = work_dir / f"{session_id}.jnii"

    if output_jnii.exists():
        logger.info(f"Skipping MCX: {output_jnii} already exists")
        return output_jnii

    logger.info(f"Running MCX: {mcx_exec} -f {json_path.name}")
    subprocess.run(
        [mcx_exec, "-f", json_path.name],
        cwd=work_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return output_jnii


def load_mcx_output(jnii_path: Path) -> np.ndarray:
    """Load MCX JNII output."""
    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    fluence = np.array(data["NIFTIData"], dtype=np.float32)
    if fluence.ndim > 3:
        fluence = fluence[..., 0, 0]
    return fluence.transpose(2, 1, 0)  # ZYX to XYZ


def project_mcx_to_detector(
    fluence_xyz: np.ndarray,
    angle_deg: float,
    camera_distance_mm: float,
    fov_mm: float,
    detector_resolution: Tuple[int, int],
    voxel_size_mm: float,
) -> np.ndarray:
    """Project MCX 3D fluence to 2D detector with voxel coverage."""
    from surface_projection import rotation_matrix_y

    nx, ny, nz = fluence_xyz.shape
    width, height = detector_resolution

    nonzero = np.argwhere(fluence_xyz > 0)
    if len(nonzero) == 0:
        return np.zeros((height, width), dtype=np.float32)

    center = np.array([nx / 2, ny / 2, nz / 2])
    coords_mm = (nonzero.astype(np.float32) - center + 0.5) * voxel_size_mm
    values = fluence_xyz[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]

    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        coords_rot = coords_mm @ R.T
    else:
        coords_rot = coords_mm

    cam_x = coords_rot[:, 0]
    cam_y = coords_rot[:, 1]
    depths = camera_distance_mm - coords_rot[:, 2]

    half_w = fov_mm / 2
    half_h = fov_mm / 2
    px_size_x = fov_mm / width
    px_size_y = fov_mm / height

    projection = np.zeros((height, width), dtype=np.float32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    half_voxel = voxel_size_mm / 2

    for idx in range(len(cam_x)):
        px, py = cam_x[idx], cam_y[idx]
        d = depths[idx]

        if abs(px) > half_w or abs(py) > half_h or d < 0:
            continue

        u_start = int((px - half_voxel + half_w) / px_size_x)
        u_end = int((px + half_voxel + half_w) / px_size_x)
        v_start = int((py - half_voxel + half_h) / px_size_y)
        v_end = int((py + half_voxel + half_h) / px_size_y)

        u_start = max(0, u_start)
        u_end = min(width - 1, u_end)
        v_start = max(0, v_start)
        v_end = min(height - 1, v_end)

        for pu in range(u_start, u_end + 1):
            for pv in range(v_start, v_end + 1):
                if d < depth_map[pv, pu]:
                    depth_map[pv, pu] = d
                    projection[pv, pu] = values[idx]

    return projection


def run_single_position(
    pos_id: str,
    source_pos_mm: np.ndarray,
    config: dict,
    atlas_binary_xyz: np.ndarray,
    atlas_zyx: np.ndarray,
    mcx_exec: str,
    output_base: Path,
) -> Dict:
    """Run single position test."""
    config_id = f"MP-{pos_id}-D4mm"
    output_dir = output_base / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Position: {pos_id}")
    logger.info(
        f"Source: [{source_pos_mm[0]:.1f}, {source_pos_mm[1]:.1f}, {source_pos_mm[2]:.1f}] mm"
    )

    tissue_params = config["tissue_params"]
    mcx_cfg = config["mcx"]
    proj_cfg = config["projection"]

    # MCX
    json_path = generate_mcx_config(
        source_pos_mm,
        atlas_zyx,
        mcx_cfg["voxel_size_mm"],
        tissue_params,
        mcx_cfg["n_photons"],
        config_id,
        output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec)
    fluence_xyz = load_mcx_output(jnii_path)

    # Process angle 0 only for speed
    angle = 0.0
    mcx_proj = project_mcx_to_detector(
        fluence_xyz,
        angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        mcx_cfg["voxel_size_mm"],
    )

    # Green
    green_proj = render_green_surface_projection(
        source_pos_mm,
        atlas_binary_xyz,
        angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        tissue_params,
        mcx_cfg["voxel_size_mm"],
    )

    # Normalize for comparison
    mcx_norm = mcx_proj / (mcx_proj.max() + 1e-10)
    green_norm = green_proj / (green_proj.max() + 1e-10)

    ncc = compute_ncc(mcx_norm, green_norm)
    rmse = compute_rmse(mcx_norm, green_norm)

    # Save
    np.save(output_dir / f"mcx_projection_a{int(angle)}.npy", mcx_proj)
    np.save(output_dir / f"green_projection_a{int(angle)}.npy", green_proj)

    logger.info(f"  NCC: {ncc:.4f}, RMSE: {rmse:.4f}")

    return {
        "position_id": pos_id,
        "source_pos": source_pos_mm.tolist(),
        "ncc": ncc,
        "rmse": rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-position Green validation")
    parser.add_argument("--mcx", default="/mnt/f/win-pro/bin/mcx.exe")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/multi_position")
    parser.add_argument("--depth", type=float, default=4.0)
    args = parser.parse_args()

    setup_logging()

    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)

    # Load atlas
    atlas_bin_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    )
    atlas_binary_xyz = create_atlas_homogeneous_volume(atlas_bin_path)
    atlas_zyx = atlas_binary_xyz.transpose(2, 1, 0).astype(np.uint8)

    # Generate positions
    positions = generate_source_positions(
        atlas_binary_xyz,
        config["mcx"]["voxel_size_mm"],
        n_positions=5,
        depth_mm=args.depth,
    )

    output_base = Path(__file__).parent / args.output
    output_base.mkdir(parents=True, exist_ok=True)

    # Run each position
    results = []
    for pos_id, source_pos in positions:
        try:
            result = run_single_position(
                pos_id,
                source_pos,
                config,
                atlas_binary_xyz,
                atlas_zyx,
                args.mcx,
                output_base,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed at position {pos_id}: {e}")

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("Multi-Position Test Summary")
    logger.info(f"{'=' * 50}")
    for r in results:
        logger.info(f"  {r['position_id']:12s}: NCC = {r['ncc']:.4f}")

    # Save summary
    with open(output_base / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
