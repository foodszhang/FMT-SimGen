#!/usr/bin/env python3
"""Multi-position × best-view Green validation.

Place sources at different body locations (dorsal, left, right, ventral)
and capture from the angle where source is closest to camera.
"""

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
    volume_xyz = volume.transpose(2, 1, 0)
    homogeneous = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)
    return homogeneous


def get_surface_positions(atlas_binary_xyz: np.ndarray, voxel_size_mm: float) -> dict:
    """Find surface positions at Y=center slice.

    Returns surface positions in mm (centered coordinates).
    """
    ny = atlas_binary_xyz.shape[1]
    y_center = ny // 2

    # Get slice at Y=center
    slice_xz = atlas_binary_xyz[:, y_center, :]  # (X, Z)

    # Find tissue voxels
    tissue_x, tissue_z = np.where(slice_xz > 0)

    if len(tissue_x) == 0:
        raise ValueError("No tissue found")

    # Convert to mm (centered)
    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2

    tissue_x_mm = (tissue_x - x_center + 0.5) * voxel_size_mm
    tissue_z_mm = (tissue_z - z_center + 0.5) * voxel_size_mm

    # Find surfaces
    dorsal_z = tissue_z_mm.max()  # max Z = dorsal (back)
    ventral_z = tissue_z_mm.min()  # min Z = ventral (belly)
    left_x = tissue_x_mm.min()  # min X = left
    right_x = tissue_x_mm.max()  # max X = right

    return {
        "dorsal_z": dorsal_z,
        "ventral_z": ventral_z,
        "left_x": left_x,
        "right_x": right_x,
        "center_x": (left_x + right_x) / 2,
        "center_z": (ventral_z + dorsal_z) / 2,
    }


def generate_test_configs(
    surface_positions: dict,
    depth_mm: float = 4.0,
) -> List[Dict]:
    """Generate test configurations for different positions.

    Each config places source at depth_mm below the nearest surface,
    with best viewing angle pointing at that surface.
    """
    cx = surface_positions["center_x"]
    cz = surface_positions["center_z"]
    y_center = 2.4  # From earlier calculation

    configs = []

    # P1: Dorsal center - source below dorsal surface, view from 0° (back)
    p1_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": "P1-dorsal",
            "source_pos": [cx, y_center, p1_z],
            "best_angle": 0.0,
            "description": "Dorsal center, 4mm deep",
            "surface": "dorsal",
        }
    )

    # P2: Left side - source to the right of left surface, view from +90°
    # +90°: camera looks from -X direction (LEFT side), so left surface is visible
    p2_x = surface_positions["left_x"] + depth_mm
    configs.append(
        {
            "id": "P2-left",
            "source_pos": [p2_x, y_center, cz],
            "best_angle": 90.0,
            "description": "Left side, 4mm from left surface",
            "surface": "left",
        }
    )

    # P3: Right side - source to the left of right surface, view from -90°
    # -90°: camera looks from +X direction (RIGHT side), so right surface is visible
    p3_x = surface_positions["right_x"] - depth_mm
    configs.append(
        {
            "id": "P3-right",
            "source_pos": [p3_x, y_center, cz],
            "best_angle": -90.0,
            "description": "Right side, 4mm from right surface",
            "surface": "right",
        }
    )

    # P4: Dorsal-lateral - left side of dorsal surface, view from -30°
    left_offset = (surface_positions["center_x"] - surface_positions["left_x"]) * 0.5
    p4_x = cx - left_offset
    p4_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": "P4-dorsal-lateral",
            "source_pos": [p4_x, y_center, p4_z],
            "best_angle": -30.0,
            "description": "Dorsal-left, 4mm from dorsal",
            "surface": "dorsal-lateral",
        }
    )

    # P5: Ventral - source above ventral surface, view from 180° (if available)
    # Since camera only does -90 to +90, use ±60° as best available
    p5_z = surface_positions["ventral_z"] + depth_mm
    configs.append(
        {
            "id": "P5-ventral",
            "source_pos": [cx, y_center, p5_z],
            "best_angle": 60.0,  # Best available for ventral
            "description": "Ventral, 4mm from belly",
            "surface": "ventral",
        }
    )

    return configs


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
        logger.info(f"  Skipping MCX: {output_jnii} exists")
        return output_jnii

    logger.info(f"  Running MCX...")
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
    # JNII is already in XYZ order - NO transpose needed
    return fluence


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


def run_single_config(
    config: dict,
    tissue_params: dict,
    mcx_cfg: dict,
    proj_cfg: dict,
    atlas_binary_xyz: np.ndarray,
    atlas_zyx: np.ndarray,
    mcx_exec: str,
    output_base: Path,
) -> dict:
    """Run single test configuration."""
    config_id = config["id"]
    source_pos = np.array(config["source_pos"])
    best_angle = config["best_angle"]

    output_dir = output_base / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Config: {config_id}")
    logger.info(f"  Description: {config['description']}")
    logger.info(
        f"  Source: [{source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f}] mm"
    )
    logger.info(f"  Best angle: {best_angle}°")

    # Run MCX
    json_path = generate_mcx_config(
        source_pos,
        atlas_zyx,
        mcx_cfg["voxel_size_mm"],
        tissue_params,
        mcx_cfg["n_photons"],
        config_id,
        output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec)
    fluence_xyz = load_mcx_output(jnii_path)

    # Project MCX at best angle
    mcx_proj = project_mcx_to_detector(
        fluence_xyz,
        best_angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        mcx_cfg["voxel_size_mm"],
    )

    # Generate Green projection
    green_proj = render_green_surface_projection(
        source_pos,
        atlas_binary_xyz,
        best_angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        tissue_params,
        mcx_cfg["voxel_size_mm"],
    )

    # Compute metrics
    mcx_norm = mcx_proj / (mcx_proj.max() + 1e-10)
    green_norm = green_proj / (green_proj.max() + 1e-10)

    ncc = compute_ncc(mcx_norm, green_norm)
    rmse = compute_rmse(mcx_norm, green_norm)

    logger.info(f"  MCX peak: {mcx_proj.max():.4e}")
    logger.info(f"  Green peak: {green_proj.max():.4e}")
    logger.info(f"  NCC: {ncc:.4f}, RMSE: {rmse:.4f}")

    # Save
    np.save(output_dir / f"mcx_a{int(best_angle)}.npy", mcx_proj)
    np.save(output_dir / f"green_a{int(best_angle)}.npy", green_proj)

    # Test additional angles
    angle_offsets = [0, 15, 30, 45, 60]
    angle_results = {}

    for offset in angle_offsets:
        test_angle = best_angle + offset
        if abs(test_angle) > 90:
            continue

        mcx_proj_off = project_mcx_to_detector(
            fluence_xyz,
            test_angle,
            proj_cfg["camera_distance_mm"],
            proj_cfg["fov_mm"],
            tuple(proj_cfg["detector_resolution"]),
            mcx_cfg["voxel_size_mm"],
        )

        green_proj_off = render_green_surface_projection(
            source_pos,
            atlas_binary_xyz,
            test_angle,
            proj_cfg["camera_distance_mm"],
            proj_cfg["fov_mm"],
            tuple(proj_cfg["detector_resolution"]),
            tissue_params,
            mcx_cfg["voxel_size_mm"],
        )

        mcx_n = mcx_proj_off / (mcx_proj_off.max() + 1e-10)
        green_n = green_proj_off / (green_proj_off.max() + 1e-10)
        ncc_off = compute_ncc(mcx_n, green_n)

        angle_results[f"offset_{offset}"] = {
            "angle": test_angle,
            "ncc": ncc_off,
        }

    return {
        "config_id": config_id,
        "source_pos": source_pos.tolist(),
        "best_angle": best_angle,
        "ncc_best": ncc,
        "rmse_best": rmse,
        "angle_sweep": angle_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-position × best-view test")
    parser.add_argument("--mcx", default="/mnt/f/win-pro/bin/mcx.exe")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/multiposition")
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

    # Get surface positions
    surface_pos = get_surface_positions(
        atlas_binary_xyz, config["mcx"]["voxel_size_mm"]
    )
    logger.info(f"Surface positions:")
    logger.info(f"  Dorsal Z: {surface_pos['dorsal_z']:.1f} mm")
    logger.info(f"  Ventral Z: {surface_pos['ventral_z']:.1f} mm")
    logger.info(f"  Left X: {surface_pos['left_x']:.1f} mm")
    logger.info(f"  Right X: {surface_pos['right_x']:.1f} mm")

    # Generate test configs
    test_configs = generate_test_configs(surface_pos, args.depth)

    output_base = Path(__file__).parent / args.output
    output_base.mkdir(parents=True, exist_ok=True)

    # Run all configs
    results = []
    for test_config in test_configs:
        try:
            result = run_single_config(
                test_config,
                config["tissue_params"],
                config["mcx"],
                config["projection"],
                atlas_binary_xyz,
                atlas_zyx,
                args.mcx,
                output_base,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed at {test_config['id']}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Multi-Position Test Summary")
    logger.info(f"{'=' * 60}")
    for r in results:
        logger.info(
            f"  {r['config_id']:20s}: NCC = {r['ncc_best']:.4f} @ {r['best_angle']:3.0f}°"
        )

    # Save results
    with open(output_base / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_base / 'summary.json'}")


if __name__ == "__main__":
    main()
