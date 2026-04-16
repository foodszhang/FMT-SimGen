#!/usr/bin/env python3
"""Stage 1.5 (v2): Point source with ATLAS shape + SURFACE-AWARE Green's function.

This version computes Green's function directly on the visible surface,
without generating an intermediate 3D volume.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import (
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
    if not atlas_bin_path.exists():
        raise FileNotFoundError(f"Atlas bin not found: {atlas_bin_path}")

    volume = np.fromfile(atlas_bin_path, dtype=np.uint8)
    volume = volume.reshape((104, 200, 190))
    # Convert ZYX to XYZ for surface projection
    volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ
    homogeneous = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    n_tissue = np.sum(homogeneous > 0)
    logger.info(f"Atlas homogeneous volume: {homogeneous.shape} (XYZ)")
    logger.info(
        f"  Tissue: {n_tissue}/{homogeneous.size} ({100 * n_tissue / homogeneous.size:.1f}%)"
    )

    return homogeneous


def generate_mcx_config(
    source_pos_mm: np.ndarray,
    atlas_volume_zyx: np.ndarray,
    voxel_size_mm: float,
    trunk_offset_mm: List[float],
    tissue_params: dict,
    n_photons: int,
    config_id: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate MCX JSON config for atlas-shaped volume.

    Args:
        source_pos_mm: Source position in CENTERED coordinates (mm)
                      Origin at volume center, X right, Y posterior, Z superior
        atlas_volume_zyx: Volume data in ZYX order
        voxel_size_mm: Voxel size in mm
        trunk_offset_mm: Offset from original atlas to trunk volume [x, y, z] (mm)
        tissue_params: Tissue optical parameters
        n_photons: Number of photons to simulate
        config_id: Configuration ID
        output_dir: Output directory

    Returns:
        (json_path, volume_bin_path)
    """
    nz, ny, nx = atlas_volume_zyx.shape

    # Convert centered coords to voxel indices (0-based, corner origin)
    # Centered: origin at volume center
    # MCX: origin at first voxel corner (0,0,0)
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

    logger.info(f"Generated MCX config: {json_path}")
    logger.info(
        f"  Source at voxel: [{source_x_vox:.1f}, {source_y_vox:.1f}, {source_z_vox:.1f}]"
    )
    return json_path, volume_bin_path


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
    try:
        subprocess.run(
            [mcx_exec, "-f", json_path.name],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MCX timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MCX failed: {e.stderr or e.stdout}")

    if not output_jnii.exists():
        raise FileNotFoundError(f"MCX output not found: {output_jnii}")

    logger.info(f"MCX complete: {output_jnii}")
    return output_jnii


def load_mcx_output(jnii_path: Path) -> np.ndarray:
    """Load MCX JNII output and return fluence volume in XYZ order."""
    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    fluence = np.array(data["NIFTIData"], dtype=np.float32)

    # Handle extra dimensions (time, wavelength)
    # MCX output is (X, Y, Z, time, wavelength) or similar
    if fluence.ndim > 3:
        # Take first time and wavelength slice
        fluence = fluence[..., 0, 0]

    # MCX JNII is already in XYZ order
    fluence_xyz = fluence
    logger.info(
        f"Loaded MCX fluence: {fluence_xyz.shape} (XYZ), max={fluence_xyz.max():.6e}"
    )
    return fluence_xyz


def project_mcx_to_detector(
    fluence_xyz: np.ndarray,
    angle_deg: float,
    camera_distance_mm: float,
    fov_mm: float,
    detector_resolution: Tuple[int, int],
    voxel_size_mm: float,
) -> np.ndarray:
    """Project MCX 3D fluence to 2D detector."""
    from surface_projection import rotation_matrix_y

    nx, ny, nz = fluence_xyz.shape
    width, height = detector_resolution

    # Get all non-zero voxels
    nonzero = np.argwhere(fluence_xyz > 0)
    if len(nonzero) == 0:
        logger.warning("No non-zero voxels in MCX output")
        return np.zeros((height, width), dtype=np.float32)

    # Voxel centers in mm (centered)
    center = np.array([nx / 2, ny / 2, nz / 2])
    coords_mm = (nonzero.astype(np.float32) - center + 0.5) * voxel_size_mm
    values = fluence_xyz[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]

    # Rotate
    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        coords_rot = coords_mm @ R.T
    else:
        coords_rot = coords_mm

    cam_x = coords_rot[:, 0]
    cam_y = coords_rot[:, 1]
    depths = camera_distance_mm - coords_rot[:, 2]

    # Project
    half_w = fov_mm / 2
    half_h = fov_mm / 2
    px_size_x = fov_mm / width
    px_size_y = fov_mm / height

    projection = np.zeros((height, width), dtype=np.float32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    # Use voxel coverage filling to avoid sampling gaps
    half_voxel = voxel_size_mm / 2

    for idx in range(len(cam_x)):
        px, py = cam_x[idx], cam_y[idx]
        d = depths[idx]

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
                    projection[pv, pu] = values[idx]

    return projection


def get_atlas_center(atlas_binary_xyz: np.ndarray, voxel_size_mm: float) -> tuple:
    """Find the center of mass and dorsal surface of the atlas volume.

    Returns:
        (center_x, center_y, dorsal_z) in mm (centered coordinates)
    """
    tissue_mask = atlas_binary_xyz > 0
    tissue_voxels = np.argwhere(tissue_mask)

    if len(tissue_voxels) == 0:
        raise ValueError("No tissue found in atlas")

    nx, ny, nz = atlas_binary_xyz.shape
    center = np.array([nx / 2, ny / 2, nz / 2])

    # Convert to mm
    tissue_mm = (tissue_voxels - center + 0.5) * voxel_size_mm

    # Center of mass
    center_x = float(tissue_mm[:, 0].mean())
    center_y = float(tissue_mm[:, 1].mean())

    # Dorsal surface (max Z)
    dorsal_z = float(tissue_mm[:, 2].max())

    return center_x, center_y, dorsal_z


def run_single_depth(
    depth_mm: float,
    config: dict,
    atlas_binary_xyz: np.ndarray,
    atlas_zyx: np.ndarray,
    mcx_exec: str,
    output_base: Path,
) -> Dict:
    """Run single depth configuration."""

    config_id = f"S1.5-D{int(depth_mm)}mm"
    output_dir = output_base / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get actual tissue center and dorsal surface from atlas volume
    mcx_cfg = config["mcx"]
    voxel_size = mcx_cfg["voxel_size_mm"]
    trunk_offset = mcx_cfg["trunk_offset_mm"]
    center_x, center_y, dorsal_z = get_atlas_center(atlas_binary_xyz, voxel_size)

    # Source position (centered on tissue, at specified depth from dorsal surface)
    source_z = dorsal_z - depth_mm
    source_pos_mm = np.array([center_x, center_y, source_z])

    # For MCX, add trunk_offset back
    source_xy_mcx = [center_x + trunk_offset[0], center_y + trunk_offset[1]]

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Config: {config_id}")
    logger.info(
        f"Source (Green): [{source_pos_mm[0]:.1f}, {source_pos_mm[1]:.1f}, {source_pos_mm[2]:.1f}] mm"
    )
    logger.info(
        f"Source (MCX): [{source_xy_mcx[0]:.1f}, {source_xy_mcx[1]:.1f}, {source_z:.1f}] mm"
    )
    logger.info(f"Depth from surface: {depth_mm} mm")

    tissue_params = config["tissue_params"]
    proj_cfg = config["projection"]

    # Generate MCX config and run
    # Pass source position in CENTERED coordinates (without trunk_offset)
    # generate_mcx_config will subtract trunk_offset internally
    json_path, _ = generate_mcx_config(
        source_pos_mm,  # [-1.0, 2.4, 8.1] - centered coords
        atlas_zyx,
        mcx_cfg["voxel_size_mm"],
        mcx_cfg["trunk_offset_mm"],
        tissue_params,
        mcx_cfg["n_photons"],
        config_id,
        output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec)
    fluence_xyz = load_mcx_output(jnii_path)

    # Run projections for each angle
    angles = proj_cfg["angles"]
    results = {
        "config_id": config_id,
        "depth_mm": depth_mm,
        "source_pos_mm": source_pos_mm.tolist(),
    }
    angle_results = {}

    for angle in angles:
        logger.info(f"  Processing angle {angle}°...")

        # MCX projection
        mcx_proj = project_mcx_to_detector(
            fluence_xyz,
            angle,
            proj_cfg["camera_distance_mm"],
            proj_cfg["fov_mm"],
            tuple(proj_cfg["detector_resolution"]),
            mcx_cfg["voxel_size_mm"],
        )

        # Green projection (surface-aware)
        green_proj = render_green_surface_projection(
            source_pos_mm,
            atlas_binary_xyz,
            angle,
            proj_cfg["camera_distance_mm"],
            proj_cfg["fov_mm"],
            tuple(proj_cfg["detector_resolution"]),
            tissue_params,
            mcx_cfg["voxel_size_mm"],
            green_type="infinite",
        )

        # Normalize both for comparison (peak normalization)
        mcx_norm = mcx_proj / (mcx_proj.max() + 1e-10)
        green_norm = green_proj / (green_proj.max() + 1e-10)

        # Compute metrics
        ncc = compute_ncc(mcx_norm, green_norm)
        rmse = compute_rmse(mcx_norm, green_norm)

        # Save projections
        np.save(output_dir / f"mcx_projection_a{angle}.npy", mcx_proj)
        np.save(output_dir / f"green_projection_a{angle}.npy", green_proj)

        angle_results[str(angle)] = {
            "ncc": ncc,
            "rmse": rmse,
            "mcx_peak": float(mcx_proj.max()),
            "green_peak": float(green_proj.max()),
            "peak_ratio": float(green_proj.max() / (mcx_proj.max() + 1e-10)),
        }

        logger.info(
            f"    NCC: {ncc:.4f}, RMSE: {rmse:.4f}, Peak ratio: {angle_results[str(angle)]['peak_ratio']:.3f}"
        )

    results["angles"] = angle_results
    results["mean_ncc"] = np.mean([a["ncc"] for a in angle_results.values()])
    results["min_ncc"] = np.min([a["ncc"] for a in angle_results.values()])
    results["max_ncc"] = np.max([a["ncc"] for a in angle_results.values()])

    logger.info(f"  Mean NCC: {results['mean_ncc']:.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.5: Surface-aware Green vs MCX"
    )
    parser.add_argument(
        "--mcx", default="/mnt/f/win-pro/bin/mcx.exe", help="MCX executable path"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument(
        "--output", default="results/stage1_5_surface", help="Output directory"
    )
    parser.add_argument(
        "--depths", type=float, nargs="+", default=None, help="Depths to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)

    # Load atlas volume
    atlas_bin_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    )
    atlas_binary_xyz = create_atlas_homogeneous_volume(atlas_bin_path)

    # Also need ZYX version for MCX
    atlas_zyx = atlas_binary_xyz.transpose(2, 1, 0).astype(np.uint8)

    # Determine depths to run
    depths = args.depths if args.depths else config["depths_mm"]

    # Output directory
    output_base = Path(__file__).parent / args.output
    output_base.mkdir(parents=True, exist_ok=True)

    # Run each depth
    all_results = {}
    for depth in depths:
        try:
            result = run_single_depth(
                depth,
                config,
                atlas_binary_xyz,
                atlas_zyx,
                args.mcx,
                output_base,
            )
            all_results[result["config_id"]] = result
        except Exception as e:
            logger.error(f"Failed at depth {depth}mm: {e}")
            import traceback

            traceback.print_exc()

    # Save summary
    with open(output_base / "stage1_5_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'=' * 50}")
    logger.info("Stage 1.5 Complete!")
    logger.info(f"Summary saved to: {output_base / 'stage1_5_summary.json'}")
    logger.info("\nResults:")
    for cid, res in all_results.items():
        logger.info(f"  {cid}: mean NCC = {res['mean_ncc']:.4f}")


if __name__ == "__main__":
    main()
