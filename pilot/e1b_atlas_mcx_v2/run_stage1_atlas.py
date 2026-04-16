#!/usr/bin/env python3
"""Stage 1.5: Point source with ATLAS-SHAPED volume (homogeneous tissue)."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmt_simgen.mcx_projection import project_volume_reference
from generate_analytic_fluence import (
    green_infinite_point_source,
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
    homogeneous = np.where(volume > 0, 1, 0).astype(np.uint8)

    n_tissue = np.sum(homogeneous > 0)
    logger.info(f"Atlas homogeneous volume: {homogeneous.shape}")
    logger.info(f"  Tissue: {n_tissue}/{homogeneous.size} ({100*n_tissue/homogeneous.size:.1f}%)")

    return homogeneous


def generate_mcx_config(
    source_pos_mm: np.ndarray,
    atlas_volume: np.ndarray,
    voxel_size_mm: float,
    trunk_offset_mm: List[float],
    tissue_params: dict,
    n_photons: int,
    config_id: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate MCX JSON config for atlas-shaped volume."""

    nz, ny, nx = atlas_volume.shape

    source_x_vox = (source_pos_mm[0] - trunk_offset_mm[0]) / voxel_size_mm
    source_y_vox = (source_pos_mm[1] - trunk_offset_mm[1]) / voxel_size_mm
    source_z_vox = (source_pos_mm[2] - trunk_offset_mm[2]) / voxel_size_mm

    volume_bin_path = output_dir / "volume.bin"
    atlas_volume.tofile(volume_bin_path)

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
    return json_path, volume_bin_path


def run_mcx_simulation(json_path: Path, mcx_exec: str = "mcx.exe", timeout: int = 600) -> Path:
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
        raise RuntimeError(f"MCX output not found: {output_jnii}")

    logger.info(f"MCX completed: {output_jnii}")
    return output_jnii


def load_mcx_fluence(jnii_path: Path) -> np.ndarray:
    import jdata as jd
    data = jd.loadjd(str(jnii_path))
    if isinstance(data, dict):
        fluence = data.get("NIFTIData", data.get("data", None))
    else:
        fluence = data
    fluence = np.asarray(fluence, dtype=np.float32)
    while fluence.ndim > 3:
        fluence = fluence.squeeze(axis=-1)
    return fluence


def run_single_depth(depth_mm: float, config: dict, atlas_volume: np.ndarray,
                     output_dir: Path, mcx_exec: str) -> Dict:
    dorsal_z = config["dorsal_z_mm"]
    source_xy = np.array(config["source_xy"])
    source_z = dorsal_z - depth_mm
    source_pos_mm = np.array([source_xy[0], source_xy[1], source_z])

    config_id = f"S1A-D{depth_mm:.0f}mm"

    logger.info(f"\n{'='*60}")
    logger.info(f"Running {config_id}: source Z = {source_z:.1f} mm")
    logger.info(f"{'='*60}")

    mcx_cfg = config["mcx"]
    voxel_size = mcx_cfg["voxel_size_mm"]
    trunk_offset = np.array(mcx_cfg["trunk_offset_mm"])

    nz, ny, nx = atlas_volume.shape
    vol_shape_xyz = (nx, ny, nz)

    sample_dir = output_dir / config_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    json_path, _ = generate_mcx_config(
        source_pos_mm=source_pos_mm,
        atlas_volume=atlas_volume,
        voxel_size_mm=voxel_size,
        trunk_offset_mm=mcx_cfg["trunk_offset_mm"],
        tissue_params=config["tissue_params"],
        n_photons=mcx_cfg["n_photons"],
        config_id=config_id,
        output_dir=sample_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec=mcx_exec)
    fluence_mcx_xyz = load_mcx_fluence(jnii_path)

    origin_mm = trunk_offset
    fluence_green_xyz = green_infinite_point_source(
        source_pos_mm=source_pos_mm,
        volume_shape_xyz=vol_shape_xyz,
        voxel_size_mm=voxel_size,
        origin_mm=origin_mm,
        tissue_params=config["tissue_params"],
    )

    proj_cfg = config["projection"]
    angles = proj_cfg["angles"]
    detector_res = tuple(proj_cfg["detector_resolution"])
    fov_mm = proj_cfg["fov_mm"]
    camera_dist = proj_cfg["camera_distance_mm"]

    results = {
        "config_id": config_id,
        "depth_mm": depth_mm,
        "source_z_mm": source_z,
        "source_pos_mm": source_pos_mm.tolist(),
        "angles": {},
    }

    for angle in angles:
        proj_mcx, _ = project_volume_reference(
            fluence_mcx_xyz, float(angle), camera_dist, fov_mm, detector_res, voxel_size
        )
        proj_green, _ = project_volume_reference(
            fluence_green_xyz, float(angle), camera_dist, fov_mm, detector_res, voxel_size
        )

        proj_mcx_norm = proj_mcx / (proj_mcx.max() + 1e-10)
        proj_green_norm = proj_green / (proj_green.max() + 1e-10)

        ncc = compute_ncc(proj_mcx_norm, proj_green_norm)
        rmse = compute_rmse(proj_mcx_norm, proj_green_norm)

        results["angles"][str(angle)] = {"ncc": float(ncc), "rmse": float(rmse)}
        logger.info(f"  Angle {angle:+3d}°: NCC = {ncc:.4f}, RMSE = {rmse:.4f}")

    nccs = [v["ncc"] for v in results["angles"].values()]
    results["mean_ncc"] = float(np.mean(nccs))
    results["min_ncc"] = float(np.min(nccs))
    results["max_ncc"] = float(np.max(nccs))

    np.savez(
        sample_dir / "comparison.npz",
        fluence_mcx=fluence_mcx_xyz,
        fluence_green=fluence_green_xyz,
        source_pos_mm=source_pos_mm,
        depth_mm=depth_mm,
    )

    logger.info(f"\nMean NCC: {results['mean_ncc']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1.5: Atlas-shaped volume")
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--output", default=Path(__file__).parent / "results" / "stage1_atlas")
    parser.add_argument("--atlas-bin", default=Path(__file__).parent.parent.parent / "output" / "shared" / "mcx_volume_trunk.bin")
    parser.add_argument("--mcx", default="mcx.exe")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--depths", type=float, nargs="+")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(Path(args.config))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading atlas-shaped volume...")
    atlas_volume = create_atlas_homogeneous_volume(Path(args.atlas_bin))

    depths = args.depths if args.depths else config["depths_mm"]
    all_results = {}

    for depth in depths:
        try:
            result = run_single_depth(depth, config, atlas_volume, output_dir, args.mcx)
            all_results[result["config_id"]] = result
        except Exception as e:
            logger.error(f"Failed depth {depth}mm: {e}")
            all_results[f"S1A-D{depth:.0f}mm"] = {"error": str(e)}

    summary_path = output_dir / "stage1_atlas_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("Stage 1.5 (Atlas-Shaped) Summary")
    print("="*60)
    for cid, res in all_results.items():
        if "error" in res:
            print(f"  {cid}: ERROR - {res['error']}")
        else:
            print(f"  {cid}: mean NCC = {res['mean_ncc']:.4f}")


if __name__ == "__main__":
    main()
