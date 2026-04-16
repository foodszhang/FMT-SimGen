#!/usr/bin/env python3
"""Stage 1: Point source × multiple depths - MCX vs Green comparison.

Runs real MCX simulations and compares with analytic Green's function.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
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


def create_homogeneous_mcx_volume(
    vol_shape_zyx: Tuple[int, int, int],
    tissue_params: dict,
) -> np.ndarray:
    """Create homogeneous tissue volume for MCX.

    Returns uint8 volume filled with label 1 (tissue).
    """
    volume = np.ones(vol_shape_zyx, dtype=np.uint8)
    return volume


def generate_mcx_config(
    source_pos_mm: np.ndarray,
    vol_shape_zyx: Tuple[int, int, int],
    voxel_size_mm: float,
    trunk_offset_mm: List[float],
    tissue_params: dict,
    n_photons: int,
    config_id: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate MCX JSON config and volume file for point source."""

    nz, ny, nx = vol_shape_zyx

    source_x_vox = (source_pos_mm[0] - trunk_offset_mm[0]) / voxel_size_mm
    source_y_vox = (source_pos_mm[1] - trunk_offset_mm[1]) / voxel_size_mm
    source_z_vox = (source_pos_mm[2] - trunk_offset_mm[2]) / voxel_size_mm

    volume = create_homogeneous_mcx_volume(vol_shape_zyx, tissue_params)
    volume_bin_path = output_dir / "volume.bin"
    volume.tofile(volume_bin_path)

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
        f"  Source (voxel): [{source_x_vox:.1f}, {source_y_vox:.1f}, {source_z_vox:.1f}]"
    )
    logger.info(f"  Volume: {nx}x{ny}x{nz} voxels")

    return json_path, volume_bin_path


def run_mcx_simulation(
    json_path: Path, mcx_exec: str = "mcx.exe", timeout: int = 600
) -> Path:
    """Run MCX simulation and return output path."""

    work_dir = json_path.parent
    session_id = json_path.stem
    output_jnii = work_dir / f"{session_id}.jnii"

    if output_jnii.exists():
        logger.info(f"Skipping MCX: {output_jnii} already exists")
        return output_jnii

    logger.info(f"Running MCX: {mcx_exec} -f {json_path.name}")
    try:
        result = subprocess.run(
            [mcx_exec, "-f", json_path.name],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stderr:
            logger.debug(f"MCX stderr: {result.stderr[:500]}")
        if result.stdout:
            logger.debug(f"MCX stdout: {result.stdout[:500]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MCX timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MCX failed: {e.stderr or e.stdout}")

    if not output_jnii.exists():
        raise RuntimeError(f"MCX output not found: {output_jnii}")

    logger.info(f"MCX completed: {output_jnii}")
    return output_jnii


def load_mcx_fluence(jnii_path: Path) -> np.ndarray:
    """Load MCX fluence volume.

    MCX output is already in (nx, ny, nz) = XYZ order.
    Just squeeze extra dimensions.
    """

    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    if isinstance(data, dict):
        fluence = data.get("NIFTIData", data.get("data", None))
        if fluence is None:
            raise KeyError(f"No fluence data in {jnii_path}")
    else:
        fluence = data

    fluence = np.asarray(fluence, dtype=np.float32)

    while fluence.ndim > 3:
        fluence = fluence.squeeze(axis=-1)

    logger.info(f"Loaded MCX fluence: {fluence.shape} (XYZ order)")
    return fluence


def run_single_depth(
    depth_mm: float,
    config: dict,
    output_dir: Path,
    mcx_exec: str,
) -> Dict:
    """Run MCX and Green comparison for a single depth."""

    dorsal_z = config["dorsal_z_mm"]
    source_xy = np.array(config["source_xy"])
    source_z = dorsal_z - depth_mm
    source_pos_mm = np.array([source_xy[0], source_xy[1], source_z])

    config_id = f"S1-D{depth_mm:.0f}mm"

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running {config_id}: source Z = {source_z:.1f} mm")
    logger.info(f"{'=' * 60}")

    mcx_cfg = config["mcx"]
    vol_shape_zyx = tuple(mcx_cfg["volume_shape_zyx"])
    voxel_size = mcx_cfg["voxel_size_mm"]
    trunk_offset = np.array(mcx_cfg["trunk_offset_mm"])

    nz, ny, nx = vol_shape_zyx
    vol_shape_xyz = (nx, ny, nz)

    sample_dir = output_dir / config_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    json_path, _ = generate_mcx_config(
        source_pos_mm=source_pos_mm,
        vol_shape_zyx=vol_shape_zyx,
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

    logger.info(
        f"MCX fluence: sum={fluence_mcx_xyz.sum():.2e}, max={fluence_mcx_xyz.max():.2e}"
    )
    logger.info(
        f"Green fluence: sum={fluence_green_xyz.sum():.2e}, max={fluence_green_xyz.max():.2e}"
    )

    fluence_mcx_norm = fluence_mcx_xyz / fluence_mcx_xyz.max()
    fluence_green_norm = fluence_green_xyz / fluence_green_xyz.max()

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
            fluence_mcx_norm,
            angle_deg=float(angle),
            camera_distance=camera_dist,
            fov_mm=fov_mm,
            detector_resolution=detector_res,
            voxel_size_mm=voxel_size,
        )

        proj_green, _ = project_volume_reference(
            fluence_green_norm,
            angle_deg=float(angle),
            camera_distance=camera_dist,
            fov_mm=fov_mm,
            detector_resolution=detector_res,
            voxel_size_mm=voxel_size,
        )

        proj_mcx_norm = proj_mcx / (proj_mcx.max() + 1e-10)
        proj_green_norm = proj_green / (proj_green.max() + 1e-10)

        ncc = compute_ncc(proj_mcx_norm, proj_green_norm)
        rmse = compute_rmse(proj_mcx_norm, proj_green_norm)

        results["angles"][str(angle)] = {
            "ncc": float(ncc),
            "rmse": float(rmse),
        }

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
    logger.info(f"Min NCC:  {results['min_ncc']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Point source depth comparison"
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent / "results" / "stage1",
        help="Output directory",
    )
    parser.add_argument(
        "--mcx",
        default="mcx.exe",
        help="MCX executable (default: mcx.exe)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        help="Override depths (default from config)",
    )

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(Path(args.config))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    depths = args.depths if args.depths else config["depths_mm"]

    all_results = {}

    for depth in depths:
        try:
            result = run_single_depth(
                depth_mm=depth,
                config=config,
                output_dir=output_dir,
                mcx_exec=args.mcx,
            )
            all_results[result["config_id"]] = result
        except Exception as e:
            logger.error(f"Failed depth {depth}mm: {e}")
            all_results[f"S1-D{depth:.0f}mm"] = {"error": str(e)}

    summary_path = output_dir / "stage1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("Stage 1 Summary")
    print("=" * 60)
    for cid, res in all_results.items():
        if "error" in res:
            print(f"  {cid}: ERROR - {res['error']}")
        else:
            print(
                f"  {cid}: mean NCC = {res['mean_ncc']:.4f} (min = {res['min_ncc']:.4f})"
            )

    thresholds = config.get("thresholds", {})
    ncc_go = thresholds.get("ncc_go", 0.95)
    ncc_caution = thresholds.get("ncc_caution", 0.85)

    go_depths = [
        d
        for d in depths
        if f"S1-D{d:.0f}mm" in all_results
        and all_results[f"S1-D{d:.0f}mm"].get("mean_ncc", 0) >= ncc_go
    ]

    print(f"\nDepths with NCC >= {ncc_go} (GO for Stage 2): {go_depths}")

    if go_depths:
        best_depth = max(go_depths)
        print(f"Recommended depth for Stage 2: {best_depth} mm")


if __name__ == "__main__":
    main()
