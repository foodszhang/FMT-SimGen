"""M3: Three-source multi-view validation.

Validates ball and Gaussian sources across 4 camera angles (0°, 90°, -90°, 180°).

Usage:
    uv run python m3_multi_view.py --n-photons 10000000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "e1b_atlas_mcx_v2"))

from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)
from source_quadrature import sample_uniform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOXEL_SIZE_MM = 0.4
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)

SOURCE_POS_PHYSICAL = np.array([-0.6, 2.4, 5.8])
SOURCE_RADIUS = 2.0
SOURCE_SIGMA = 1.0

VIEW_ANGLES = [0, 90, -90, 180]

TISSUE_PARAMS = {"mua_mm": 0.08697, "mus_prime_mm": 4.2907}

ARCHIVED_MEDIA = [
    {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0, "tag": 0, "name": "background"},
    {
        "mua": 0.08697,
        "mus": 42.9071,
        "g": 0.9,
        "n": 1.37,
        "tag": 1,
        "name": "soft_tissue",
    },
    {"mua": 0.04, "mus": 200.0, "g": 0.9, "n": 1.37, "tag": 2, "name": "bone"},
    {"mua": 0.0648, "mus": 23.92, "g": 0.9, "n": 1.37, "tag": 3, "name": "brain"},
    {
        "mua": 0.05881,
        "mus": 42.83873333333333,
        "g": 0.85,
        "n": 1.37,
        "tag": 4,
        "name": "heart",
    },
    {
        "mua": 0.01304,
        "mus": 224.51875,
        "g": 0.92,
        "n": 1.37,
        "tag": 5,
        "name": "stomach",
    },
    {
        "mua": 0.06597,
        "mus": 107.2862,
        "g": 0.85,
        "n": 1.37,
        "tag": 6,
        "name": "abdominal",
    },
    {"mua": 0.35182, "mus": 67.8066, "g": 0.9, "n": 1.37, "tag": 7, "name": "liver"},
    {"mua": 0.06597, "mus": 107.2862, "g": 0.85, "n": 1.37, "tag": 8, "name": "kidney"},
    {
        "mua": 0.19639,
        "mus": 608.6888333333333,
        "g": 0.94,
        "n": 1.37,
        "tag": 9,
        "name": "lung",
    },
]


def load_volume() -> np.ndarray:
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    return volume.reshape(VOLUME_SHAPE_XYZ)


def physical_to_voxel(pos_physical: np.ndarray) -> np.ndarray:
    center = np.array(VOLUME_SHAPE_XYZ) / 2
    return pos_physical / VOXEL_SIZE_MM + center - 0.5


def compute_green_on_fluence_surface(
    tissue_mask: np.ndarray,
    source_pos_physical: np.ndarray,
    source_radius: float,
    angle_deg: float,
    scheme: str = "7-point",
) -> np.ndarray:
    surface_coords, valid_mask = project_get_surface_coords(
        tissue_mask,
        angle_deg,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    axes = np.array([source_radius, source_radius, source_radius])
    points, weights = sample_uniform(
        center=source_pos_physical, axes=axes, alpha=1.0, scheme=scheme
    )
    green_proj = np.zeros(DETECTOR_RESOLUTION[::-1], dtype=np.float32)
    for pt, w in zip(points, weights):
        proj_i = green_infinite_point_source_on_surface(
            pt, surface_coords, valid_mask, TISSUE_PARAMS
        )
        green_proj += w * proj_i

    return green_proj


def compute_gaussian_green_on_fluence_surface(
    tissue_mask: np.ndarray,
    source_pos_physical: np.ndarray,
    sigma: float,
    angle_deg: float,
    n_samples: int = 27,
) -> np.ndarray:
    surface_coords, valid_mask = project_get_surface_coords(
        tissue_mask,
        angle_deg,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    np.random.seed(42)
    samples = np.random.randn(n_samples, 3) * sigma
    weights = np.ones(n_samples) / n_samples

    green_proj = np.zeros(DETECTOR_RESOLUTION[::-1], dtype=np.float32)
    for s, w in zip(samples, weights):
        pt = source_pos_physical + s
        proj_i = green_infinite_point_source_on_surface(
            pt, surface_coords, valid_mask, TISSUE_PARAMS
        )
        green_proj += w * proj_i

    return green_proj


def compute_gaussian_green_on_fluence_surface(
    fluence: np.ndarray,
    source_pos_physical: np.ndarray,
    sigma: float,
    angle_deg: float,
    n_samples: int = 27,
) -> np.ndarray:
    fluence_threshold = fluence.max() * 1e-6
    fluence_mask = fluence > fluence_threshold

    surface_coords, valid_mask = project_get_surface_coords(
        fluence_mask,
        angle_deg,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    np.random.seed(42)
    samples = np.random.randn(n_samples, 3) * sigma
    weights = np.ones(n_samples) / n_samples

    green_proj = np.zeros(DETECTOR_RESOLUTION[::-1], dtype=np.float32)
    for s, w in zip(samples, weights):
        pt = source_pos_physical + s
        proj_i = green_infinite_point_source_on_surface(
            pt, surface_coords, valid_mask, TISSUE_PARAMS
        )
        green_proj += w * proj_i

    return green_proj


def run_mcx_for_source(
    source_kind: str,
    source_pos_physical: np.ndarray,
    source_param: float,
    n_photons: int,
    output_dir: Path,
) -> np.ndarray:
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    source_voxel = physical_to_voxel(source_pos_physical)

    if source_kind == "ball":
        pattern_size = int(np.ceil(source_param * 2 / VOXEL_SIZE_MM)) + 4
        pattern_size = max(pattern_size, 11) | 1
        half = pattern_size // 2
        x = (np.arange(pattern_size) - half) * VOXEL_SIZE_MM
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        pattern = (r <= source_param).astype(np.float32)
        pattern = pattern / pattern.sum()
        pattern_shape = (pattern_size, pattern_size, pattern_size)
    elif source_kind == "gaussian":
        pattern_size = int(np.ceil(source_param * 6 / VOXEL_SIZE_MM)) | 1
        pattern_size = max(pattern_size, 11)
        half = pattern_size // 2
        x = (np.arange(pattern_size) - half) * VOXEL_SIZE_MM
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r2 = X**2 + Y**2 + Z**2
        pattern = np.exp(-r2 / (2 * source_param**2))
        pattern = pattern / pattern.sum()
        pattern_shape = (pattern_size, pattern_size, pattern_size)
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")

    origin = np.round(source_voxel - np.array(pattern_shape) / 2).astype(int)

    pattern_path = output_dir / f"source_{source_kind}.bin"
    pattern_zyx = pattern.transpose(2, 1, 0)
    pattern_zyx.astype(np.float32).tofile(pattern_path)

    config = {
        "Domain": {
            "VolumeFile": str(VOLUME_PATH.resolve()),
            "Dim": [int(x) for x in VOLUME_SHAPE_XYZ],
            "OriginType": 1,
            "LengthUnit": float(VOXEL_SIZE_MM),
            "Media": ARCHIVED_MEDIA,
        },
        "Session": {
            "Photons": n_photons,
            "RNGSeed": 42,
            "ID": f"m3_{source_kind}",
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5e-8,
            "DT": 5e-8,
        },
        "Optode": {
            "Source": {
                "Pos": [int(x) for x in origin],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {
                    "Nx": int(pattern_shape[0]),
                    "Ny": int(pattern_shape[1]),
                    "Nz": int(pattern_shape[2]),
                    "Data": pattern_path.name,
                },
                "Param1": [int(x) for x in pattern_shape],
            }
        },
    }

    json_path = output_dir / f"m3_{source_kind}.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Running MCX for {source_kind} source...")
    result = subprocess.run(
        ["/mnt/f/win-pro/bin/mcx.exe", "-f", json_path.name],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        logger.error(f"MCX failed: {result.stderr}")
        raise RuntimeError(f"MCX failed for {source_kind}")

    import jdata as jd

    jnii_path = output_dir / f"m3_{source_kind}.jnii"
    data = jd.loadjd(str(jnii_path))
    nifti = data["NIFTIData"] if isinstance(data, dict) else data
    if nifti.ndim == 5:
        nifti = nifti[:, :, :, 0, 0]

    logger.info(f"MCX done: fluence shape {nifti.shape}, non-zero {np.sum(nifti > 0)}")
    return nifti


def compute_metrics(mcx_proj: np.ndarray, green_proj: np.ndarray) -> dict:
    valid = (mcx_proj > 0) & (green_proj > 0)

    if np.sum(valid) < 10:
        return {"ncc": 0.0, "k": 0.0, "n_valid": 0}

    mcx_vals = mcx_proj[valid]
    green_vals = green_proj[valid]

    ncc = np.corrcoef(np.log10(mcx_vals + 1e-10), np.log10(green_vals + 1e-10))[0, 1]
    k = mcx_vals.sum() / green_vals.sum()

    return {
        "ncc": float(ncc),
        "k": float(k),
        "n_valid": int(np.sum(valid)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="M3: Multi-view validation for ball and Gaussian sources"
    )
    parser.add_argument("--n-photons", type=int, default=10_000_000, help="MCX photons")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = (
        Path(args.output)
        if args.output
        else Path("pilot/paper04b_forward/mvp_pipeline/results/m3")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "source_pos_physical": SOURCE_POS_PHYSICAL.tolist(),
        "voxel_size_mm": VOXEL_SIZE_MM,
        "n_photons": args.n_photons,
        "view_angles": VIEW_ANGLES,
        "sources": {},
    }

    tissue_mask = load_volume() > 0

    source_configs = [
        ("ball", SOURCE_RADIUS),
        ("gaussian", SOURCE_SIGMA),
    ]

    for kind, param in source_configs:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Testing {kind} source")
        logger.info(f"{'=' * 50}")

        source_dir = output_dir / kind
        source_dir.mkdir(parents=True, exist_ok=True)

        fluence = run_mcx_for_source(
            kind, SOURCE_POS_PHYSICAL, param, args.n_photons, source_dir
        )

        results["sources"][kind] = {"param": param, "views": {}}

        for angle in VIEW_ANGLES:
            logger.info(f"  Angle {angle}°...")

            if kind == "ball":
                green_proj = compute_green_on_fluence_surface(
                    tissue_mask, SOURCE_POS_PHYSICAL, param, angle
                )
            else:
                green_proj = compute_gaussian_green_on_fluence_surface(
                    tissue_mask, SOURCE_POS_PHYSICAL, param, angle
                )

            from fmt_simgen.mcx_projection import project_volume_reference

            mcx_proj, _ = project_volume_reference(
                fluence,
                angle,
                CAMERA_DISTANCE_MM,
                FOV_MM,
                DETECTOR_RESOLUTION,
                VOXEL_SIZE_MM,
            )

            metrics = compute_metrics(mcx_proj, green_proj)
            results["sources"][kind]["views"][str(angle)] = metrics

            logger.info(f"    NCC: {metrics['ncc']:.4f}, k: {metrics['k']:.2e}")

            np.save(source_dir / f"mcx_proj_a{angle}.npy", mcx_proj)
            np.save(source_dir / f"green_proj_a{angle}.npy", green_proj)

    with open(output_dir / "m3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("M3 Results Summary")
    print(f"{'=' * 70}")
    print(f"{'Source':<10} {'Angle':<8} {'NCC':<10} {'k':<15} {'Pass'}")
    print("-" * 70)

    all_pass = True
    for kind, data in results["sources"].items():
        for angle_str, metrics in data["views"].items():
            passed = metrics["ncc"] >= 0.90
            if not passed:
                all_pass = False
            print(
                f"{kind:<10} {angle_str + '°':<8} {metrics['ncc']:<10.4f} {metrics['k']:<15.2e} {'✓' if passed else '✗'}"
            )
    print(f"{'=' * 70}")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    logger.info(f"Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
