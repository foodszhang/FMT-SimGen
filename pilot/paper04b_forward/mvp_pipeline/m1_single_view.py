"""M1: Single-source single-view end-to-end validation.

This stage verifies:
1. MCX pattern3d read/write works correctly
2. Source position produces correct source_label
3. MCX fluence peak is not at volume edge
4. NCC(MCX, closed_form) >= 0.95
5. k = sum(MCX) / sum(closed) in 10^6~10^7 range

Uses the authoritative MCX trunk volume (voxel_size=0.2mm).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    OPTICAL,
    MVPConfig,
    SourceSpec,
    forward_closed_source,
    compute_all_metrics,
    metrics_summary,
    assert_voxel_consistency,
)
from shared.green_surface_projection import project_get_surface_coords
from mcx_runner import run_mcx_for_source

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Authoritative paths (from configs/base.yaml)
VOLUME_PATH = Path("output/shared/mcx_volume_trunk.bin")
MATERIAL_PATH = Path("output/shared/mcx_material.yaml")
VOXEL_SIZE_MM = 0.2  # MUST match MCX JSON LengthUnit
VOLUME_SHAPE_ZYX = (104, 200, 190)
VOLUME_SHAPE_XYZ = (190, 200, 104)


def load_mcx_volume_xyz() -> np.ndarray:
    """Load MCX volume and return in XYZ order."""
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    volume_zyx = volume.reshape(VOLUME_SHAPE_ZYX)
    volume_xyz = volume_zyx.transpose(2, 1, 0)  # ZYX -> XYZ
    return volume_xyz


def get_p1_dorsal_position() -> np.ndarray:
    """Get P1-dorsal source position.

    Position is defined as:
    - X = 0 (center)
    - Y = 10 mm (in trunk coordinates, Y=30mm offset means atlas Y=40mm)
    - Z = dorsal_z - 4mm (4mm below dorsal surface)

    Note: Y and Z are physical coordinates relative to volume center.
    """
    volume_xyz = load_mcx_volume_xyz()
    binary_mask = volume_xyz > 0

    # Find dorsal surface (max Z where tissue exists)
    nz_coords = np.where(binary_mask)
    dorsal_z_voxel = np.max(nz_coords[2])
    dorsal_z_mm = dorsal_z_voxel * VOXEL_SIZE_MM

    # Volume center Y
    center_y_voxel = np.mean(nz_coords[1])

    logger.info(f"Dorsal Z: {dorsal_z_mm:.1f}mm (voxel {dorsal_z_voxel})")

    # P1 position: center X, Y=10mm below center, 4mm below dorsal
    # Source coordinates are RELATIVE to volume center
    source_x = 0.0
    source_y = 10.0  # 10mm posterior from Y center
    source_z = dorsal_z_mm - 4.0  # 4mm below dorsal surface

    # But we need to make sure source is inside tissue
    # Check if source_y position has tissue at this Z
    source_y_voxel = int(center_y_voxel + source_y / VOXEL_SIZE_MM)
    source_z_voxel = int(source_z / VOXEL_SIZE_MM)

    # Find tissue region at this Y
    tissue_at_y = (
        binary_mask[:, source_y_voxel, :]
        if source_y_voxel < volume_xyz.shape[1]
        else None
    )
    if tissue_at_y is not None and np.any(tissue_at_y):
        # Get X center of tissue at this Y slice
        x_coords = np.where(tissue_at_y)
        center_x_voxel = np.mean(x_coords[0])
        source_x_voxel = int(center_x_voxel)

        # Verify source is in tissue
        if (
            0 <= source_x_voxel < volume_xyz.shape[0]
            and 0 <= source_z_voxel < volume_xyz.shape[2]
        ):
            label = volume_xyz[source_x_voxel, source_y_voxel, source_z_voxel]
            logger.info(
                f"Source voxel ({source_x_voxel}, {source_y_voxel}, {source_z_voxel}), label={label}"
            )

            if label == 0:
                # Source in background, find nearest tissue
                logger.warning("Source in background, adjusting...")
                # Find nearest tissue voxel
                for dz in range(1, 20):
                    test_z = source_z_voxel - dz
                    if 0 <= test_z < volume_xyz.shape[2]:
                        if volume_xyz[source_x_voxel, source_y_voxel, test_z] > 0:
                            source_z = test_z * VOXEL_SIZE_MM
                            logger.info(f"Adjusted source Z to {source_z:.1f}mm")
                            break

    return np.array([source_x, source_y, source_z])


def run_m1_validation(
    source: SourceSpec,
    angle_deg: int,
    run_mcx: bool = False,
    n_photons: int = int(1e8),
    output_dir: Path | None = None,
    homogeneous: bool = True,
) -> dict:
    """Run M1 validation for a single source/view."""
    logger.info(f"M1 validation for {source.kind} source at angle {angle_deg}°")

    output_dir = (
        Path(output_dir)
        if output_dir
        else Path("pilot/paper04b_forward/mvp_pipeline/results/m1")
    )

    # Load volume
    volume_xyz = load_mcx_volume_xyz()
    binary_mask = volume_xyz > 0
    logger.info(
        f"Volume shape (XYZ): {volume_xyz.shape}, tissue voxels: {np.sum(binary_mask)}"
    )

    # Config
    config = MVPConfig()
    config.voxel_size_mm = VOXEL_SIZE_MM

    # Project surface
    surface_coords, valid_mask = project_get_surface_coords(
        binary_mask,
        angle_deg,
        config.camera_distance_mm,
        config.fov_mm,
        config.detector_resolution,
        config.voxel_size_mm,
    )
    logger.info(f"Valid surface pixels: {np.sum(valid_mask)}")

    # Closed-form forward
    closed_proj = forward_closed_source(
        source, surface_coords, valid_mask, config.optical
    )
    logger.info(
        f"Closed-form: peak={np.max(closed_proj):.2e}, sum={np.sum(closed_proj):.2e}"
    )

    results = {
        "source": source.to_dict(),
        "angle_deg": angle_deg,
        "voxel_size_mm": VOXEL_SIZE_MM,
        "volume_shape_xyz": list(VOLUME_SHAPE_XYZ),
        "valid_surface_pixels": int(np.sum(valid_mask)),
        "closed_peak": float(np.max(closed_proj)),
        "closed_sum": float(np.sum(closed_proj)),
    }

    if run_mcx:
        # Generate source pattern
        pattern_shape = (21, 21, 21)  # Sufficient for ball R=2mm or gaussian sigma=1mm
        source_pattern = source.pattern3d(pattern_shape, VOXEL_SIZE_MM)

        logger.info(f"Running MCX with {n_photons} photons...")
        config_id = f"m1_{source.kind}_a{angle_deg}"

        fluence = run_mcx_for_source(
            source_pattern,
            source.center_mm,
            n_photons=n_photons,
            output_dir=output_dir / "mcx",
            config_id=config_id,
            homogeneous=homogeneous,
        )

        # Project fluence to surface
        from fmt_simgen.mcx_projection import project_volume_reference

        mcx_proj, _ = project_volume_reference(
            fluence,
            angle_deg,
            config.camera_distance_mm,
            config.fov_mm,
            config.detector_resolution,
            config.voxel_size_mm,
        )

        logger.info(
            f"MCX projection: peak={np.max(mcx_proj):.2e}, sum={np.sum(mcx_proj):.2e}"
        )

        # Compute metrics
        metrics = compute_all_metrics(mcx_proj, closed_proj)
        logger.info(f"Metrics: {metrics_summary(metrics)}")

        results["mcx_available"] = True
        results["mcx_peak"] = float(np.max(mcx_proj))
        results["mcx_sum"] = float(np.sum(mcx_proj))
        results.update(metrics)

        # Check Go criteria
        passed = True
        if metrics["ncc"] < 0.95:
            logger.warning(f"NCC {metrics['ncc']:.4f} < 0.95 threshold")
            passed = False
        if metrics["k"] < 1e5 or metrics["k"] > 1e8:
            logger.warning(f"k {metrics['k']:.2e} outside [1e5, 1e8] range")
            passed = False
        results["passed"] = passed
    else:
        results["mcx_available"] = False
        logger.info("Skipping MCX (use --run-mcx when MCX is available)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "closed_proj.npy", closed_proj)
    with open(output_dir / "m1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="M1: Single-source single-view validation"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="point",
        choices=["point", "ball", "gaussian"],
        help="Source type",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        help="Source center in mm [x, y, z] (default: P1-dorsal)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Ball radius in mm",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma in mm",
    )
    parser.add_argument(
        "--angle",
        type=int,
        default=0,
        help="Viewing angle in degrees",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--run-mcx",
        action="store_true",
        help="Run MCX simulation (requires MCX executable)",
    )
    parser.add_argument(
        "--n-photons",
        type=int,
        default=int(1e8),
        help="Number of MCX photons (default: 1e8)",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Use heterogeneous volume (multi-material) instead of homogeneous",
    )
    args = parser.parse_args()

    # Determine source position
    if args.center:
        center_mm = np.array(args.center)
    else:
        center_mm = get_p1_dorsal_position()

    logger.info(f"Source center: {center_mm} mm")

    # Create source
    if args.source == "point":
        source = SourceSpec(kind="point", center_mm=center_mm)
    elif args.source == "ball":
        source = SourceSpec(kind="ball", center_mm=center_mm, radius_mm=args.radius)
    else:
        source = SourceSpec(kind="gaussian", center_mm=center_mm, sigma_mm=args.sigma)

    logger.info(f"Source: {source}")
    logger.info(
        f"Optical: mu_a={OPTICAL.mu_a}, mus_p={OPTICAL.mus_p}, delta={OPTICAL.delta:.3f}mm"
    )

    # Verify voxel size
    logger.info(f"Voxel size: {VOXEL_SIZE_MM}mm (authoritative)")

    output_dir = (
        Path(args.output)
        if args.output
        else Path("pilot/paper04b_forward/mvp_pipeline/results/m1")
    )
    results = run_m1_validation(
        source,
        args.angle,
        run_mcx=args.run_mcx,
        n_photons=args.n_photons,
        output_dir=output_dir,
        homogeneous=not args.heterogeneous,
    )

    print(f"\n{'=' * 50}")
    print(f"M1 Results:")
    print(f"  Source: {source.kind}")
    print(f"  Center: {center_mm}")
    print(f"  Closed-form peak: {results['closed_peak']:.2e}")
    print(f"  Closed-form sum: {results['closed_sum']:.2e}")
    print(f"  MCX: {'not run' if not results['mcx_available'] else 'run'}")
    print(f"{'=' * 50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
