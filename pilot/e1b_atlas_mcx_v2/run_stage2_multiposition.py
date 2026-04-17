#!/usr/bin/env python3
"""Stage 2 Multi-Position: Volume Source at Multiple Positions.

Extension of Stage 1.5 (multiposition point source) to volume sources.
Tests uniform spherical/ellipsoidal sources at P1-P5 positions with
multiple viewing angles to find best-view NCC.

Key differences from Stage 1.5:
- Source is volume (r=1-2mm) instead of point
- Uses cubature sampling (SR-6/grid-27/stratified-33) for analytic projection
- Same projection parameters as multiposition for direct comparison

Design decisions (discovered during implementation):
1. Projection parameters (200mm/50mm/256px) match multiposition test rather than
   original Stage 2 proposal (25mm/22mm/113px). This enables direct comparison
   with Stage 1.5 results and provides sufficient angular coverage for the
   larger FOV needed when source is off-center.
2. Each position runs independent MCX simulation because volume source pattern
   is different for each source location (different bounding box).
3. All 5 positions use spherical source (r=2mm) with 7-point cubature as baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_stage2_uniform_source import (
    create_voxelized_uniform_sphere,
    create_voxelized_uniform_ellipsoid,
    run_mcx_volume_source,
    render_green_uniform_source_projection,
    compute_ncc,
    compute_rmse,
    ATLAS_VOLUME_SHAPE,
    VOXEL_SIZE_MM,
    DEFAULT_TISSUE_PARAMS,
)
from surface_projection import project_get_surface_coords
from fmt_simgen.mcx_projection import project_volume_reference

# IMPORTANT: Match multiposition parameters for direct comparison
# Discovery: Original Stage 2 used (25, 22, 113) but multiposition uses (200, 50, 256)
# The larger FOV is needed for off-center sources at oblique angles
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)

# Best angles from Stage 1.5 multiposition results
# P1: 0°, P2: +90°, P3: -90°, P4: -30°, P5: 60°
BEST_ANGLES = {
    "P1-dorsal": 0.0,
    "P2-left": 90.0,
    "P3-right": -90.0,
    "P4-dorsal-lat": -30.0,
    "P5-ventral": 60.0,
}

logger = logging.getLogger(__name__)


def get_surface_positions(atlas_binary_xyz: np.ndarray, voxel_size_mm: float) -> dict:
    """Find surface positions at Y=center slice (copied from multiposition)."""
    ny = atlas_binary_xyz.shape[1]
    y_center = ny // 2

    slice_xz = atlas_binary_xyz[:, y_center, :]
    tissue_x, tissue_z = np.where(slice_xz > 0)

    if len(tissue_x) == 0:
        raise ValueError("No tissue found")

    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2

    tissue_x_mm = (tissue_x - x_center + 0.5) * voxel_size_mm
    tissue_z_mm = (tissue_z - z_center + 0.5) * voxel_size_mm

    return {
        "dorsal_z": tissue_z_mm.max(),
        "ventral_z": tissue_z_mm.min(),
        "left_x": tissue_x_mm.min(),
        "right_x": tissue_x_mm.max(),
        "center_x": (tissue_x_mm.min() + tissue_x_mm.max()) / 2,
        "center_z": (tissue_z_mm.min() + tissue_z_mm.max()) / 2,
    }


def generate_volume_source_configs(
    surface_positions: dict,
    source_radius_mm: float = 2.0,
    depth_mm: float = 4.0,
    sampling_scheme: str = "7-point",
) -> List[Dict]:
    """Generate volume source configs for P1-P5 positions.

    Each config places a spherical volume source at depth_mm from surface.
    """
    cx = surface_positions["center_x"]
    cz = surface_positions["center_z"]
    y_center = 2.4  # From multiposition

    configs = []

    # P1: Dorsal center
    p1_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P1-dorsal-r{source_radius_mm}",
            "source_pos": [cx, y_center, p1_z],
            "best_angle": BEST_ANGLES["P1-dorsal"],
            "description": f"Dorsal, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    # P2: Left side
    p2_x = surface_positions["left_x"] + depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P2-left-r{source_radius_mm}",
            "source_pos": [p2_x, y_center, cz],
            "best_angle": BEST_ANGLES["P2-left"],
            "description": f"Left side, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    # P3: Right side
    p3_x = surface_positions["right_x"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P3-right-r{source_radius_mm}",
            "source_pos": [p3_x, y_center, cz],
            "best_angle": BEST_ANGLES["P3-right"],
            "description": f"Right side, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    # P4: Dorsal-lateral
    left_offset = (surface_positions["center_x"] - surface_positions["left_x"]) * 0.5
    p4_x = cx - left_offset
    p4_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P4-dorsal-lat-r{source_radius_mm}",
            "source_pos": [p4_x, y_center, p4_z],
            "best_angle": BEST_ANGLES["P4-dorsal-lat"],
            "description": f"Dorsal-lateral, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    # P5: Ventral
    p5_z = surface_positions["ventral_z"] + depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P5-ventral-r{source_radius_mm}",
            "source_pos": [cx, y_center, p5_z],
            "best_angle": BEST_ANGLES["P5-ventral"],
            "description": f"Ventral, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    return configs


def run_single_position_best_angle(
    config: Dict,
    atlas_binary: np.ndarray,
    output_base_dir: Path,
    tissue_params: dict = DEFAULT_TISSUE_PARAMS,
    n_photons: int = 1e8,
) -> Dict:
    """Run single position with Stage 1.5 best angle.

    Uses the same best angles as multiposition test for direct comparison.
    """
    config_id = config["id"]
    source_pos = np.array(config["source_pos"])
    source_radius = config["radius"]
    scheme = config["scheme"]
    best_angle = config["best_angle"]

    output_dir = output_base_dir / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Position: %s", config_id)
    logger.info("Source: %s mm, r=%.1fmm", source_pos, source_radius)
    logger.info("Scheme: %s, Angle: %.0f°", scheme, best_angle)
    logger.info("=" * 70)

    # Create source mask
    source_mask = create_voxelized_uniform_sphere(source_pos, source_radius)
    n_voxels = np.sum(source_mask)
    logger.info("Source voxels: %d", n_voxels)

    # Run MCX (once per position)
    fluence_file = output_dir / "fluence.npy"
    if not fluence_file.exists():
        logger.info("Running MCX...")
        fluence = run_mcx_volume_source(
            volume_file=output_dir / "tissue_volume.bin",
            output_dir=output_dir / "mcx_run",
            source_mask=source_mask,
            tissue_params=tissue_params,
            n_photons=n_photons,
        )
    else:
        logger.info("Loading existing fluence...")
        fluence = np.load(fluence_file)

    # Project at best angle
    logger.info("Projecting at angle %.0f°...", best_angle)

    # Project MCX
    mcx_proj, _ = project_volume_reference(
        fluence,
        best_angle,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    # Compute Green projection
    green_proj = render_green_uniform_source_projection(
        source_pos,
        source_radius,
        atlas_binary,
        best_angle,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        tissue_params,
        VOXEL_SIZE_MM,
        scheme,
    )

    # Compute metrics
    ncc = compute_ncc(mcx_proj, green_proj)
    rmse = compute_rmse(mcx_proj, green_proj)

    logger.info("NCC=%.4f, RMSE=%.4f", ncc, rmse)

    # Save projections
    np.save(output_dir / f"mcx_a{int(best_angle)}.npy", mcx_proj)
    np.save(output_dir / f"green_a{int(best_angle)}.npy", green_proj)

    # Compile results
    result = {
        "config_id": config_id,
        "source_pos": source_pos.tolist(),
        "source_radius_mm": source_radius,
        "sampling_scheme": scheme,
        "best_angle": best_angle,
        "ncc": ncc,
        "rmse": rmse,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_base_dir = Path(__file__).parent / "results" / "stage2_multiposition"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    atlas_binary = volume.transpose(2, 1, 0).astype(np.uint8)

    # Get surface positions
    surface_pos = get_surface_positions(atlas_binary, VOXEL_SIZE_MM)
    logger.info("Surface positions:")
    logger.info("  Dorsal Z: %.1f mm", surface_pos["dorsal_z"])
    logger.info("  Ventral Z: %.1f mm", surface_pos["ventral_z"])
    logger.info("  Left X: %.1f mm", surface_pos["left_x"])
    logger.info("  Right X: %.1f mm", surface_pos["right_x"])

    # Generate configs: r=2mm sphere with 7-point cubature
    configs = generate_volume_source_configs(
        surface_pos,
        source_radius_mm=2.0,
        depth_mm=4.0,
        sampling_scheme="7-point",
    )

    # Run all positions
    all_results = []
    for config in configs:
        try:
            result = run_single_position_best_angle(
                config,
                atlas_binary,
                output_base_dir,
                n_photons=int(1e8),
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Config %s failed: %s", config["id"], e)
            import traceback

            traceback.print_exc()

    # Save summary
    with open(output_base_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Stage 2 Multi-Position Volume Source Summary")
    logger.info("=" * 70)
    logger.info(f"{'Position':<25} {'Angle':<12} {'NCC':<8} {'Status':<10}")
    logger.info("-" * 70)
    for r in all_results:
        status = "✅ PASS" if r["ncc"] >= 0.95 else "❌ FAIL"
        logger.info(
            f"{r['config_id']:<25} {r['best_angle']:>6.0f}°      {r['ncc']:>6.3f}   {status}"
        )
    logger.info("=" * 70)

    # Statistics
    nccs = [r["ncc"] for r in all_results]
    logger.info("NCC Statistics:")
    logger.info("  Mean: %.4f", np.mean(nccs))
    logger.info("  Min:  %.4f", np.min(nccs))
    logger.info("  Max:  %.4f", np.max(nccs))


if __name__ == "__main__":
    main()
