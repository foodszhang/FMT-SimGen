#!/usr/bin/env python3
"""Test MCX pattern3d source coordinate system.

Create a small spherical source and run MCX once to verify:
1. Source pattern is correctly interpreted
2. Fluence peak is at expected location
3. Projection looks reasonable
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_stage2_uniform_source import (
    create_voxelized_uniform_sphere,
    run_mcx_volume_source,
    ATLAS_VOLUME_SHAPE,
    VOXEL_SIZE_MM,
    DEFAULT_TISSUE_PARAMS,
)
from fmt_simgen.mcx_projection import project_volume_reference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MCX executable
MCX_EXE = "/mnt/f/win-pro/bin/mcx.exe"


def test_pattern3d_small_source():
    """Test with a small r=1mm source at 4mm depth."""

    output_dir = Path(__file__).parent / "results" / "stage2_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source parameters
    # Using centered coordinates (origin at volume center)
    source_center_mm = np.array([0.0, 2.4, 4.0])  # 4mm depth from dorsal
    source_radius_mm = 1.0

    logger.info("=" * 60)
    logger.info("Testing pattern3d with small spherical source")
    logger.info("Source center: %s mm", source_center_mm)
    logger.info("Source radius: %.1f mm", source_radius_mm)
    logger.info("=" * 60)

    # Create source mask
    source_mask = create_voxelized_uniform_sphere(source_center_mm, source_radius_mm)
    n_source_voxels = np.sum(source_mask)
    logger.info("Source voxels: %d", n_source_voxels)

    # Save source mask for inspection
    np.save(output_dir / "source_mask.npy", source_mask)

    # Find source bounds
    source_indices = np.argwhere(source_mask > 0)
    logger.info("Source bounds (voxel indices):")
    logger.info("  X: %d - %d", source_indices[:, 0].min(), source_indices[:, 0].max())
    logger.info("  Y: %d - %d", source_indices[:, 1].min(), source_indices[:, 1].max())
    logger.info("  Z: %d - %d", source_indices[:, 2].min(), source_indices[:, 2].max())

    # Convert to physical coordinates
    nx, ny, nz = ATLAS_VOLUME_SHAPE
    center_vox = np.array([nx / 2, ny / 2, nz / 2])
    source_phys = (source_indices - center_vox + 0.5) * VOXEL_SIZE_MM
    logger.info("Source bounds (physical mm, centered):")
    logger.info("  X: %.2f - %.2f mm", source_phys[:, 0].min(), source_phys[:, 0].max())
    logger.info("  Y: %.2f - %.2f mm", source_phys[:, 1].min(), source_phys[:, 1].max())
    logger.info("  Z: %.2f - %.2f mm", source_phys[:, 2].min(), source_phys[:, 2].max())

    # Run MCX
    logger.info("Running MCX...")
    try:
        fluence = run_mcx_volume_source(
            volume_file=output_dir / "tissue_volume.bin",
            output_dir=output_dir / "mcx_run",
            source_mask=source_mask,
            tissue_params=DEFAULT_TISSUE_PARAMS,
            n_photons=int(1e8),
        )
        logger.info("MCX completed successfully!")
    except Exception as e:
        logger.error("MCX failed: %s", e)
        raise

    # Check fluence
    logger.info("Fluence statistics:")
    logger.info("  Shape: %s", fluence.shape)
    logger.info("  Min: %.3e", fluence.min())
    logger.info("  Max: %.3e", fluence.max())
    logger.info("  Mean: %.3e", fluence.mean())

    # Find peak location
    peak_idx = np.unravel_index(np.argmax(fluence), fluence.shape)
    peak_val = fluence[peak_idx]
    peak_phys = (np.array(peak_idx) - center_vox + 0.5) * VOXEL_SIZE_MM
    logger.info("Fluence peak:")
    logger.info("  Voxel index: %s", peak_idx)
    logger.info("  Physical (centered): %s mm", peak_phys)
    logger.info("  Value: %.3e", peak_val)

    # Expected: peak should be near source center
    expected_center = source_center_mm
    distance_to_expected = np.linalg.norm(peak_phys - expected_center)
    logger.info("Distance from peak to expected center: %.2f mm", distance_to_expected)

    if distance_to_expected < 2.0:  # Within 2mm is good
        logger.info("✅ Peak location looks reasonable!")
    else:
        logger.warning(
            "⚠️ Peak location seems off! Expected near %s, got %s",
            expected_center,
            peak_phys,
        )

    # Project to 2D
    logger.info("Projecting to 2D...")
    angle_deg = 0.0
    proj, depth_map = project_volume_reference(
        fluence,
        angle_deg,
        camera_distance=25.0,
        fov_mm=22.0,
        detector_resolution=(113, 113),
        voxel_size_mm=VOXEL_SIZE_MM,
    )

    logger.info("Projection statistics:")
    logger.info("  Shape: %s", proj.shape)
    logger.info("  Min: %.3e", proj.min())
    logger.info("  Max: %.3e", proj.max())
    logger.info("  Mean: %.3e", proj.mean())

    # Save projection
    np.save(output_dir / "test_proj.npy", proj)
    logger.info("Saved projection to %s", output_dir / "test_proj.npy")

    # Quick visualization
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Source mask (center slice)
        center_x = source_mask.shape[0] // 2
        axes[0].imshow(source_mask[center_x, :, :].T, origin="lower", cmap="Reds")
        axes[0].set_title(f"Source Mask (X={center_x})")
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("Z")

        # Fluence (center slice)
        fluence_max = fluence[center_x, :, :].max()
        axes[1].imshow(
            fluence[center_x, :, :].T,
            origin="lower",
            cmap="hot",
            vmin=0,
            vmax=fluence_max,
        )
        axes[1].set_title(f"Fluence (X={center_x}, max={fluence_max:.2e})")
        axes[1].set_xlabel("Y")
        axes[1].set_ylabel("Z")

        # Projection
        proj_norm = proj / proj.max() if proj.max() > 0 else proj
        axes[2].imshow(proj_norm, origin="lower", cmap="hot")
        axes[2].set_title(f"Projection 0° (max={proj.max():.2e})")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")

        plt.tight_layout()
        plt.savefig(output_dir / "test_visualization.png", dpi=150)
        logger.info("Saved visualization to %s", output_dir / "test_visualization.png")

    except Exception as e:
        logger.warning("Could not create visualization: %s", e)

    logger.info("=" * 60)
    logger.info("Test completed!")
    logger.info("Check output in: %s", output_dir)
    logger.info("=" * 60)

    return fluence, proj


if __name__ == "__main__":
    fluence, proj = test_pattern3d_small_source()
