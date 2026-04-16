#!/usr/bin/env python3
"""Plot Stage 1.5 projections with mouse outline visible."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def main():
    results_dir = Path(__file__).parent / "results/stage1_5_surface"
    figures_dir = Path(__file__).parent / "results/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas to get surface outline
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Parameters
    voxel_size = 0.2
    camera_distance = 200.0
    fov_mm = 50.0
    detector_resolution = (256, 256)

    # Select depth
    depth_mm = 4
    config_id = f"S1.5-D{depth_mm}mm"
    result_dir = results_dir / config_id

    # Load projections for angle 0
    mcx_proj = np.load(result_dir / "mcx_projection_a0.npy")
    green_proj = np.load(result_dir / "green_projection_a0.npy")

    # Get surface coordinates for this angle
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary, 0.0, camera_distance, fov_mm, detector_resolution, voxel_size
    )

    print(f"Loaded {config_id}, angle 0°")
    print(f"  MCX: max={mcx_proj.max():.4e}")
    print(f"  Green: max={green_proj.max():.4e}")
    print(f"  Valid surface pixels: {np.sum(valid_mask)}")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Linear scale with outline
    vmax = max(mcx_proj.max(), green_proj.max())

    # MCX with surface outline
    im0 = axes[0, 0].imshow(mcx_proj, cmap="hot", vmin=0, vmax=vmax)
    # Overlay surface mask as contour
    axes[0, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 0].set_title(f"MCX Projection\nD={depth_mm}mm, 0°")
    axes[0, 0].set_xlabel("X pixel")
    axes[0, 0].set_ylabel("Y pixel")
    plt.colorbar(im0, ax=axes[0, 0])

    # Green with surface outline
    im1 = axes[0, 1].imshow(green_proj, cmap="hot", vmin=0, vmax=vmax)
    axes[0, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 1].set_title(f"Green Projection\nD={depth_mm}mm, 0°")
    axes[0, 1].set_xlabel("X pixel")
    axes[0, 1].set_ylabel("Y pixel")
    plt.colorbar(im1, ax=axes[0, 1])

    # Surface mask only
    im2 = axes[0, 2].imshow(valid_mask, cmap="gray")
    axes[0, 2].set_title("Mouse Surface Outline\n(Valid Projection Area)")
    axes[0, 2].set_xlabel("X pixel")
    axes[0, 2].set_ylabel("Y pixel")

    # Row 1: Log scale
    mcx_log = np.log10(mcx_proj + 1e-10)
    green_log = np.log10(green_proj + 1e-10)
    vmax_log = max(mcx_log.max(), green_log.max())
    vmin_log = -5

    im3 = axes[1, 0].imshow(mcx_log, cmap="hot", vmin=vmin_log, vmax=vmax_log)
    axes[1, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 0].set_title("MCX (log10)")
    axes[1, 0].set_xlabel("X pixel")
    axes[1, 0].set_ylabel("Y pixel")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(green_log, cmap="hot", vmin=vmin_log, vmax=vmax_log)
    axes[1, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 1].set_title("Green (log10)")
    axes[1, 1].set_xlabel("X pixel")
    axes[1, 1].set_ylabel("Y pixel")
    plt.colorbar(im4, ax=axes[1, 1])

    # Z coordinate map
    z_coords = surface_coords[:, :, 2]
    z_coords[~valid_mask] = np.nan
    im5 = axes[1, 2].imshow(z_coords, cmap="viridis")
    axes[1, 2].set_title("Surface Z Coordinate (mm)")
    axes[1, 2].set_xlabel("X pixel")
    axes[1, 2].set_ylabel("Y pixel")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle(
        f"Stage 1.5: Atlas Shape + Surface-Aware Green\n{config_id}", fontsize=14
    )
    plt.tight_layout()

    output_path = figures_dir / f"stage1_5_with_outline_D{depth_mm}mm.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")

    # Also create a side-by-side comparison with old Stage 1.5 (if available)
    # For now, just show the surface outline clearly
    fig2, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Show MCX projection
    mcx_norm = mcx_proj / mcx_proj.max()
    im = ax.imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)

    # Overlay surface outline
    ax.contour(valid_mask, levels=[0.5], colors="cyan", linewidths=2)

    ax.set_title(
        f"Stage 1.5: MCX Projection with Mouse Outline\n{config_id}, 0°", fontsize=14
    )
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.colorbar(im, ax=ax, label="Normalized Intensity")

    plt.tight_layout()
    output_path2 = figures_dir / f"stage1_5_outline_only_D{depth_mm}mm.png"
    plt.savefig(output_path2, dpi=150)
    print(f"Saved to: {output_path2}")


if __name__ == "__main__":
    main()
