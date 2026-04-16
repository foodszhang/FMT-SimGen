#!/usr/bin/env python3
"""Debug Green function peak location."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
    compute_diffusion_params,
)


def main():
    # Load atlas
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Parameters
    voxel_size = 0.2
    camera_distance = 200.0
    fov_mm = 50.0
    detector_resolution = (256, 256)
    angle = 0.0

    tissue_params = {
        "mua_mm": 0.08697,
        "mus_prime_mm": 4.29071,
        "g": 0.9,
        "n": 1.37,
    }

    # Source position
    source_pos = np.array([17.0, 48.0, 8.1])

    # Get surface coordinates
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary, angle, camera_distance, fov_mm, detector_resolution, voxel_size
    )

    # Compute Green's function
    green_proj = green_infinite_point_source_on_surface(
        source_pos, surface_coords, valid_mask, tissue_params
    )

    # Find peak
    peak_idx = np.unravel_index(np.argmax(green_proj), green_proj.shape)
    peak_value = green_proj[peak_idx]
    peak_coord = surface_coords[peak_idx]

    print(f"Source position: {source_pos}")
    print(f"\nGreen function peak:")
    print(f"  Pixel: {peak_idx}")
    print(f"  Value: {peak_value:.6e}")
    print(f"  Surface coord: {peak_coord}")

    # Compute distance
    dist = np.linalg.norm(peak_coord - source_pos)
    print(f"  Distance from source: {dist:.2f} mm")

    # Expected: point directly above source on dorsal surface
    dorsal_z = 10.1  # mm
    expected_dorsal = np.array([17.0, 48.0, dorsal_z])
    dist_expected = np.linalg.norm(expected_dorsal - source_pos)
    print(f"\nExpected dorsal point: {expected_dorsal}")
    print(f"  Distance from source: {dist_expected:.2f} mm")

    # Find pixel closest to expected dorsal point
    dx = surface_coords[:, :, 0] - expected_dorsal[0]
    dy = surface_coords[:, :, 1] - expected_dorsal[1]
    dz = surface_coords[:, :, 2] - expected_dorsal[2]
    dist_to_expected = np.sqrt(dx**2 + dy**2 + dz**2)
    dist_to_expected[~valid_mask] = np.inf

    closest_idx = np.unravel_index(np.argmin(dist_to_expected), dist_to_expected.shape)
    closest_coord = surface_coords[closest_idx]
    closest_dist = dist_to_expected[closest_idx]

    print(f"\nClosest surface point to expected:")
    print(f"  Pixel: {closest_idx}")
    print(f"  Surface coord: {closest_coord}")
    print(f"  Distance to expected: {closest_dist:.2f} mm")
    print(f"  Green value: {green_proj[closest_idx]:.6e}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Green projection
    im0 = axes[0, 0].imshow(green_proj, cmap="hot")
    axes[0, 0].plot(
        peak_idx[1], peak_idx[0], "b*", markersize=15, label=f"Peak {peak_idx}"
    )
    axes[0, 0].plot(
        closest_idx[1],
        closest_idx[0],
        "g+",
        markersize=15,
        label=f"Closest to expected {closest_idx}",
    )
    axes[0, 0].set_title("Green Projection")
    axes[0, 0].legend()
    plt.colorbar(im0, ax=axes[0, 0])

    # Green log
    green_log = np.log10(green_proj + 1e-20)
    im1 = axes[0, 1].imshow(green_log, cmap="hot")
    axes[0, 1].plot(peak_idx[1], peak_idx[0], "b*", markersize=15)
    axes[0, 1].plot(closest_idx[1], closest_idx[0], "g+", markersize=15)
    axes[0, 1].set_title("Green (log10)")
    plt.colorbar(im1, ax=axes[0, 1])

    # Distance from source
    dx_all = surface_coords[:, :, 0] - source_pos[0]
    dy_all = surface_coords[:, :, 1] - source_pos[1]
    dz_all = surface_coords[:, :, 2] - source_pos[2]
    dist_all = np.sqrt(dx_all**2 + dy_all**2 + dz_all**2)
    dist_all[~valid_mask] = np.nan

    im2 = axes[0, 2].imshow(dist_all, cmap="viridis")
    axes[0, 2].plot(peak_idx[1], peak_idx[0], "r*", markersize=15)
    axes[0, 2].set_title("Distance from source (mm)")
    plt.colorbar(im2, ax=axes[0, 2])

    # Z coordinate
    z_coords = surface_coords[:, :, 2]
    z_coords[~valid_mask] = np.nan
    im3 = axes[1, 0].imshow(z_coords, cmap="viridis")
    axes[1, 0].plot(peak_idx[1], peak_idx[0], "r*", markersize=15)
    axes[1, 0].set_title("Surface Z (mm)")
    plt.colorbar(im3, ax=axes[1, 0])

    # Surface X coordinate
    x_coords = surface_coords[:, :, 0]
    x_coords[~valid_mask] = np.nan
    im4 = axes[1, 1].imshow(x_coords, cmap="viridis")
    axes[1, 1].plot(peak_idx[1], peak_idx[0], "r*", markersize=15)
    axes[1, 1].set_title("Surface X (mm)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Surface Y coordinate
    y_coords = surface_coords[:, :, 1]
    y_coords[~valid_mask] = np.nan
    im5 = axes[1, 2].imshow(y_coords, cmap="viridis")
    axes[1, 2].plot(peak_idx[1], peak_idx[0], "r*", markersize=15)
    axes[1, 2].set_title("Surface Y (mm)")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    output_path = Path(__file__).parent / "_debug_green_peak.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
