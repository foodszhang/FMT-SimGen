#!/usr/bin/env python3
"""Debug surface coordinate computation."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import project_get_surface_coords, rotation_matrix_y


def main():
    # Load atlas
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    print(f"Atlas shape (XYZ): {atlas_binary.shape}")

    # Parameters
    voxel_size = 0.2
    camera_distance = 200.0
    fov_mm = 50.0
    detector_resolution = (256, 256)
    angle = 0.0

    # Get surface coordinates
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary, angle, camera_distance, fov_mm, detector_resolution, voxel_size
    )

    print(f"Surface coords shape: {surface_coords.shape}")
    print(f"Valid mask sum: {np.sum(valid_mask)}")

    # Analyze Z coordinates of surface
    z_coords = surface_coords[:, :, 2]
    z_valid = z_coords[valid_mask]

    print(f"\nSurface Z statistics (mm):")
    print(f"  Min: {z_valid.min():.2f}")
    print(f"  Max: {z_valid.max():.2f}")
    print(f"  Mean: {z_valid.mean():.2f}")

    # Find pixel with max Z (should be dorsal)
    max_z_idx = np.unravel_index(np.argmax(z_coords), z_coords.shape)
    min_z_idx = np.unravel_index(
        np.argmin(z_coords * valid_mask - 1e10 * ~valid_mask), z_coords.shape
    )

    print(f"\nPixel with max Z (dorsal): {max_z_idx}")
    print(f"  Z value: {z_coords[max_z_idx]:.2f} mm")
    print(f"  Coords: {surface_coords[max_z_idx]}")

    print(f"\nPixel with min Z (ventral): {min_z_idx}")
    print(f"  Z value: {z_coords[min_z_idx]:.2f} mm")
    print(f"  Coords: {surface_coords[min_z_idx]}")

    # Visualize Z map
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Z coordinate map
    z_map = np.where(valid_mask, z_coords, np.nan)
    im0 = axes[0].imshow(z_map, cmap="viridis", vmin=z_valid.min(), vmax=z_valid.max())
    axes[0].set_title("Surface Z coordinate (mm)")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")
    plt.colorbar(im0, ax=axes[0])

    # Mark dorsal and ventral points
    axes[0].plot(
        max_z_idx[1], max_z_idx[0], "r*", markersize=15, label=f"Max Z ({max_z_idx})"
    )
    axes[0].plot(
        min_z_idx[1], min_z_idx[0], "b*", markersize=15, label=f"Min Z ({min_z_idx})"
    )
    axes[0].legend()

    # Valid mask
    im1 = axes[1].imshow(valid_mask, cmap="gray")
    axes[1].set_title("Valid surface mask")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("Y pixel")

    # Histogram of Z values
    axes[2].hist(z_valid, bins=50, edgecolor="black")
    axes[2].set_title("Distribution of surface Z values")
    axes[2].set_xlabel("Z (mm)")
    axes[2].set_ylabel("Count")
    axes[2].axvline(
        z_valid.max(), color="r", linestyle="--", label=f"Max Z = {z_valid.max():.1f}"
    )
    axes[2].axvline(
        z_valid.min(), color="b", linestyle="--", label=f"Min Z = {z_valid.min():.1f}"
    )
    axes[2].legend()

    plt.tight_layout()
    output_path = Path(__file__).parent / "_debug_surface_coords.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")

    # Now test with source
    print("\n" + "=" * 50)
    source_pos = np.array([17.0, 48.0, 8.1])  # Z = 8.1 mm, dorsal at 10.1 mm

    # Compute distance from source to each surface point
    dx = surface_coords[:, :, 0] - source_pos[0]
    dy = surface_coords[:, :, 1] - source_pos[1]
    dz = surface_coords[:, :, 2] - source_pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Find surface point closest to source
    r_valid = np.where(valid_mask, r, np.inf)
    closest_idx = np.unravel_index(np.argmin(r_valid), r.shape)

    print(f"Source position: {source_pos}")
    print(f"Closest surface point: {closest_idx}")
    print(f"  Distance: {r[closest_idx]:.2f} mm")
    print(f"  Surface coord: {surface_coords[closest_idx]}")
    print(f"  Expected: ~2mm (source is 2mm below dorsal surface)")


if __name__ == "__main__":
    main()
