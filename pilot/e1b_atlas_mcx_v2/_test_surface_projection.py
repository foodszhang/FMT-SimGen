#!/usr/bin/env python3
"""Test surface-aware projection on a single point source."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import (
    project_get_surface_coords,
    render_green_surface_projection,
)


def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load atlas binary mask (XYZ order)
    atlas_bin_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    )
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    print(f"Atlas volume shape (XYZ): {atlas_binary.shape}")
    print(f"Tissue voxels: {np.sum(atlas_binary > 0)}")

    # Source position (2mm depth)
    dorsal_z = config["dorsal_z_mm"]
    source_xy = config["source_xy"]
    depth_mm = 2.0
    source_z = dorsal_z - depth_mm
    source_pos = np.array([source_xy[0], source_xy[1], source_z])

    print(f"\nSource position: {source_pos} mm")
    print(f"Depth from surface: {depth_mm} mm")

    # Tissue params
    tissue_params = config["tissue_params"]
    proj_cfg = config["projection"]
    mcx_cfg = config["mcx"]

    # Test at 0 degrees
    angle = 0

    print(f"\nRendering Green projection at {angle}°...")
    green_proj = render_green_surface_projection(
        source_pos,
        atlas_binary,
        angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        tissue_params,
        mcx_cfg["voxel_size_mm"],
        green_type="infinite",
    )

    print(f"Green projection shape: {green_proj.shape}")
    print(f"Green projection max: {green_proj.max():.6e}")
    print(f"Green projection non-zero pixels: {np.sum(green_proj > 0)}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Green projection
    im0 = axes[0].imshow(green_proj, cmap="hot")
    axes[0].set_title(f"Green Surface Projection\n0°, max={green_proj.max():.4e}")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")
    plt.colorbar(im0, ax=axes[0])

    # Log scale
    green_log = np.log10(green_proj + 1e-10)
    im1 = axes[1].imshow(green_log, cmap="hot")
    axes[1].set_title("Green Projection (log10)")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("Y pixel")
    plt.colorbar(im1, ax=axes[1])

    # Surface mask (just to verify)
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary,
        angle,
        proj_cfg["camera_distance_mm"],
        proj_cfg["fov_mm"],
        tuple(proj_cfg["detector_resolution"]),
        mcx_cfg["voxel_size_mm"],
    )
    im2 = axes[2].imshow(valid_mask, cmap="gray")
    axes[2].set_title(f"Surface Mask\n{np.sum(valid_mask)} valid pixels")
    axes[2].set_xlabel("X pixel")
    axes[2].set_ylabel("Y pixel")

    plt.tight_layout()
    output_path = Path(__file__).parent / "_test_surface_projection.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nTest visualization saved to: {output_path}")

    # Also test at other angles
    print("\nTesting multiple angles...")
    for angle in [-60, -30, 30, 60]:
        proj = render_green_surface_projection(
            source_pos,
            atlas_binary,
            angle,
            proj_cfg["camera_distance_mm"],
            proj_cfg["fov_mm"],
            tuple(proj_cfg["detector_resolution"]),
            tissue_params,
            mcx_cfg["voxel_size_mm"],
            green_type="infinite",
        )
        print(f"  {angle:3d}°: max={proj.max():.4e}, nonzero={np.sum(proj > 0)}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
