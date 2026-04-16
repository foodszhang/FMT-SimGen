#!/usr/bin/env python3
"""Visualize source positions in the atlas."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_dir = Path(__file__).parent / "results/multiposition"
    figures_dir = Path(__file__).parent / "results/figures"

    # Load results
    with open(results_dir / "summary.json") as f:
        results = json.load(f)

    # Load atlas for background
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Get Y=center slice
    ny = atlas_binary.shape[1]
    y_center = ny // 2
    slice_xz = atlas_binary[:, y_center, :]  # (X, Z)

    # Convert to mm
    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2
    voxel_size = 0.2

    x_mm = (np.arange(nx) - x_center + 0.5) * voxel_size
    z_mm = (np.arange(nz) - z_center + 0.5) * voxel_size

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Show tissue
    ax.contourf(
        x_mm, z_mm, slice_xz.T, levels=[0.5, 1.5], colors="lightgray", alpha=0.5
    )
    ax.contour(x_mm, z_mm, slice_xz.T, levels=[0.5], colors="black", linewidths=2)

    # Plot source positions
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, r in enumerate(results):
        pos = r["source_pos"]
        angle = r["best_angle"]
        ncc = r["ncc_best"]

        ax.plot(
            pos[0],
            pos[2],
            "o",
            markersize=15,
            color=colors[i],
            label=f"{r['config_id']}\n({pos[0]:.1f}, {pos[2]:.1f})mm, {angle}°, NCC={ncc:.3f}",
        )

        # Draw viewing direction arrow
        arrow_length = 5
        dx = arrow_length * np.cos(np.radians(angle + 90))  # +90 because 0° is from +Z
        dz = arrow_length * np.sin(np.radians(angle + 90))
        ax.arrow(
            pos[0],
            pos[2],
            dx,
            dz,
            head_width=1,
            head_length=0.5,
            fc=colors[i],
            ec=colors[i],
            alpha=0.7,
        )

    ax.set_xlabel("X (mm) - Left to Right", fontsize=12)
    ax.set_ylabel("Z (mm) - Ventral to Dorsal", fontsize=12)
    ax.set_title(
        "Source Positions in Atlas (Y=center slice)\nArrows indicate viewing direction",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path = figures_dir / "source_positions_map.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")

    # Also create 3D view info
    print("\nSource position details:")
    print("=" * 70)
    for r in results:
        pos = r["source_pos"]
        print(
            f"{r['config_id']:20s}: X={pos[0]:6.1f}mm, Y={pos[1]:6.1f}mm, Z={pos[2]:6.1f}mm"
        )
        if "left" in r["config_id"]:
            print(
                f"                      -> {abs(pos[0] - (-12.3)):.1f}mm from left surface"
            )
        elif "right" in r["config_id"]:
            print(
                f"                      -> {abs(pos[0] - 10.9):.1f}mm from right surface"
            )
        elif "dorsal" in r["config_id"]:
            print(
                f"                      -> {abs(pos[2] - 10.1):.1f}mm from dorsal surface"
            )
        elif "ventral" in r["config_id"]:
            print(
                f"                      -> {abs(pos[2] - (-8.1)):.1f}mm from ventral surface"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
