#!/usr/bin/env python3
"""Compare old vs new Stage 1.5 results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def main():
    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas for outline
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Get surface outline
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2
    )

    depth_mm = 4

    # Load old Stage 1.5 (fixed plane)
    old_dir = results_dir / f"stage1_atlas/S1A-D{depth_mm}mm"
    old_data = np.load(old_dir / "comparison.npz")
    old_mcx = old_data["mcx"]
    old_green = old_data["green"]

    # Load new Stage 1.5 (surface-aware)
    new_dir = results_dir / f"stage1_5_surface/S1.5-D{depth_mm}mm"
    new_mcx = np.load(new_dir / "mcx_projection_a0.npy")
    new_green = np.load(new_dir / "green_projection_a0.npy")

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    vmax_old = max(old_mcx.max(), old_green.max())
    vmax_new = max(new_mcx.max(), new_green.max())

    # Row 0: Old Stage 1.5 (Fixed Plane Green)
    im0 = axes[0, 0].imshow(old_mcx, cmap="hot", vmin=0, vmax=vmax_old)
    axes[0, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 0].set_title(f"Old: MCX\n(Fixed Plane Green)")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(old_green, cmap="hot", vmin=0, vmax=vmax_old)
    axes[0, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 1].set_title(f"Old: Green (Fixed Plane)\nNCC ≈ 0.57")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    # Difference
    old_diff = np.abs(old_mcx / old_mcx.max() - old_green / old_green.max())
    im2 = axes[0, 2].imshow(old_diff, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 2].set_title("Old: |Residual|")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2])

    # Profile
    row = 128
    axes[0, 3].plot(old_mcx[row, :] / old_mcx.max(), label="MCX", linewidth=2)
    axes[0, 3].plot(old_green[row, :] / old_green.max(), label="Green", linewidth=2)
    axes[0, 3].set_title(f"Old: Profile (row {row})")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Row 1: New Stage 1.5 (Surface-Aware Green)
    im4 = axes[1, 0].imshow(new_mcx, cmap="hot", vmin=0, vmax=vmax_new)
    axes[1, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 0].set_title(f"New: MCX\n(Surface-Aware Green)")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(new_green, cmap="hot", vmin=0, vmax=vmax_new)
    axes[1, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 1].set_title(f"New: Green (Surface-Aware)\nNCC ≈ 0.999")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1])

    # Difference
    new_diff = np.abs(new_mcx / new_mcx.max() - new_green / new_green.max())
    im6 = axes[1, 2].imshow(new_diff, cmap="hot", vmin=0, vmax=new_diff.max())
    axes[1, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 2].set_title("New: |Residual|")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2])

    # Profile
    axes[1, 3].plot(new_mcx[row, :] / new_mcx.max(), label="MCX", linewidth=2)
    axes[1, 3].plot(new_green[row, :] / new_green.max(), label="Green", linewidth=2)
    axes[1, 3].set_title(f"New: Profile (row {row})")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    plt.suptitle(
        f"Stage 1.5 Before/After: Fixed Plane vs Surface-Aware Green\nDepth = {depth_mm}mm, Angle = 0°",
        fontsize=14,
    )
    plt.tight_layout()

    output_path = figures_dir / "stage1_5_before_after_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
