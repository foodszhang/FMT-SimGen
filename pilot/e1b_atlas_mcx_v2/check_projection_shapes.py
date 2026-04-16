#!/usr/bin/env python3
"""Check projection shapes separately."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def main():
    results_dir = Path(__file__).parent / "results/stage1_5_surface"
    figures_dir = Path(__file__).parent / "results/figures"

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
    config_id = f"S1.5-D{depth_mm}mm"
    result_dir = results_dir / config_id

    # Load projections
    mcx_proj = np.load(result_dir / "mcx_projection_a0.npy")
    green_proj = np.load(result_dir / "green_projection_a0.npy")

    print(
        f"MCX: shape={mcx_proj.shape}, max={mcx_proj.max():.4e}, min={mcx_proj.min():.4e}"
    )
    print(
        f"Green: shape={green_proj.shape}, max={green_proj.max():.4e}, min={green_proj.min():.4e}"
    )

    # Separate normalization
    mcx_norm = mcx_proj / mcx_proj.max()
    green_norm = green_proj / green_proj.max()

    # Create figure - separate normalization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Individual normalization
    im0 = axes[0, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)
    axes[0, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0, 0].set_title(f"MCX (separate norm)\nmax={mcx_proj.max():.4e}")
    axes[0, 0].set_xlabel("X pixel")
    axes[0, 0].set_ylabel("Y pixel")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[0, 1].set_title(f"Green (separate norm)\nmax={green_proj.max():.4e}")
    axes[0, 1].set_xlabel("X pixel")
    axes[0, 1].set_ylabel("Y pixel")
    plt.colorbar(im1, ax=axes[0, 1])

    # Overlay both with different colors
    # Green in green channel, MCX in red channel
    overlay = np.zeros((*mcx_norm.shape, 3))
    overlay[:, :, 0] = mcx_norm  # Red = MCX
    overlay[:, :, 1] = green_norm  # Green = Green
    overlay[:, :, 2] = 0  # Blue = 0

    im2 = axes[0, 2].imshow(overlay)
    axes[0, 2].contour(valid_mask, levels=[0.5], colors="white", linewidths=1)
    axes[0, 2].set_title("Overlay: Red=MCX, Green=Green\nYellow=Match")
    axes[0, 2].set_xlabel("X pixel")
    axes[0, 2].set_ylabel("Y pixel")

    # Row 1: Log scale
    mcx_log = np.log10(mcx_norm + 1e-10)
    green_log = np.log10(green_norm + 1e-10)

    im3 = axes[1, 0].imshow(mcx_log, cmap="hot", vmin=-5, vmax=0)
    axes[1, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[1, 0].set_title("MCX (log10)")
    axes[1, 0].set_xlabel("X pixel")
    axes[1, 0].set_ylabel("Y pixel")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(green_log, cmap="hot", vmin=-20, vmax=0)
    axes[1, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[1, 1].set_title("Green (log10)")
    axes[1, 1].set_xlabel("X pixel")
    axes[1, 1].set_ylabel("Y pixel")
    plt.colorbar(im4, ax=axes[1, 1])

    # Difference (normalized separately)
    diff = np.abs(mcx_norm - green_norm)
    im5 = axes[1, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
    axes[1, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
    axes[1, 2].set_title(f"|Difference|\nmax diff = {diff.max():.4f}")
    axes[1, 2].set_xlabel("X pixel")
    axes[1, 2].set_ylabel("Y pixel")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle(
        f"Stage 1.5: Separate Normalization Check\n{config_id}, 0°", fontsize=14
    )
    plt.tight_layout()

    output_path = figures_dir / "stage1_5_separate_norm.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")

    # Find where MCX signal is non-zero
    mcx_nonzero = np.argwhere(mcx_norm > 0.01)  # > 1% of peak
    if len(mcx_nonzero) > 0:
        print(f"\nMCX signal > 1% of peak:")
        print(
            f"  Pixel range X: [{mcx_nonzero[:, 1].min()}, {mcx_nonzero[:, 1].max()}]"
        )
        print(
            f"  Pixel range Y: [{mcx_nonzero[:, 0].min()}, {mcx_nonzero[:, 0].max()}]"
        )

    # Check if signal is clipped by mouse boundary
    # Find mouse boundary pixels
    boundary = np.zeros_like(valid_mask, dtype=bool)
    from scipy import ndimage

    eroded = ndimage.binary_erosion(valid_mask)
    boundary = valid_mask & ~eroded

    # Check if MCX signal touches boundary
    mcx_at_boundary = mcx_norm[boundary].max()
    print(f"\nMCX signal at mouse boundary: {mcx_at_boundary:.4f} (max normalized)")
    print(f"  If > 0.01, signal is reaching the boundary")


if __name__ == "__main__":
    main()
