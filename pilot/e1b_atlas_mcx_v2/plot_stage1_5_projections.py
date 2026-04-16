#!/usr/bin/env python3
"""Plot Stage 1.5 projection comparison (MCX vs Green)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    results_dir = Path(__file__).parent / "results/stage1_5_surface"
    figures_dir = Path(__file__).parent / "results/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Select a depth to visualize
    depth_mm = 4  # Best NCC
    config_id = f"S1.5-D{depth_mm}mm"
    result_dir = results_dir / config_id

    # Load projections for angle 0
    mcx_proj = np.load(result_dir / "mcx_projection_a0.npy")
    green_proj = np.load(result_dir / "green_projection_a0.npy")

    print(f"Loaded projections for {config_id}, angle 0°")
    print(f"  MCX: shape={mcx_proj.shape}, max={mcx_proj.max():.4e}")
    print(f"  Green: shape={green_proj.shape}, max={green_proj.max():.4e}")

    # Normalize both to their own peak for shape comparison
    mcx_norm = mcx_proj / mcx_proj.max()
    green_norm = green_proj / green_proj.max()

    # Compute residual
    residual = np.abs(mcx_norm - green_norm)

    # Create figure with 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 0: Linear scale
    vmax = max(mcx_norm.max(), green_norm.max())

    im0 = axes[0, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"MCX Projection\n0°, peak={mcx_proj.max():.4e}")
    axes[0, 0].set_xlabel("X pixel")
    axes[0, 0].set_ylabel("Y pixel")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"Green Projection\n0°, peak={green_proj.max():.4e}")
    axes[0, 1].set_xlabel("X pixel")
    axes[0, 1].set_ylabel("Y pixel")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(residual, cmap="hot", vmin=0, vmax=residual.max())
    axes[0, 2].set_title(f"|Residual|\nmax={residual.max():.4f}")
    axes[0, 2].set_xlabel("X pixel")
    axes[0, 2].set_ylabel("Y pixel")
    plt.colorbar(im2, ax=axes[0, 2])

    # Profile comparison (normalized to MCX peak)
    center_row = mcx_proj.shape[0] // 2
    axes[0, 3].plot(mcx_norm[center_row, :], label="MCX", linewidth=2)
    axes[0, 3].plot(green_norm[center_row, :], label="Green", linewidth=2)
    axes[0, 3].set_title(f"Profile (row {center_row})")
    axes[0, 3].set_xlabel("X pixel")
    axes[0, 3].set_ylabel("Normalized intensity")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Row 1: Log scale
    mcx_log = np.log10(mcx_norm + 1e-10)
    green_log = np.log10(green_norm + 1e-10)
    vmax_log = max(mcx_log.max(), green_log.max())
    vmin_log = -5

    im4 = axes[1, 0].imshow(mcx_log, cmap="hot", vmin=vmin_log, vmax=vmax_log)
    axes[1, 0].set_title("MCX (log10)")
    axes[1, 0].set_xlabel("X pixel")
    axes[1, 0].set_ylabel("Y pixel")
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(green_log, cmap="hot", vmin=vmin_log, vmax=vmax_log)
    axes[1, 1].set_title("Green (log10)")
    axes[1, 1].set_xlabel("X pixel")
    axes[1, 1].set_ylabel("Y pixel")
    plt.colorbar(im5, ax=axes[1, 1])

    # Surface mask (show where surface points are)
    axes[1, 2].axis("off")

    # Log profile
    axes[1, 3].plot(mcx_log[center_row, :], label="MCX", linewidth=2)
    axes[1, 3].plot(green_log[center_row, :], label="Green", linewidth=2)
    axes[1, 3].set_title(f"Profile Log10 (row {center_row})")
    axes[1, 3].set_xlabel("X pixel")
    axes[1, 3].set_ylabel("Log10 intensity")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    plt.suptitle(
        f"Stage 1.5: Atlas Shape + Surface-Aware Green\n{config_id}, 0° view",
        fontsize=14,
    )
    plt.tight_layout()

    output_path = figures_dir / f"stage1_5_projections_D{depth_mm}mm.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")

    # Also create multi-angle comparison for this depth
    fig2, axes2 = plt.subplots(3, 5, figsize=(20, 12))
    angles = [-60, -30, 0, 30, 60]

    for col, angle in enumerate(angles):
        mcx = np.load(result_dir / f"mcx_projection_a{angle}.npy")
        green = np.load(result_dir / f"green_projection_a{angle}.npy")

        mcx_n = mcx / mcx.max()
        green_n = green / green.max()
        resid = np.abs(mcx_n - green_n)

        vmax = max(mcx_n.max(), green_n.max())

        # Row 0: MCX
        im = axes2[0, col].imshow(mcx_n, cmap="hot", vmin=0, vmax=vmax)
        axes2[0, col].set_title(f"MCX {angle}°", fontsize=12)
        axes2[0, col].axis("off")

        # Row 1: Green
        im = axes2[1, col].imshow(green_n, cmap="hot", vmin=0, vmax=vmax)
        axes2[1, col].set_title(f"Green {angle}°", fontsize=12)
        axes2[1, col].axis("off")

        # Row 2: Residual
        im = axes2[2, col].imshow(resid, cmap="hot", vmin=0, vmax=resid.max())
        axes2[2, col].set_title(f"|Residual| {angle}°", fontsize=12)
        axes2[2, col].axis("off")

    plt.suptitle(
        f"Stage 1.5: Multi-Angle Projection Comparison\n{config_id}", fontsize=16
    )
    plt.tight_layout()

    output_path2 = figures_dir / f"stage1_5_multiangle_D{depth_mm}mm.png"
    plt.savefig(output_path2, dpi=150)
    print(f"Saved to: {output_path2}")


if __name__ == "__main__":
    main()
