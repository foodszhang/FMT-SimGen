#!/usr/bin/env python3
"""Plot multi-position test results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def main():
    results_dir = Path(__file__).parent / "results/multiposition"
    figures_dir = Path(__file__).parent / "results/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_dir / "summary.json") as f:
        results = json.load(f)

    # Load atlas for outline
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Pre-compute outlines for each angle
    angles_needed = set()
    for r in results:
        angles_needed.add(int(r["best_angle"]))

    outlines = {}
    for angle in angles_needed:
        surface_coords, valid_mask = project_get_surface_coords(
            atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2
        )
        outlines[angle] = valid_mask

    # Create figure: 5 rows × 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))

    for row, result in enumerate(results):
        config_id = result["config_id"]
        angle = int(result["best_angle"])

        # Get outline for this angle
        valid_mask = outlines[angle]

        # Load projections
        result_dir = results_dir / config_id
        mcx = np.load(result_dir / f"mcx_a{angle}.npy")
        green = np.load(result_dir / f"green_a{angle}.npy")

        # SEPARATE normalization - each normalized to its own peak
        mcx_norm = mcx / mcx.max()
        green_norm = green / green.max()
        diff = np.abs(mcx_norm - green_norm)

        # MCX - separate colorbar
        im0 = axes[row, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 0].set_title(f"{config_id}\nMCX {angle}° (max={mcx.max():.3e})")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Green - separate colorbar
        im1 = axes[row, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 1].set_title(
            f"Green (max={green.max():.3e})\nNCC={result['ncc_best']:.3f}"
        )
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Diff - separate colorbar
        im2 = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
        axes[row, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 2].set_title(f"|Residual|, max={diff.max():.3f}")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Profile
        row_idx = mcx.shape[0] // 2
        axes[row, 3].plot(mcx_norm[row_idx, :], label="MCX", linewidth=2)
        axes[row, 3].plot(green_norm[row_idx, :], label="Green", linewidth=2)
        axes[row, 3].set_title(f"Profile (row {row_idx})")
        axes[row, 3].legend()
        axes[row, 3].grid(True, alpha=0.3)

    plt.suptitle("Multi-Position × Best-View Test: Surface-Aware Green", fontsize=16)
    plt.tight_layout()

    output_path = figures_dir / "multiposition_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")

    # Create summary table
    print("\n" + "=" * 70)
    print("Multi-Position Test Summary")
    print("=" * 70)
    print(f"{'Position':<20} {'Best Angle':<12} {'NCC':<8} {'RMSE':<8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['config_id']:<20} {r['best_angle']:>6.0f}°      {r['ncc_best']:>6.3f}   {r['rmse_best']:>6.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
