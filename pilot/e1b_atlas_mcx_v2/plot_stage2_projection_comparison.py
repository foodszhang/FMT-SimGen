#!/usr/bin/env python3
"""Plot Stage 2 volume source projection comparison (similar to multiposition_results.png).

5 rows x 4 columns grid showing:
- Column 1: MCX projection (normalized)
- Column 2: Green projection (normalized)
- Column 3: |Residual|
- Column 4: Center profile

Each row is a position (P1-P5) with Stage 1.5 best angle.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def load_atlas_for_outline():
    """Load atlas binary mask for surface outline."""
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    return np.where(volume_xyz > 0, 1, 0).astype(np.uint8)


def main():
    results_dir = Path(__file__).parent / "results" / "stage2_multiposition"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_dir / "summary.json") as f:
        results = json.load(f)

    # Load atlas for outline
    atlas_binary = load_atlas_for_outline()

    # Pre-compute outlines for each angle used
    angles_needed = set()
    for r in results:
        angles_needed.add(int(r["best_angle"]))

    outlines = {}
    for angle in angles_needed:
        surface_coords, valid_mask = project_get_surface_coords(
            atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2
        )
        outlines[angle] = valid_mask

    # Create figure: 5 rows x 4 columns
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

        # SEPARATE normalization
        mcx_norm = mcx / mcx.max()
        green_norm = green / green.max()
        diff = np.abs(mcx_norm - green_norm)

        # Column 1: MCX
        im0 = axes[row, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 0].set_title(
            f"{config_id.replace('S2-Vol-', '')}\nMCX {angle}° (max={mcx.max():.3e})"
        )
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Column 2: Green
        im1 = axes[row, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 1].set_title(
            f"Green (max={green.max():.3e})\nNCC={result['ncc']:.3f}"
        )
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Column 3: Residual
        im2 = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
        axes[row, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 2].set_title(f"|Residual|, max={diff.max():.3f}")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Column 4: Profile
        row_idx = mcx.shape[0] // 2
        axes[row, 3].plot(mcx_norm[row_idx, :], label="MCX", linewidth=2)
        axes[row, 3].plot(green_norm[row_idx, :], label="Green", linewidth=2)
        axes[row, 3].set_title(f"Profile (row {row_idx})")
        axes[row, 3].legend()
        axes[row, 3].grid(True, alpha=0.3)

    plt.suptitle(
        "Stage 2 Volume Source: Multi-Position × Best-Angle (7-point Cubature)",
        fontsize=16,
    )
    plt.tight_layout()

    output_path = figures_dir / "stage2_volume_projection_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Stage 2 Volume Source Summary (Best Angles from Stage 1.5)")
    print("=" * 70)
    print(f"{'Position':<25} {'Angle':<10} {'NCC':<8} {'Status':<10}")
    print("-" * 70)
    for r in results:
        ncc = r["ncc"]
        if ncc >= 0.95:
            status = "✅ PASS"
        elif ncc >= 0.90:
            status = "⚠️ CAUTION"
        else:
            status = "❌ FAIL"
        print(
            f"{r['config_id']:<25} {r['best_angle']:>6.0f}°    {ncc:>6.3f}   {status}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
