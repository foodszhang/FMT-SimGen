#!/usr/bin/env python3
"""Plot Stage 2 results: Uniform Volume Source Validation.

Format matches multiposition_results.png:
- Rows = configs (S2-U1 to S2-U5)
- Cols = [MCX (norm), Green (norm), |Residual|, Profile]
- Each with separate colorbar
- Surface outline (cyan contour)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def load_stage2_results(results_dir: Path) -> list:
    """Load all Stage 2 results."""
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    # Load individual results
    results = []
    for result_file in sorted(results_dir.glob("*/results.json")):
        with open(result_file) as f:
            results.append(json.load(f))
    return results


def load_atlas_for_outline():
    """Load atlas binary mask for surface outline."""
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)
    return atlas_binary


def plot_stage2_comparison(results_dir: Path, output_path: Path):
    """Create Stage 2 comparison figure (matching multiposition_results.png format).

    Rows = configs (S2-U1 to S2-U5)
    Cols = [MCX (norm), Green (norm), |Residual|, Profile]
    """
    configs = [
        "S2-U1-r1mm-sr6",
        "S2-U2-r2mm-sr6",
        "S2-U3-r2mm-grid27",
        "S2-U4-r2mm-strat33",
        "S2-U5-ellipsoid-sr6",
    ]

    row_labels = [
        "S2-U1: r=1mm, 7-point",
        "S2-U2: r=2mm, 7-point",
        "S2-U3: r=2mm, grid-27",
        "S2-U4: r=2mm, stratified-33",
        "S2-U5: ellipsoid, 7-point",
    ]

    # Load atlas for outline (all use 0° angle)
    # Use SAME projection parameters as multiposition (200/50/256)
    atlas_binary = load_atlas_for_outline()
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary,
        angle_deg=0.0,
        camera_distance_mm=200.0,  # Match multiposition
        fov_mm=50.0,  # Match multiposition
        detector_resolution=(256, 256),  # Match multiposition
        voxel_size_mm=0.2,
    )

    # Create figure: 5 rows × 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))

    for row, (config_id, label) in enumerate(zip(configs, row_labels)):
        config_dir = results_dir / config_id

        if not config_dir.exists() or not (config_dir / "mcx_proj.npy").exists():
            for col in range(4):
                axes[row, col].text(
                    0.5,
                    0.5,
                    "Not Run",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                )
                axes[row, col].axis("off")
            continue

        # Load projections
        mcx = np.load(config_dir / "mcx_proj.npy")
        green = np.load(config_dir / "green_proj.npy")

        # Load result metadata
        with open(config_dir / "results.json") as f:
            result = json.load(f)
        ncc = result["ncc"]

        # SEPARATE normalization - each to its own peak
        mcx_norm = mcx / mcx.max() if mcx.max() > 0 else mcx
        green_norm = green / green.max() if green.max() > 0 else green
        diff = np.abs(mcx_norm - green_norm)

        # Column 1: MCX (separate normalization)
        im0 = axes[row, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 0].set_title(f"{label}\nMCX 0° (max={mcx.max():.3e})")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Column 2: Green (separate normalization)
        im1 = axes[row, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1.5)
        axes[row, 1].set_title(f"Green (max={green.max():.3e})\nNCC={ncc:.4f}")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Column 3: |Residual|
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

    plt.suptitle("Stage 2: Uniform Volume Source × Cubature Validation", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison to: {output_path}")


def plot_scheme_comparison_bar(results_dir: Path, output_path: Path):
    """Create bar chart comparing different cubature schemes."""
    results = load_stage2_results(results_dir)

    if not results:
        print("No results found")
        return

    # Group by source radius
    r1_results = [r for r in results if r.get("source_radius_mm") == 1.0]
    r2_results = [r for r in results if r.get("source_radius_mm") == 2.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cubature Scheme Comparison", fontsize=14)

    # Plot r=1mm
    if r1_results:
        schemes = [r["sampling_scheme"] for r in r1_results]
        nccs = [r["ncc"] for r in r1_results]

        bars0 = axes[0].bar(schemes, nccs, color="steelblue")
        axes[0].axhline(y=0.95, color="g", linestyle="--", label="Go threshold (0.95)")
        axes[0].axhline(y=0.90, color="orange", linestyle="--", label="Caution (0.90)")
        axes[0].set_ylabel("NCC", fontsize=12)
        axes[0].set_title("Source Radius = 1mm", fontsize=12)
        axes[0].set_ylim([0, 1.0])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        for bar, ncc in zip(bars0, nccs):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{ncc:.4f}",
                ha="center",
                fontsize=9,
            )

    # Plot r=2mm
    if r2_results:
        schemes = [r["sampling_scheme"] for r in r2_results]
        nccs = [r["ncc"] for r in r2_results]

        bars1 = axes[1].bar(schemes, nccs, color="coral")
        axes[1].axhline(y=0.95, color="g", linestyle="--", label="Go threshold (0.95)")
        axes[1].axhline(y=0.90, color="orange", linestyle="--", label="Caution (0.90)")
        axes[1].set_ylabel("NCC", fontsize=12)
        axes[1].set_title("Source Radius = 2mm", fontsize=12)
        axes[1].set_ylim([0, 1.0])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        for bar, ncc in zip(bars1, nccs):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{ncc:.4f}",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved scheme comparison to: {output_path}")


def print_summary_table(results_dir: Path):
    """Print summary table of results."""
    results = load_stage2_results(results_dir)

    if not results:
        print("No results found")
        return

    print("\n" + "=" * 80)
    print("Stage 2 Results Summary: Uniform Volume Source Validation")
    print("=" * 80)
    print(
        f"{'Config':<25} {'Scheme':<15} {'NCC':>8} {'RMSE':>10} {'Peak Ratio':>12} {'Status':>10}"
    )
    print("-" * 80)

    for r in results:
        config_id = r["config_id"]
        scheme = r["sampling_scheme"]
        ncc = r["ncc"]
        rmse = r["rmse"]
        peak_ratio = r["peak_ratio"]

        if ncc >= 0.95:
            status = "✅ PASS"
        elif ncc >= 0.90:
            status = "⚠️ CAUTION"
        else:
            status = "❌ FAIL"

        print(
            f"{config_id:<25} {scheme:<15} {ncc:>8.4f} {rmse:>10.4f} "
            f"{peak_ratio:>12.3f} {status:>10}"
        )

    print("=" * 80)

    # Compute statistics
    nccs = [r["ncc"] for r in results]
    print(f"\nNCC Statistics:")
    print(f"  Mean: {np.mean(nccs):.4f}")
    print(f"  Min:  {np.min(nccs):.4f}")
    print(f"  Max:  {np.max(nccs):.4f}")
    print(f"  Std:  {np.std(nccs):.4f}")

    pass_count = sum(1 for n in nccs if n >= 0.95)
    print(
        f"\nPass Rate (NCC ≥ 0.95): {pass_count}/{len(nccs)} ({100 * pass_count / len(nccs):.1f}%)"
    )


def main():
    results_dir = Path(__file__).parent / "results" / "stage2"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Print summary table
    print_summary_table(results_dir)

    # Generate plots
    if any(results_dir.glob("*/mcx_proj.npy")):
        plot_stage2_comparison(results_dir, figures_dir / "stage2_results.png")
        plot_scheme_comparison_bar(
            results_dir, figures_dir / "stage2_scheme_comparison.png"
        )
    else:
        print("\nNo projection data found. Run Stage 2 experiments first.")


if __name__ == "__main__":
    main()
