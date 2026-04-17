#!/usr/bin/env python3
"""Plot Stage 2 multiposition volume source results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def load_results(results_dir: Path):
    """Load all position results."""
    with open(results_dir / "summary.json") as f:
        return json.load(f)


def load_atlas():
    """Load atlas binary."""
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    return volume.transpose(2, 1, 0).astype(np.uint8)


def plot_angle_sweep_comparison(results, output_path):
    """Plot angle sweep for all positions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ["steelblue", "coral", "seagreen", "gold", "mediumpurple"]

    for i, (result, color) in enumerate(zip(results, colors)):
        ax = axes[i]
        config_id = result["config_id"].replace("S2-Vol-", "").replace("-r2.0", "")

        # Extract angle sweep data
        angles = []
        nccs = []
        for key in sorted(result["angle_sweep"].keys()):
            angle_data = result["angle_sweep"][key]
            angles.append(angle_data["angle"])
            nccs.append(angle_data["ncc"])

        # Plot
        ax.plot(angles, nccs, "o-", color=color, linewidth=2, markersize=8)
        ax.axhline(y=0.95, color="g", linestyle="--", alpha=0.5, label="Go (0.95)")
        ax.axhline(
            y=0.90, color="orange", linestyle="--", alpha=0.5, label="Caution (0.90)"
        )

        # Mark best
        best_idx = nccs.index(max(nccs))
        ax.plot(angles[best_idx], nccs[best_idx], "r*", markersize=15)

        ax.set_title(f"{config_id}\nBest: {angles[best_idx]}°, NCC={max(nccs):.3f}")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("NCC")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.suptitle(
        "Stage 2 Multi-Position: Volume Source Angle Sweep (7-point cubature)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_comparison_with_point_source(results, output_path):
    """Compare volume source vs point source NCC."""
    # Point source reference data from multiposition
    point_source_ncc = {
        "P1-dorsal": 0.998,
        "P2-left": 0.993,
        "P3-right": 0.993,
        "P4-dorsal-lateral": 0.993,
        "P5-ventral": 0.963,
    }

    positions = ["P1-dorsal", "P2-left", "P3-right", "P4-dorsal-lateral", "P5-ventral"]
    volume_ncc = [r["ncc_best"] for r in results]
    point_ncc = [point_source_ncc[p] for p in positions]

    x = np.arange(len(positions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        point_ncc,
        width,
        label="Point Source (Stage 1.5)",
        color="steelblue",
    )
    bars2 = ax.bar(
        x + width / 2, volume_ncc, width, label="Volume Source (Stage 2)", color="coral"
    )

    ax.axhline(y=0.95, color="g", linestyle="--", alpha=0.5, label="Go threshold")
    ax.axhline(
        y=0.90, color="orange", linestyle="--", alpha=0.5, label="Caution threshold"
    )

    ax.set_ylabel("NCC")
    ax.set_title("Point Source vs Volume Source: Multi-Position Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def print_summary_table(results):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("Stage 2 Multi-Position Volume Source Summary")
    print("=" * 80)
    print(f"{'Position':<25} {'Best Angle':<12} {'NCC':<8} {'Status':<10}")
    print("-" * 80)

    for r in results:
        ncc = r["ncc_best"]
        if ncc >= 0.95:
            status = "✅ PASS"
        elif ncc >= 0.90:
            status = "⚠️ CAUTION"
        else:
            status = "❌ FAIL"

        print(
            f"{r['config_id']:<25} {r['best_angle']:>6.0f}°      {ncc:>6.3f}   {status}"
        )

    print("=" * 80)

    nccs = [r["ncc_best"] for r in results]
    print(f"\nNCC Statistics:")
    print(f"  Mean: {np.mean(nccs):.4f}")
    print(f"  Min:  {np.min(nccs):.4f}")
    print(f"  Max:  {np.max(nccs):.4f}")

    pass_count = sum(1 for n in nccs if n >= 0.95)
    print(
        f"\nPass Rate (NCC ≥ 0.95): {pass_count}/{len(nccs)} ({100 * pass_count / len(nccs):.1f}%)"
    )


def main():
    results_dir = Path(__file__).parent / "results" / "stage2_multiposition"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    print_summary_table(results)

    # Generate plots
    plot_angle_sweep_comparison(results, figures_dir / "stage2_mp_angle_sweep.png")
    plot_comparison_with_point_source(results, figures_dir / "stage2_mp_vs_point.png")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
