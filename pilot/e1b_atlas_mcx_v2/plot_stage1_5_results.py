#!/usr/bin/env python3
"""Plot Stage 1.5 results and compare with Stage 1."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def main():
    results_dir = Path(__file__).parent / "results"

    # Load Stage 1 summary
    stage1_summary = load_summary(results_dir / "stage1/stage1_summary.json")

    # Load Stage 1.5 (surface-aware) summary
    stage1_5_summary = load_summary(
        results_dir / "stage1_5_surface/stage1_5_summary.json"
    )

    # Load old Stage 1.5 (fixed plane) summary for comparison
    stage1_5_old_summary = load_summary(
        results_dir / "stage1_atlas/stage1_atlas_summary.json"
    )

    # Extract data
    depths = []
    stage1_ncc = []
    stage1_5_ncc = []
    stage1_5_old_ncc = []

    for config_id in sorted(stage1_5_summary.keys()):
        depth = stage1_5_summary[config_id]["depth_mm"]
        depths.append(depth)
        stage1_5_ncc.append(stage1_5_summary[config_id]["mean_ncc"])

        # Find corresponding Stage 1 config
        s1_config_id = f"S1-D{int(depth)}mm"
        if s1_config_id in stage1_summary:
            stage1_ncc.append(stage1_summary[s1_config_id]["mean_ncc"])
        else:
            stage1_ncc.append(np.nan)

        # Find corresponding old Stage 1.5 config
        s1_5_old_id = f"S1A-D{int(depth)}mm"
        if s1_5_old_id in stage1_5_old_summary:
            stage1_5_old_ncc.append(stage1_5_old_summary[s1_5_old_id]["mean_ncc"])
        else:
            stage1_5_old_ncc.append(np.nan)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        depths,
        stage1_ncc,
        "o-",
        label="Stage 1: Cube + Fixed Plane Green",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        depths,
        stage1_5_old_ncc,
        "s--",
        label="Stage 1.5 (old): Atlas + Fixed Plane Green",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        depths,
        stage1_5_ncc,
        "^-",
        label="Stage 1.5 (new): Atlas + Surface-Aware Green",
        linewidth=2,
        markersize=8,
    )

    ax.axhline(y=0.95, color="g", linestyle=":", label="Go threshold (0.95)")
    ax.axhline(y=0.85, color="orange", linestyle=":", label="Caution threshold (0.85)")

    ax.set_xlabel("Source Depth (mm)", fontsize=12)
    ax.set_ylabel("Mean NCC", fontsize=12)
    ax.set_title(
        "Stage 1 vs Stage 1.5: Surface-Aware Green Function Comparison", fontsize=14
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add text annotations
    for i, (d, ncc) in enumerate(zip(depths, stage1_5_ncc)):
        ax.annotate(
            f"{ncc:.3f}",
            (d, ncc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = Path(__file__).parent / "results/figures/stage1_5_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("Stage 1.5 Results: Surface-Aware Green Function")
    print("=" * 70)
    print(
        f"{'Depth':<10} {'Stage 1':<12} {'Old S1.5':<12} {'New S1.5':<12} {'Improvement':<12}"
    )
    print("-" * 70)
    for i, d in enumerate(depths):
        improvement = stage1_5_ncc[i] - stage1_5_old_ncc[i]
        print(
            f"{d:.0f}mm      {stage1_ncc[i]:.4f}      {stage1_5_old_ncc[i]:.4f}      {stage1_5_ncc[i]:.4f}      +{improvement:.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
