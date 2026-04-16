#!/usr/bin/env python3
"""Plot Stage 1 results (Version 2): Shape comparison + Peak attenuation curve.

Key insight: MCX and Green have different absolute scales (photon counts vs
normalized response). We compare shapes (NCC) separately from showing the
physical signal attenuation.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmt_simgen.mcx_projection import project_volume_reference


def load_stage1_results(results_dir: Path) -> dict:
    """Load all Stage 1 results from summary file."""
    summary_path = results_dir / "stage1_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def plot_ncc_and_attenuation(results: dict, output_path: Path, thresholds: dict = None):
    """Plot NCC vs depth + Peak attenuation on secondary axis."""

    depths = []
    mean_nccs = []
    min_nccs = []
    max_nccs = []

    for config_id in sorted(results.keys()):
        res = results[config_id]
        depths.append(res["depth_mm"])
        mean_nccs.append(res["mean_ncc"])
        min_nccs.append(res["min_ncc"])
        max_nccs.append(res["max_ncc"])

    depths = np.array(depths)
    mean_nccs = np.array(mean_nccs)
    min_nccs = np.array(min_nccs)
    max_nccs = np.array(max_nccs)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary axis: NCC
    color_ncc = "#2E86AB"
    ax1.plot(
        depths,
        mean_nccs,
        "o-",
        linewidth=2.5,
        markersize=10,
        label="Mean NCC",
        color=color_ncc,
    )
    ax1.fill_between(
        depths, min_nccs, max_nccs, alpha=0.25, color=color_ncc, label="Min-Max Range"
    )

    # Threshold lines
    if thresholds:
        ncc_go = thresholds.get("ncc_go", 0.95)
        ncc_caution = thresholds.get("ncc_caution", 0.85)

        ax1.axhline(
            y=ncc_go,
            color="#06A77D",
            linestyle="--",
            linewidth=2,
            label=f"GO (NCC ≥ {ncc_go})",
        )
        ax1.axhline(
            y=ncc_caution,
            color="#F4A261",
            linestyle="--",
            linewidth=2,
            label=f"CAUTION (NCC ≥ {ncc_caution})",
        )

        ax1.axhspan(ncc_go, 1.0, alpha=0.08, color="#06A77D")
        ax1.axhspan(ncc_caution, ncc_go, alpha=0.08, color="#F4A261")
        ax1.axhspan(0.7, ncc_caution, alpha=0.08, color="#E63946")

    ax1.set_xlabel("Depth from Dorsal Surface (mm)", fontsize=13)
    ax1.set_ylabel("NCC (Shape Similarity)", fontsize=13, color=color_ncc)
    ax1.tick_params(axis="y", labelcolor=color_ncc)
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xlim(0, 14)
    ax1.grid(True, alpha=0.3)

    # Add NCC annotations
    for d, ncc in zip(depths, mean_nccs):
        ax1.annotate(
            f"{ncc:.4f}",
            xy=(d, ncc),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color_ncc,
        )

    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_title(
        "Stage 1: Green vs MCX Accuracy (Homogeneous Cube)\n"
        "NCC measures shape similarity (invariant to absolute scale)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_peak_attenuation(results_dir: Path, output_path: Path):
    """Plot MCX and Green peak values vs depth to show signal attenuation."""

    detector_res = (256, 256)
    fov_mm = 50
    camera_dist = 200
    voxel_size = 0.2
    angle = 0

    depths = []
    mcx_peaks = []
    green_peaks = []

    for depth_mm in [2, 4, 6, 9, 12]:
        config_id = f"S1-D{depth_mm:.0f}mm"
        sample_dir = results_dir / config_id

        if not sample_dir.exists():
            continue

        data = np.load(sample_dir / "comparison.npz")
        fluence_mcx = data["fluence_mcx"]
        fluence_green = data["fluence_green"]

        proj_mcx, _ = project_volume_reference(
            fluence_mcx, angle, camera_dist, fov_mm, detector_res, voxel_size
        )
        proj_green, _ = project_volume_reference(
            fluence_green, angle, camera_dist, fov_mm, detector_res, voxel_size
        )

        depths.append(depth_mm)
        mcx_peaks.append(proj_mcx.max())
        green_peaks.append(proj_green.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Linear scale
    ax1.plot(
        depths, mcx_peaks, "o-", linewidth=2, markersize=8, label="MCX", color="#E63946"
    )
    ax1.plot(
        depths,
        green_peaks,
        "s--",
        linewidth=2,
        markersize=8,
        label="Green",
        color="#2A9D8F",
    )
    ax1.set_xlabel("Depth from Dorsal Surface (mm)", fontsize=12)
    ax1.set_ylabel("Projection Peak (linear)", fontsize=12)
    ax1.set_title("Signal Attenuation (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Log scale
    ax2.semilogy(
        depths, mcx_peaks, "o-", linewidth=2, markersize=8, label="MCX", color="#E63946"
    )
    ax2.semilogy(
        depths,
        green_peaks,
        "s--",
        linewidth=2,
        markersize=8,
        label="Green",
        color="#2A9D8F",
    )
    ax2.set_xlabel("Depth from Dorsal Surface (mm)", fontsize=12)
    ax2.set_ylabel("Projection Peak (log scale)", fontsize=12)
    ax2.set_title("Signal Attenuation (Log Scale)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")

    # Add peak values as text
    for d, mp, gp in zip(depths, mcx_peaks, green_peaks):
        ax1.annotate(
            f"{mp:.1e}",
            xy=(d, mp),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#E63946",
        )
        ax2.annotate(
            f"{mp:.1e}",
            xy=(d, mp),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="#E63946",
        )

    plt.suptitle(
        "Stage 1: Signal Attenuation vs Depth\n"
        "Note: Different scales (MCX=photon counts, Green=normalized response)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_shape_comparison(
    results_dir: Path, output_path: Path, depths_to_show=[2, 4, 6, 9, 12]
):
    """Plot 2D projection comparison focusing on SHAPE (not absolute intensity).

    Each row is normalized independently to show shape similarity.
    """

    detector_res = (256, 256)
    fov_mm = 50
    camera_dist = 200
    voxel_size = 0.2
    angle = 0

    n_depths = len(depths_to_show)
    fig, axes = plt.subplots(n_depths, 4, figsize=(16, 4 * n_depths))

    for row_idx, depth_mm in enumerate(depths_to_show):
        config_id = f"S1-D{depth_mm:.0f}mm"
        sample_dir = results_dir / config_id

        if not sample_dir.exists():
            continue

        # Load data
        data = np.load(sample_dir / "comparison.npz")
        fluence_mcx = data["fluence_mcx"]
        fluence_green = data["fluence_green"]

        # Project
        proj_mcx, _ = project_volume_reference(
            fluence_mcx, angle, camera_dist, fov_mm, detector_res, voxel_size
        )
        proj_green, _ = project_volume_reference(
            fluence_green, angle, camera_dist, fov_mm, detector_res, voxel_size
        )

        # Normalize each to its own peak (for SHAPE comparison)
        mcx_peak = proj_mcx.max()
        green_peak = proj_green.max()
        proj_mcx_norm = proj_mcx / mcx_peak if mcx_peak > 0 else proj_mcx
        proj_green_norm = proj_green / green_peak if green_peak > 0 else proj_green

        # Compute residual
        residual = np.abs(proj_mcx_norm - proj_green_norm)

        # Compute NCC
        a_flat = proj_mcx_norm.flatten()
        b_flat = proj_green_norm.flatten()
        ncc = np.corrcoef(a_flat, b_flat)[0, 1]

        # Plot - use same vmax for MCX and Green within each row
        row_vmax = 1.0

        im0 = axes[row_idx, 0].imshow(
            proj_mcx_norm.T, origin="lower", cmap="hot", vmin=0, vmax=row_vmax
        )
        title = f"MCX (d={depth_mm}mm)\npeak={mcx_peak:.2e}"
        axes[row_idx, 0].set_title(title, fontsize=10)
        axes[row_idx, 0].axis("off")

        im1 = axes[row_idx, 1].imshow(
            proj_green_norm.T, origin="lower", cmap="hot", vmin=0, vmax=row_vmax
        )
        axes[row_idx, 1].set_title(
            f"Green (d={depth_mm}mm)\npeak={green_peak:.2e}", fontsize=10
        )
        axes[row_idx, 1].axis("off")

        im2 = axes[row_idx, 2].imshow(
            residual.T, origin="lower", cmap="viridis", vmin=0, vmax=residual.max()
        )
        axes[row_idx, 2].set_title(f"|Residual|\nNCC={ncc:.4f}", fontsize=10)
        axes[row_idx, 2].axis("off")

        # Center profile
        center_y = proj_mcx_norm.shape[1] // 2
        x_vals = np.arange(proj_mcx_norm.shape[0])
        axes[row_idx, 3].plot(
            x_vals, proj_mcx_norm[:, center_y], "k-", linewidth=2, label="MCX"
        )
        axes[row_idx, 3].plot(
            x_vals, proj_green_norm[:, center_y], "b--", linewidth=2, label="Green"
        )
        axes[row_idx, 3].set_title(
            f"Profile (normalized)\nratio={green_peak / mcx_peak:.2e}", fontsize=10
        )
        axes[row_idx, 3].set_xlabel("X (pixels)")
        axes[row_idx, 3].set_ylabel("Norm. Intensity")
        axes[row_idx, 3].legend(fontsize=8)
        axes[row_idx, 3].grid(True, alpha=0.3)
        axes[row_idx, 3].set_ylim(0, 1.1)

    # Shared colorbar for MCX/Green columns
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Intensity\n(row-wise)", rotation=270, labelpad=20)

    plt.suptitle(
        "Stage 1: Shape Comparison (Row-wise Normalization)\n"
        "NCC measures shape similarity, independent of absolute scale",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    results_dir = Path(__file__).parent / "results" / "stage1"
    figures_dir = Path(__file__).parent / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_stage1_results(results_dir)

    # Load config for thresholds
    config_path = Path(__file__).parent / "config.yaml"
    thresholds = {"ncc_go": 0.95, "ncc_caution": 0.85}
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
            thresholds = config.get("thresholds", thresholds)
    except Exception as e:
        print(f"Could not load config: {e}")

    # Plot 1: NCC vs depth
    plot_ncc_and_attenuation(
        results, figures_dir / "stage1_ncc_vs_depth_v2.png", thresholds
    )

    # Plot 2: Peak attenuation
    plot_peak_attenuation(results_dir, figures_dir / "stage1_peak_attenuation.png")

    # Plot 3: Shape comparison (row-wise normalization)
    plot_shape_comparison(
        results_dir,
        figures_dir / "stage1_shape_comparison_v2.png",
        depths_to_show=[2, 4, 6, 9, 12],
    )

    print("\nStage 1 visualization (V2) complete!")
    print("Files:")
    print(f"  - {figures_dir}/stage1_ncc_vs_depth_v2.png")
    print(f"  - {figures_dir}/stage1_peak_attenuation.png")
    print(f"  - {figures_dir}/stage1_shape_comparison_v2.png")


if __name__ == "__main__":
    main()
