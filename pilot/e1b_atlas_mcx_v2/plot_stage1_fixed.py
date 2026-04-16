#!/usr/bin/env python3
"""Plot Stage 1 results with FIXED normalization.

Key fix: Use global colorbar across all depths to show real signal attenuation.
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


def plot_ncc_vs_depth_with_peak_ratio(
    results: dict, output_path: Path, thresholds: dict = None
):
    """Plot NCC vs depth with peak ratio on secondary axis (Figure S1-A fixed)."""

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

        # Color zones
        ax1.axhspan(ncc_go, 1.0, alpha=0.08, color="#06A77D")
        ax1.axhspan(ncc_caution, ncc_go, alpha=0.08, color="#F4A261")
        ax1.axhspan(0.7, ncc_caution, alpha=0.08, color="#E63946")

    ax1.set_xlabel("Depth from Dorsal Surface (mm)", fontsize=13)
    ax1.set_ylabel("Normalized Cross-Correlation (NCC)", fontsize=13, color=color_ncc)
    ax1.tick_params(axis="y", labelcolor=color_ncc)
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xlim(0, 14)
    ax1.grid(True, alpha=0.3)

    # Add NCC annotations
    for d, ncc in zip(depths, mean_nccs):
        ax1.annotate(
            f"{ncc:.4f}",
            xy=(d, ncc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color_ncc,
        )

    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_title(
        "Stage 1: Analytic Green's Function Accuracy vs Depth (Homogeneous Cube)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_depth_comparison_global_norm(
    results_dir: Path, output_path: Path, depths_to_show=[2, 4, 6, 9, 12]
):
    """Plot 2D projection comparison with GLOBAL normalization (Figure S1-B fixed).

    Key fix: All depths share the same vmax (from shallowest depth) to show
    real signal attenuation.
    """

    detector_res = (256, 256)
    fov_mm = 50
    camera_dist = 200
    voxel_size = 0.2
    angle = 0  # Frontal view

    # First pass: collect all data and find global vmax (from shallowest depth)
    all_data = {}
    global_vmax_mcx = 0
    global_vmax_green = 0

    for depth_mm in depths_to_show:
        config_id = f"S1-D{depth_mm:.0f}mm"
        sample_dir = results_dir / config_id

        if not sample_dir.exists():
            print(f"Warning: {sample_dir} not found, skipping depth {depth_mm}")
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

        # Store raw projections (before normalization)
        all_data[depth_mm] = {
            "proj_mcx": proj_mcx,
            "proj_green": proj_green,
        }

        # Update global vmax (from MCX only, since Green is analytical)
        global_vmax_mcx = max(global_vmax_mcx, proj_mcx.max())

    if not all_data:
        print("No data to plot!")
        return

    # Use the shallowest depth's MCX peak as global vmax
    shallowest_depth = min(all_data.keys())
    global_vmax = all_data[shallowest_depth]["proj_mcx"].max()

    print(f"Global vmax (from {shallowest_depth}mm MCX): {global_vmax:.4e}")

    # Second pass: plot with global normalization
    n_depths = len(all_data)
    fig, axes = plt.subplots(n_depths, 4, figsize=(16, 4 * n_depths))

    if n_depths == 1:
        axes = axes.reshape(1, -1)

    for row_idx, depth_mm in enumerate(sorted(all_data.keys())):
        proj_mcx = all_data[depth_mm]["proj_mcx"]
        proj_green = all_data[depth_mm]["proj_green"]

        # Normalize to global vmax for visualization
        proj_mcx_norm = proj_mcx / global_vmax
        proj_green_norm = proj_green / global_vmax

        # Compute residual
        residual = np.abs(proj_mcx_norm - proj_green_norm)

        # Compute NCC on normalized projections
        a_flat = (proj_mcx / proj_mcx.max()).flatten()
        b_flat = (proj_green / proj_green.max()).flatten()
        ncc = np.corrcoef(a_flat, b_flat)[0, 1]

        # Get absolute peak values
        mcx_peak = proj_mcx.max()
        green_peak = proj_green.max()
        peak_ratio = green_peak / mcx_peak if mcx_peak > 0 else 0

        # Plot
        # Column 1: MCX (global norm)
        im0 = axes[row_idx, 0].imshow(
            proj_mcx_norm.T, origin="lower", cmap="hot", vmin=0, vmax=1.0
        )
        title = f"MCX (Depth {depth_mm}mm)\npeak={mcx_peak:.2e}"
        if row_idx == 0:
            title = f"MCX (Depth {depth_mm}mm, Global Norm)\npeak={mcx_peak:.2e}"
        axes[row_idx, 0].set_title(title, fontsize=10)
        axes[row_idx, 0].axis("off")

        # Column 2: Green (global norm)
        im1 = axes[row_idx, 1].imshow(
            proj_green_norm.T, origin="lower", cmap="hot", vmin=0, vmax=1.0
        )
        axes[row_idx, 1].set_title(
            f"Green (Depth {depth_mm}mm)\npeak={green_peak:.2e}", fontsize=10
        )
        axes[row_idx, 1].axis("off")

        # Column 3: Residual (independent scale)
        im2 = axes[row_idx, 2].imshow(
            residual.T, origin="lower", cmap="viridis", vmin=0, vmax=residual.max()
        )
        axes[row_idx, 2].set_title(f"|Residual|\nNCC={ncc:.4f}", fontsize=10)
        axes[row_idx, 2].axis("off")

        # Column 4: Center profile
        center_y = proj_mcx_norm.shape[1] // 2
        x_vals = np.arange(proj_mcx_norm.shape[0])
        axes[row_idx, 3].plot(
            x_vals, proj_mcx_norm[:, center_y], "k-", linewidth=2, label="MCX"
        )
        axes[row_idx, 3].plot(
            x_vals, proj_green_norm[:, center_y], "b--", linewidth=2, label="Green"
        )
        axes[row_idx, 3].set_title(
            f"Profile (Y={center_y})\nratio={peak_ratio:.2f}", fontsize=10
        )
        axes[row_idx, 3].set_xlabel("X (pixels)")
        axes[row_idx, 3].set_ylabel("Normalized (to global)")
        axes[row_idx, 3].legend(fontsize=8)
        axes[row_idx, 3].grid(True, alpha=0.3)
        axes[row_idx, 3].set_ylim(0, 1.1)

    # Add colorbar for MCX/Green columns (shared)
    cbar_ax = fig.add_axes([0.02, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        "Normalized Intensity\n(to shallowest MCX peak)", rotation=270, labelpad=20
    )

    plt.suptitle(
        "Stage 1: MCX vs Green 2D Projections (Global Normalization)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])
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
    thresholds = None
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
            thresholds = config.get("thresholds", {"ncc_go": 0.95, "ncc_caution": 0.85})
    except Exception as e:
        print(f"Could not load config: {e}")
        thresholds = {"ncc_go": 0.95, "ncc_caution": 0.85}

    # Plot NCC vs depth
    plot_ncc_vs_depth_with_peak_ratio(
        results, figures_dir / "stage1_ncc_vs_depth_fixed.png", thresholds
    )

    # Plot 2D comparisons with global normalization
    plot_depth_comparison_global_norm(
        results_dir,
        figures_dir / "stage1_projection_comparison_fixed.png",
        depths_to_show=[2, 4, 6, 9, 12],
    )

    print("\nStage 1 visualization (FIXED) complete!")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
