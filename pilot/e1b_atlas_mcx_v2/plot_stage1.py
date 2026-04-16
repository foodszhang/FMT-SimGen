#!/usr/bin/env python3
"""Plot Stage 1 results: NCC vs depth curve and 2D projection comparisons."""

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


def plot_ncc_vs_depth(results: dict, output_path: Path, thresholds: dict = None):
    """Plot NCC vs depth curve (Figure S1-A)."""

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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean NCC with error band (min/max)
    ax.plot(
        depths,
        mean_nccs,
        "o-",
        linewidth=2,
        markersize=8,
        label="Mean NCC",
        color="#2E86AB",
    )
    ax.fill_between(
        depths, min_nccs, max_nccs, alpha=0.3, color="#2E86AB", label="Min-Max Range"
    )

    # Threshold lines
    if thresholds:
        ncc_go = thresholds.get("ncc_go", 0.95)
        ncc_caution = thresholds.get("ncc_caution", 0.85)

        ax.axhline(
            y=ncc_go,
            color="#06A77D",
            linestyle="--",
            linewidth=1.5,
            label=f"GO (NCC ≥ {ncc_go})",
        )
        ax.axhline(
            y=ncc_caution,
            color="#F4A261",
            linestyle="--",
            linewidth=1.5,
            label=f"CAUTION (NCC ≥ {ncc_caution})",
        )

        # Color zones
        ax.axhspan(ncc_go, 1.0, alpha=0.1, color="#06A77D")
        ax.axhspan(ncc_caution, ncc_go, alpha=0.1, color="#F4A261")
        ax.axhspan(0, ncc_caution, alpha=0.1, color="#E63946")

    ax.set_xlabel("Depth from Dorsal Surface (mm)", fontsize=12)
    ax.set_ylabel("Normalized Cross-Correlation (NCC)", fontsize=12)
    ax.set_title(
        "Stage 1: Analytic Green's Function Accuracy vs Depth",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0.7, 1.01)
    ax.set_xlim(0, 14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    # Add text annotations for each point
    for d, ncc in zip(depths, mean_nccs):
        ax.annotate(
            f"{ncc:.4f}",
            xy=(d, ncc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_depth_comparison(
    results_dir: Path, output_path: Path, depths_to_show=[2, 6, 12]
):
    """Plot 2D projection comparison for selected depths (Figure S1-B)."""

    fig, axes = plt.subplots(
        len(depths_to_show), 4, figsize=(16, 4 * len(depths_to_show))
    )

    detector_res = (256, 256)
    fov_mm = 50
    camera_dist = 200
    voxel_size = 0.2
    angle = 0  # Frontal view

    for row_idx, depth_mm in enumerate(depths_to_show):
        config_id = f"S1-D{depth_mm:.0f}mm"
        sample_dir = results_dir / config_id

        if not sample_dir.exists():
            print(f"Warning: {sample_dir} not found, skipping")
            continue

        # Load data
        data = np.load(sample_dir / "comparison.npz")
        fluence_mcx = data["fluence_mcx"]
        fluence_green = data["fluence_green"]

        # Normalize
        fluence_mcx_norm = fluence_mcx / fluence_mcx.max()
        fluence_green_norm = fluence_green / fluence_green.max()

        # Project
        proj_mcx, _ = project_volume_reference(
            fluence_mcx_norm, angle, camera_dist, fov_mm, detector_res, voxel_size
        )
        proj_green, _ = project_volume_reference(
            fluence_green_norm, angle, camera_dist, fov_mm, detector_res, voxel_size
        )

        # Normalize projections
        proj_mcx_norm = proj_mcx / (proj_mcx.max() + 1e-10)
        proj_green_norm = proj_green / (proj_green.max() + 1e-10)

        # Compute residual
        residual = np.abs(proj_mcx_norm - proj_green_norm)

        # Compute NCC
        a_flat = proj_mcx_norm.flatten()
        b_flat = proj_green_norm.flatten()
        ncc = np.corrcoef(a_flat, b_flat)[0, 1]

        # Plot
        vmax = max(proj_mcx_norm.max(), proj_green_norm.max())

        im0 = axes[row_idx, 0].imshow(
            proj_mcx_norm.T, origin="lower", cmap="hot", vmin=0, vmax=vmax
        )
        axes[row_idx, 0].set_title(
            f"MCX (Depth {depth_mm}mm)" if row_idx == 0 else f"MCX"
        )
        axes[row_idx, 0].axis("off")

        im1 = axes[row_idx, 1].imshow(
            proj_green_norm.T, origin="lower", cmap="hot", vmin=0, vmax=vmax
        )
        axes[row_idx, 1].set_title(
            f"Green (Depth {depth_mm}mm)" if row_idx == 0 else f"Green"
        )
        axes[row_idx, 1].axis("off")

        im2 = axes[row_idx, 2].imshow(
            residual.T, origin="lower", cmap="viridis", vmin=0, vmax=residual.max()
        )
        axes[row_idx, 2].set_title(
            f"|Residual| (NCC={ncc:.4f})"
            if row_idx == 0
            else f"|Residual| (NCC={ncc:.4f})"
        )
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
            f"Center Profile (Y={center_y})" if row_idx == 0 else f"Center Profile"
        )
        axes[row_idx, 3].set_xlabel("X (pixels)")
        axes[row_idx, 3].set_ylabel("Normalized Intensity")
        axes[row_idx, 3].legend()
        axes[row_idx, 3].grid(True, alpha=0.3)

    plt.tight_layout()
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
    plot_ncc_vs_depth(results, figures_dir / "stage1_ncc_vs_depth.png", thresholds)

    # Plot 2D comparisons
    plot_depth_comparison(
        results_dir,
        figures_dir / "stage1_projection_comparison.png",
        depths_to_show=[2, 6, 12],
    )

    print("\nStage 1 visualization complete!")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
