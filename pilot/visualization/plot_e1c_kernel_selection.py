"""Figure 3: E1c Kernel selection comparison.

Shows that Green-function family outperforms Gaussian PSF.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color, get_label, format_number


def load_e1c_data(results_dir: Path) -> Dict:
    """Load E1c kernel selection data."""
    data = {"per_config": {}, "mean_metrics": {}}

    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return data

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
        data["mean_metrics"] = summary.get("mean_metrics", {})
        data["per_config"] = summary.get("per_config", {})
        data["selected"] = summary.get("selected_green_function", "green_halfspace")
    except Exception as e:
        print(f"Warning: Failed to load summary: {e}")

    # Try to load profile data from gt_surface
    gt_dir = results_dir / "gt_surface"
    if gt_dir.exists():
        for config_dir in gt_dir.iterdir():
            if config_dir.is_dir():
                cid = config_dir.name
                npz_file = config_dir / f"{cid}_surface_gt.npz"
                if npz_file.exists():
                    try:
                        npz_data = np.load(npz_file)
                        if cid not in data["per_config"]:
                            data["per_config"][cid] = {}
                        data["per_config"][cid]["profile"] = {
                            "x": npz_data.get("x", npz_data.get("grid_x", None)),
                            "y": npz_data.get("y", npz_data.get("grid_y", None)),
                            "intensity": npz_data.get("intensity", None),
                        }
                    except Exception as e:
                        print(f"Warning: Failed to load {npz_file}: {e}")

    return data


def plot_figure3(data: Dict, output_dir: Path):
    """Generate Figure 3: Kernel selection comparison."""
    set_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.patch.set_facecolor("white")

    mean_metrics = data.get("mean_metrics", {})
    per_config = data.get("per_config", {})

    # Panel A: NCC bar chart
    ax = axes[0]
    kernels = ["gaussian_2d", "green_infinite", "green_halfspace"]
    labels = ["Gaussian PSF", "Green (infinite)", "Green (half-space)"]
    colors = [
        get_color("gaussian"),
        get_color("green_infinite"),
        get_color("green_halfspace"),
    ]

    nccs = []
    valid_labels = []
    valid_colors = []

    for kernel, label, color in zip(kernels, labels, colors):
        if kernel in mean_metrics:
            ncc = mean_metrics[kernel].get("mean_ncc", None)
            if ncc is not None:
                nccs.append(ncc)
                valid_labels.append(label)
                valid_colors.append(color)

    if nccs:
        bars = ax.bar(
            range(len(nccs)), nccs, color=valid_colors, edgecolor="black", linewidth=0.5
        )
        ax.set_xticks(range(len(nccs)))
        ax.set_xticklabels(valid_labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Mean NCC")
        ax.set_title("Panel A: NCC Comparison")
        ax.set_ylim(0.8, 1.01)
        ax.axhline(y=0.99, color="gray", linestyle="--", lw=1, label="Threshold")

        # Add value labels on bars
        for bar, ncc in zip(bars, nccs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{ncc:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: FWHM ratio comparison
    ax = axes[1]

    fwhm_ratios = []
    valid_labels_fwhm = []
    valid_colors_fwhm = []

    for kernel, label, color in zip(kernels, labels, colors):
        if kernel in mean_metrics:
            ratio = mean_metrics[kernel].get("mean_fwhm_ratio", None)
            if ratio is not None:
                fwhm_ratios.append(ratio)
                valid_labels_fwhm.append(label)
                valid_colors_fwhm.append(color)

    if fwhm_ratios:
        bars = ax.bar(
            range(len(fwhm_ratios)),
            fwhm_ratios,
            color=valid_colors_fwhm,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(fwhm_ratios)))
        ax.set_xticklabels(valid_labels_fwhm, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Mean FWHM Ratio")
        ax.set_title("Panel B: FWHM Ratio Comparison")
        ax.axhline(y=1.0, color="gray", linestyle="--", lw=1, label="Ideal")
        ax.set_ylim(0.6, 1.1)

        for bar, ratio in zip(bars, fwhm_ratios):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{ratio:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Summary metrics table
    ax = axes[2]
    ax.axis("off")

    # Create summary table
    table_data = []
    table_data.append(["Method", "NCC", "FWHM Ratio", "Decision"])

    for kernel, label in zip(kernels, labels):
        if kernel in mean_metrics:
            ncc = mean_metrics[kernel].get("mean_ncc", 0)
            fwhm = mean_metrics[kernel].get("mean_fwhm_ratio", 0)

            # Determine decision
            if ncc > 0.99 and abs(fwhm - 1.0) < 0.1:
                decision = "GO"
            elif ncc > 0.95:
                decision = "CAUTION"
            else:
                decision = "NO-GO"

            # Highlight selected method
            if kernel == data.get("selected", ""):
                label = f"{label} (*)"
                decision = "SELECTED"

            table_data.append(
                [label, format_number(ncc, 3), format_number(fwhm, 3), decision]
            )

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(fontweight="bold")

    ax.set_title("Panel C: Summary Metrics", pad=20)

    plt.tight_layout()

    output_path = output_dir / "fig3_e1c_kernel_selection"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E1c figures."""
    base_dir = Path("/home/foods/pro/FMT-SimGen/pilot/e1c_green_function_selection")
    results_dir = base_dir / "results"
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading E1c data...")
    data = load_e1c_data(results_dir)

    if not data.get("mean_metrics"):
        print("Warning: No E1c summary data found. Skipping Figure 3.")
        return

    print("Generating Figure 3: Kernel selection...")
    plot_figure3(data, output_dir)

    print("E1c figure complete!")


if __name__ == "__main__":
    main()
