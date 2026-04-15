"""Figure 4: E1d Atlas surface geometry comparison.

Shows that atlas surface must be incorporated vs flat plane assumption.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color, get_label, format_number


def load_e1d_data(results_dir: Path) -> Dict:
    """Load E1d atlas experiment data."""
    data = {
        "summary": {},
        "part_a": {},
        "part_b": {},
        "part_c": {},
        "optimizations": {},
    }

    # Load summary JSON
    summary_file = results_dir / "e1d_r2_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            data["summary"] = summary.get("summary", {})
            data["part_a"] = summary.get("part_a_geometry_results", {})
            data["part_b"] = summary.get("part_b_quadrature_results", {})
            data["part_c"] = summary.get("part_c_inverse_degradation", {})
        except Exception as e:
            print(f"Warning: Failed to load summary: {e}")

    # Load optimization history if available
    atlas_exp_dir = results_dir / "atlas_experiments"
    if atlas_exp_dir.exists():
        for npz_file in atlas_exp_dir.glob("*.npz"):
            try:
                exp_name = npz_file.stem
                npz_data = np.load(npz_file)
                data["optimizations"][exp_name] = {
                    "losses": npz_data.get("losses", None),
                    "center_z": npz_data.get("center_z_history", None),
                    "center_x": npz_data.get("center_x_history", None),
                    "center_y": npz_data.get("center_y_history", None),
                    "sigma_history": npz_data.get("sigma_history", None),
                }
            except Exception as e:
                print(f"Warning: Failed to load {npz_file}: {e}")

    return data


def plot_figure4(data: Dict, output_dir: Path):
    """Generate Figure 4: Atlas vs Flat geometry comparison."""
    set_paper_style()

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.patch.set_facecolor("white")

    part_a = data.get("part_a", {})

    # Row 1: Forward comparison (using summary data)
    # Panel A: Z height colormap placeholder (we'll use text summary)
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("Panel A: Atlas Surface Height", fontsize=10)

    # Show surface height summary
    surface_text = "Atlas Surface Geometry\n\n"
    surface_text += "Realistic mouse torso\n"
    surface_text += "with curvature variations\n\n"
    surface_text += "vs.\n\n"
    surface_text += "Flat plane assumption\n"
    surface_text += "(constant Z)"
    ax.text(
        0.5,
        0.5,
        surface_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # Panel B: Atlas forward response summary
    ax = axes[0, 1]
    a1_shallow = part_a.get("A1_atlas_self_consistent_shallow", {})
    a1_deep = part_a.get("A1_atlas_self_consistent_deep", {})

    categories = ["Shallow\n(d=3mm)", "Deep\n(d=10mm)"]
    pos_errors = [
        a1_shallow.get("position_error_mm", 0),
        a1_deep.get("position_error_mm", 0),
    ]

    bars = ax.bar(
        categories,
        pos_errors,
        color=[get_color("atlas"), get_color("atlas")],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0.5, color="red", linestyle="--", lw=1, label="Threshold (0.5mm)")
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("Panel B: Atlas Forward (A1)")
    ax.set_ylim(0, 1.0)

    for bar, err in zip(bars, pos_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{err:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Flat forward response
    ax = axes[0, 2]
    a2_shallow = part_a.get("A2_atlas_vs_flat_shallow", {})
    a2_deep = part_a.get("A2_atlas_vs_flat_deep", {})

    pos_errors_flat = [
        a2_shallow.get("position_error_mm", 0),
        a2_deep.get("position_error_mm", 0),
    ]

    bars = ax.bar(
        categories,
        pos_errors_flat,
        color=[get_color("flat"), get_color("flat")],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0.5, color="red", linestyle="--", lw=1, label="Threshold (0.5mm)")
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("Panel C: Flat Assumption (A2)")
    ax.set_ylim(0, 4.0)

    for bar, err in zip(bars, pos_errors_flat):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(bar.get_height() + 0.05, 3.8),
            f"{err:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: Inverse comparison
    # Panel D: A1 convergence (placeholder)
    ax = axes[1, 0]
    ax.set_title("Panel D: A1 Atlas Convergence", fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.text(
        0.5,
        0.5,
        "Convergence\nHistory\n(Available in\noptimization files)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)

    # Panel E: A2 convergence
    ax = axes[1, 1]
    ax.set_title("Panel E: A2 Flat Convergence", fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.text(
        0.5,
        0.5,
        "Convergence\nHistory\n(Misconverged\ndue to geometry\nmismatch)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="red",
    )
    ax.grid(True, alpha=0.3)

    # Panel F: Position error comparison
    ax = axes[1, 2]

    x_labels = ["A1-shallow", "A1-deep", "A2-shallow", "A2-deep", "A3-lateral"]
    all_errors = [
        a1_shallow.get("position_error_mm", 0),
        a1_deep.get("position_error_mm", 0),
        a2_shallow.get("position_error_mm", 0),
        a2_deep.get("position_error_mm", 0),
        part_a.get("A3_lateral_source", {}).get("position_error_mm", 0),
    ]
    colors = [
        get_color("atlas"),
        get_color("atlas"),
        get_color("flat"),
        get_color("flat"),
        get_color("atlas"),
    ]

    bars = ax.bar(
        range(len(x_labels)), all_errors, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.axhline(y=0.5, color="red", linestyle="--", lw=1.5, label="Threshold")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Position Error (mm)")
    ax.set_title("Panel F: Position Error Summary")
    ax.set_ylim(0, 4.0)

    for bar, err in zip(bars, all_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(bar.get_height() + 0.05, 3.8),
            f"{err:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = output_dir / "fig4_e1d_atlas_vs_flat"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E1d-atlas figures."""
    base_dir = Path("/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")
    results_dir = base_dir / "results"
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading E1d data...")
    data = load_e1d_data(results_dir)

    if not data.get("part_a"):
        print("Warning: No E1d part A data found. Creating placeholder figure.")

    print("Generating Figure 4: Atlas vs Flat...")
    plot_figure4(data, output_dir)

    print("E1d-atlas figure complete!")


if __name__ == "__main__":
    main()
