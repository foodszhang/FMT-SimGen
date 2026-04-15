"""Figure 5: E1d Quadrature comparison - accuracy vs speed tradeoff.

Shows that SR-6 is sufficiently accurate with minimal points.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color, format_number


def load_quadrature_data(results_dir: Path) -> Dict:
    """Load quadrature comparison data."""
    data = {"gaussian": {}, "uniform": {}}

    summary_file = results_dir / "e1d_r2_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            data["gaussian"] = summary.get("part_b_quadrature_results", {}).get(
                "gaussian_source", {}
            )
            data["uniform"] = summary.get("part_b_quadrature_results", {}).get(
                "uniform_source", {}
            )
            data["recommendation"] = summary.get("part_b_quadrature_results", {}).get(
                "recommendation", {}
            )
        except Exception as e:
            print(f"Warning: Failed to load summary: {e}")

    return data


def plot_figure5(data: Dict, output_dir: Path):
    """Generate Figure 5: Quadrature accuracy-speed tradeoff."""
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")

    gaussian_data = data.get("gaussian", {})
    uniform_data = data.get("uniform", {})

    # Define scheme properties
    scheme_info = {
        "1-point": {"n": 1, "color": get_color("one_point"), "marker": "x"},
        "sr-6": {"n": 6, "color": get_color("sr6"), "marker": "p"},  # pentagon for SR-6
        "ut-7": {"n": 7, "color": get_color("ut7"), "marker": "D"},
        "7-point": {"n": 7, "color": "#92c5de", "marker": "o"},
        "grid-27": {"n": 27, "color": get_color("grid27"), "marker": "v"},
        "stratified-33": {"n": 33, "color": get_color("stratified_33"), "marker": "^"},
        "mc-512": {"n": 512, "color": get_color("mc512"), "marker": "s"},
    }

    # Panel A: NCC vs number of points
    ax = axes[0]

    # Gaussian source
    gaussian_n = []
    gaussian_ncc = []
    gaussian_labels = []

    for scheme, info in scheme_info.items():
        if scheme in gaussian_data:
            ncc = gaussian_data[scheme].get("ncc", None)
            if ncc is not None:
                gaussian_n.append(info["n"])
                gaussian_ncc.append(ncc)
                gaussian_labels.append(scheme)
                ax.scatter(
                    info["n"],
                    ncc,
                    s=150,
                    c=info["color"],
                    marker=info["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    label=scheme if info["n"] <= 30 else None,
                    zorder=5,
                )

    # Uniform source
    uniform_n = []
    uniform_ncc = []

    for scheme, info in scheme_info.items():
        if scheme in uniform_data:
            ncc = uniform_data[scheme].get("ncc", None)
            if ncc is not None:
                uniform_n.append(info["n"])
                uniform_ncc.append(ncc)
                ax.scatter(
                    info["n"],
                    ncc,
                    s=100,
                    c=info["color"],
                    marker=info["marker"],
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.5,
                    zorder=4,
                )

    ax.axhline(y=0.99, color="gray", linestyle="--", lw=1, label="NCC=0.99 threshold")
    ax.set_xlabel("Number of points")
    ax.set_ylabel("NCC")
    ax.set_title("Panel A: Accuracy vs Sampling Points")
    ax.set_xscale("log")
    ax.set_ylim(0.985, 1.001)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(loc="lower right", fontsize=7, ncol=2)

    # Add annotation for SR-6
    if "sr-6" in gaussian_data:
        sr6_ncc = gaussian_data["sr-6"].get("ncc", 0)
        ax.annotate(
            f"SR-6\n(NCC={sr6_ncc:.4f})",
            xy=(6, sr6_ncc),
            xytext=(15, sr6_ncc + 0.002),
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
        )

    # Panel B: Summary comparison table
    ax = axes[1]
    ax.axis("off")

    # Create comparison table
    table_data = []
    table_data.append(["Scheme", "Points", "NCC (Gauss)", "NCC (Uni)", "Verdict"])

    for scheme in ["1-point", "sr-6", "ut-7", "7-point", "grid-27"]:
        if scheme in gaussian_data or scheme in uniform_data:
            n_points = scheme_info.get(scheme, {}).get("n", "-")
            ncc_g = gaussian_data.get(scheme, {}).get("ncc", None)
            ncc_u = uniform_data.get(scheme, {}).get("ncc", None)

            ncc_g_str = format_number(ncc_g, 4) if ncc_g is not None else "-"
            ncc_u_str = format_number(ncc_u, 4) if ncc_u is not None else "-"

            # Verdict
            if scheme == "sr-6":
                verdict = "[RECOMMENDED]"
            elif ncc_g is not None and ncc_g > 0.998:
                verdict = "GOOD"
            elif ncc_g is not None and ncc_g > 0.99:
                verdict = "OK"
            else:
                verdict = "POOR"

            table_data.append(
                [scheme.upper(), str(n_points), ncc_g_str, ncc_u_str, verdict]
            )

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.15, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(fontweight="bold")

    # Highlight SR-6 row
    for i in range(5):
        table[(2, i)].set_facecolor("#FFFACD")  # Light yellow

    ax.set_title("Panel B: Quadrature Scheme Comparison", pad=20)

    # Add recommendation text
    rec_text = data.get("recommendation", {}).get("reason", "")
    if rec_text:
        ax.text(
            0.5,
            -0.1,
            f"Recommendation: {rec_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
        )

    plt.tight_layout()

    output_path = output_dir / "fig5_e1d_quadrature_comparison"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E1d-quadrature figures."""
    base_dir = Path("/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")
    results_dir = base_dir / "results"
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading quadrature data...")
    data = load_quadrature_data(results_dir)

    if not data.get("gaussian") and not data.get("uniform"):
        print("Warning: No quadrature data found. Creating placeholder figure.")

    print("Generating Figure 5: Quadrature comparison...")
    plot_figure5(data, output_dir)

    print("E1d-quadrature figure complete!")


if __name__ == "__main__":
    main()
