#!/usr/bin/env python3
"""Figure X: E1b-Atlas MCX vs Analytic Green on Atlas Surface.

This is the CRITICAL figure showing that analytic Green's function
matches MCX even on realistic atlas surface geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color


def load_e1b_results(results_dir):
    """Load E1b experiment results."""
    configs = [
        ("E1b-A1-shallow", "Shallow (d=3mm)"),
        ("E1b-A2-deep", "Deep (d=7mm)"),
        ("E1b-A3-lateral", "Lateral"),
    ]

    data = []
    for config_id, label in configs:
        result_path = results_dir / f"{config_id}_results.npz"
        if result_path.exists():
            result = np.load(result_path, allow_pickle=True)
            metrics = (
                result["metrics"].item()
                if isinstance(result["metrics"], np.ndarray)
                else result["metrics"]
            )

            # Handle case where metrics might be a string
            if isinstance(metrics, str):
                metrics = json.loads(metrics)

            data.append(
                {
                    "id": config_id,
                    "label": label,
                    "surface_coords": result["surface_coords"],
                    "mcx_response": result["mcx_response"],
                    "green_response": result["green_response"],
                    "metrics": metrics,
                    "source_center": result.get("source_center", None),
                }
            )
        else:
            print(f"Warning: {result_path} not found")

    return data


def plot_e1b_atlas_comparison(data, output_path):
    """Generate the CRITICAL E1b figure.

    Layout: 3 rows (configs) x 4 columns
    - Col 1: MCX surface response (scatter)
    - Col 2: Green surface response (scatter)
    - Col 3: |Residual| (scatter)
    - Col 4: MCX vs Green scatter plot
    """
    set_paper_style()

    n_configs = len(data)
    fig, axes = plt.subplots(n_configs, 4, figsize=(16, 4.5 * n_configs))
    fig.patch.set_facecolor("white")

    for row, cfg in enumerate(data):
        surface_coords = cfg["surface_coords"]
        mcx_resp = cfg["mcx_response"]
        green_resp = cfg["green_response"]
        metrics = cfg["metrics"]
        label = cfg["label"]

        # Normalize for comparison
        mcx_norm = mcx_resp / mcx_resp.max()
        green_norm = green_resp / green_resp.max()
        vmax = 1.0

        # Panel 1: MCX
        ax = axes[row, 0]
        sc1 = ax.scatter(
            surface_coords[:, 0],
            surface_coords[:, 1],
            c=mcx_norm,
            cmap="inferno",
            s=5,
            vmin=0,
            vmax=vmax,
            alpha=0.8,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"MCX ({label})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(0, 35)
        ax.set_ylim(35, 60)

        # Panel 2: Green
        ax = axes[row, 1]
        sc2 = ax.scatter(
            surface_coords[:, 0],
            surface_coords[:, 1],
            c=green_norm,
            cmap="inferno",
            s=5,
            vmin=0,
            vmax=vmax,
            alpha=0.8,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Analytic Green", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(0, 35)
        ax.set_ylim(35, 60)

        # Panel 3: Residual
        ax = axes[row, 2]
        residual = np.abs(mcx_norm - green_norm)
        vmax_res = residual.max()
        sc3 = ax.scatter(
            surface_coords[:, 0],
            surface_coords[:, 1],
            c=residual,
            cmap="hot",
            s=5,
            vmin=0,
            vmax=vmax_res,
            alpha=0.8,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"|Residual|, max={vmax_res:.4f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(0, 35)
        ax.set_ylim(35, 60)

        # Panel 4: Scatter (MCX vs Green)
        ax = axes[row, 3]
        ax.scatter(mcx_norm, green_norm, s=2, alpha=0.3, c="steelblue", rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect agreement")

        ncc = metrics.get("ncc", 0)
        rmse = metrics.get("rmse", 0)

        ax.set_xlabel("MCX (normalized)")
        ax.set_ylabel("Green (normalized)")
        ax.set_title(f"Scatter: NCC={ncc:.4f}, RMSE={rmse:.4f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper left")

        # Add text box with metrics
        textstr = f"NCC: {ncc:.4f}\nRMSE: {rmse:.6f}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.02, 0.55, 0.01, 0.35])
    plt.colorbar(sc1, cax=cbar_ax1, label="Normalized Response")

    cbar_ax2 = fig.add_axes([0.02, 0.12, 0.01, 0.35])
    plt.colorbar(sc3, cax=cbar_ax2, label="|Residual|")

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    # Save
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E1b paper figure."""
    results_dir = Path(__file__).parent.parent / "e1b_atlas_mcx" / "results"
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading E1b results...")
    data = load_e1b_results(results_dir)

    if not data:
        print("No E1b results found. Run experiment first.")
        return

    print(f"Loaded {len(data)} configs")

    print("Generating E1b figure...")
    output_path = output_dir / "fig_e1b_atlas_mcx_vs_green"
    plot_e1b_atlas_comparison(data, output_path)

    print("\n" + "=" * 60)
    print("E1b Figure Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
