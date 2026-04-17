#!/usr/bin/env python3
"""Generate visualization summary of Stage 2 v2 diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def create_summary_figure():
    """Create a summary figure with key findings."""

    results_base = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"

    positions = [
        ("S2-Vol-P1-dorsal-r2.0", "P1-dorsal", 0),
        ("S2-Vol-P2-left-r2.0", "P2-left", 90),
        ("S2-Vol-P3-right-r2.0", "P3-right", -90),
        ("S2-Vol-P4-dorsal-lat-r2.0", "P4-dorsal-lat", -30),
        ("S2-Vol-P5-ventral-r2.0", "P5-ventral", 60),
    ]

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 3, 1)
    names = []
    k_sums = []
    nccs = []
    for pos_id, name, angle in positions:
        mcx = np.load(results_base / pos_id / f"mcx_a{angle}.npy")
        green = np.load(results_base / pos_id / f"green_a{angle}.npy")
        with open(results_base / pos_id / "results.json") as f:
            results = json.load(f)
        names.append(name.replace("-", "\n"))
        k_sums.append(mcx.sum() / green.sum())
        nccs.append(results["ncc"])

    x = np.arange(len(names))
    bars = ax1.bar(
        x, np.log10(k_sums), color=["green", "yellow", "yellow", "yellow", "red"]
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("log₁₀(k)", fontsize=12)
    ax1.set_title("Scale Factor (k = MCX/Green)", fontsize=14)
    ax1.axhline(y=7, color="gray", linestyle="--", alpha=0.5, label="log₁₀(1e7)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    colors = ["green" if n >= 0.9 else "red" for n in nccs]
    bars = ax2.bar(x, nccs, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("NCC", fontsize=12)
    ax2.set_title("Normalized Cross-Correlation", fontsize=14)
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="threshold")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(
        np.log10(k_sums), nccs, s=100, c=["green", "yellow", "yellow", "yellow", "red"]
    )
    for i, name in enumerate(names):
        ax3.annotate(
            name.replace("\n", "-"),
            (np.log10(k_sums[i]), nccs[i]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax3.set_xlabel("log₁₀(k)", fontsize=12)
    ax3.set_ylabel("NCC", fontsize=12)
    ax3.set_title("Scale Factor vs NCC", fontsize=14)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    p5_mcx = np.load(results_base / "S2-Vol-P5-ventral-r2.0" / "mcx_a60.npy")
    p5_green = np.load(results_base / "S2-Vol-P5-ventral-r2.0" / "green_a60.npy")

    p5_mcx_norm = p5_mcx / p5_mcx.max()
    p5_green_norm = p5_green / p5_green.max()

    ax4.imshow(p5_mcx_norm, cmap="hot", origin="lower")
    ax4.set_title("P5 MCX (normalized)", fontsize=14)
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(p5_green_norm, cmap="hot", origin="lower")
    ax5.set_title("P5 Green (normalized)", fontsize=14)
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    tissues = ["Liver", "Soft tissue", "Stomach"]
    percentages = [50.0, 33.3, 16.7]
    colors = ["red", "green", "blue"]
    ax6.pie(
        percentages, labels=tissues, colors=colors, autopct="%1.1f%%", startangle=90
    )
    ax6.set_title("P5 Path Tissue Composition", fontsize=14)

    fig.suptitle(
        "Stage 2 v2 Diagnostic Summary\n"
        "4/5 positions pass (NCC ≥ 0.9)\n"
        "P5 failure: heterogeneity (50% liver path, 2.35× μa)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "diagnostic_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'diagnostic_summary.png'}")


if __name__ == "__main__":
    create_summary_figure()
