"""D2b: Analyze surface-space NCC by distance and region.

The D2 experiment showed NCC=0.86, which is in the boundary range.
This script analyzes WHERE the discrepancy occurs:
1. By distance from source
2. By depth from surface
3. By tissue type (if available)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_by_distance(
    vertices_mm: np.ndarray,
    phi_mcx: np.ndarray,
    phi_closed: np.ndarray,
    valid: np.ndarray,
    source_pos_mm: np.ndarray,
    output_dir: Path,
):
    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    bins = [0, 3, 6, 9, 12, 15, 20, 30, 50]
    bin_labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-20", "20-30", "30-50"]

    results = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = valid & (dist >= lo) & (dist < hi) & (phi_mcx > 0) & (phi_closed > 0)
        if np.sum(mask) < 10:
            results.append({"range": bin_labels[i], "n": 0, "ncc": 0, "k": 0})
            continue

        mcx_vals = phi_mcx[mask]
        closed_vals = phi_closed[mask]

        log_mcx = np.log10(mcx_vals + 1e-20)
        log_closed = np.log10(closed_vals + 1e-20)
        ncc = np.corrcoef(log_mcx, log_closed)[0, 1]
        k = np.sum(mcx_vals) / np.sum(closed_vals)

        results.append(
            {
                "range": bin_labels[i],
                "n": int(np.sum(mask)),
                "ncc": float(ncc),
                "k": float(k),
                "mean_dist": float(np.mean(dist[mask])),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = range(len(bin_labels))
    ncc_vals = [r["ncc"] for r in results]
    n_vals = [r["n"] for r in results]
    k_vals = [r["k"] for r in results]

    ax = axes[0]
    ax.bar(x, ncc_vals, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("NCC (log space)")
    ax.set_title("NCC vs Distance")
    ax.axhline(0.9, color="green", linestyle="--", label="Target 0.9")
    ax.axhline(0.7, color="orange", linestyle="--", label="Threshold 0.7")
    ax.legend()

    ax = axes[1]
    ax.bar(x, n_vals, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("Vertex count")
    ax.set_title("Sample count vs Distance")

    ax = axes[2]
    ax.bar(x, k_vals, color="purple")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("k = sum(MCX) / sum(closed)")
    ax.set_title("Scale factor k vs Distance")
    ax.axhline(1.51e7, color="red", linestyle="--", label="Global k")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "d2b_distance_analysis.png", dpi=150)
    plt.close()

    return results


def analyze_scatter(
    vertices_mm: np.ndarray,
    phi_mcx: np.ndarray,
    phi_closed: np.ndarray,
    valid: np.ndarray,
    source_pos_mm: np.ndarray,
    output_dir: Path,
):
    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    mask = valid & (phi_mcx > 0) & (phi_closed > 0) & (dist <= 20)

    mcx_vals = phi_mcx[mask]
    closed_vals = phi_closed[mask]
    dist_vals = dist[mask]

    log_mcx = np.log10(mcx_vals + 1e-20)
    log_closed = np.log10(closed_vals + 1e-20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(log_closed, log_mcx, c=dist_vals, cmap="viridis", alpha=0.3, s=1)
    ax.plot([-10, 2], [-10, 2], "r--", label="y=x")
    ax.set_xlabel("log10(Closed-form)")
    ax.set_ylabel("log10(MCX)")
    ax.set_title(f"Scatter: NCC={np.corrcoef(log_mcx, log_closed)[0, 1]:.4f}")
    plt.colorbar(sc, ax=ax, label="Distance (mm)")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    ratio = log_mcx - log_closed
    sc = ax.scatter(dist_vals, ratio, c=dist_vals, cmap="viridis", alpha=0.3, s=1)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("log10(MCX) - log10(Closed)")
    ax.set_title("Residual vs Distance")
    plt.colorbar(sc, ax=ax, label="Distance (mm)")

    plt.tight_layout()
    plt.savefig(output_dir / "d2b_scatter.png", dpi=150)
    plt.close()


def main():
    output_dir = Path("pilot/paper04b_forward/results/d2")

    vertices_mm = np.load(output_dir / "vertices_mm.npy")
    phi_mcx = np.load(output_dir / "phi_mcx.npy")
    phi_closed = np.load(output_dir / "phi_closed.npy")
    valid = np.load(output_dir / "valid.npy")

    with open(output_dir / "d2_results.json") as f:
        meta = json.load(f)
    source_pos_mm = np.array(meta["source_pos_mm"])

    logger.info("Analyzing NCC by distance...")
    dist_results = analyze_by_distance(
        vertices_mm, phi_mcx, phi_closed, valid, source_pos_mm, output_dir
    )

    logger.info("Creating scatter plot...")
    analyze_scatter(vertices_mm, phi_mcx, phi_closed, valid, source_pos_mm, output_dir)

    print("\n" + "=" * 60)
    print("NCC by Distance from Source")
    print("=" * 60)
    print(f"{'Range':<10} {'N':<10} {'NCC':<10} {'k':<15}")
    print("-" * 60)
    for r in dist_results:
        if r["n"] > 0:
            print(f"{r['range']:<10} {r['n']:<10} {r['ncc']:<10.4f} {r['k']:<15.2e}")
    print("=" * 60)

    with open(output_dir / "d2b_distance_results.json", "w") as f:
        json.dump(dist_results, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
