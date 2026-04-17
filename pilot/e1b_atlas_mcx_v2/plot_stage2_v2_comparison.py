#!/usr/bin/env python3
"""Plot Stage 2 v2 projection comparison for P1 and P2."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from surface_projection import compute_ncc, compute_rmse


def plot_single_comparison(result_dir: Path, angle: float, output_path: Path):
    """Plot single position comparison."""
    mcx_proj = np.load(result_dir / f"mcx_a{int(angle)}.npy")
    green_proj = np.load(result_dir / f"green_a{int(angle)}.npy")

    mcx_norm = mcx_proj / mcx_proj.max()
    green_norm = green_proj / green_proj.max()

    residual = mcx_norm - green_norm

    ncc = compute_ncc(mcx_proj, green_proj)
    rmse = compute_rmse(mcx_proj, green_proj)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(mcx_norm, cmap="hot", origin="lower")
    axes[0, 0].set_title("MCX Projection (normalized)", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(green_norm, cmap="hot", origin="lower")
    axes[0, 1].set_title("Green Projection (normalized)", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    vmax = np.abs(residual).max()
    im2 = axes[1, 0].imshow(
        residual, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax
    )
    axes[1, 0].set_title(f"Residual (MCX - Green)\nRMSE={rmse:.4f}", fontsize=12)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    center_row = mcx_norm.shape[0] // 2
    axes[1, 1].plot(mcx_norm[center_row, :], "r-", label="MCX", linewidth=1.5)
    axes[1, 1].plot(green_norm[center_row, :], "b--", label="Green", linewidth=1.5)
    axes[1, 1].set_title(f"Center Profile (row={center_row})", fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Normalized Intensity")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{result_dir.name}: NCC = {ncc:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return ncc, rmse


def main():
    base_dir = Path(__file__).parent / "results" / "stage2_multiposition_v2"
    output_dir = (
        Path(__file__).parent / "results" / "stage2_multiposition_v2" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = [
        ("S2-Vol-P1-dorsal-r2.0", 0, "P1-dorsal"),
        ("S2-Vol-P2-left-r2.0", 90, "P2-left"),
        ("S2-Vol-P3-right-r2.0", -90, "P3-right"),
        ("S2-Vol-P4-dorsal-lat-r2.0", -30, "P4-dorsal-lat"),
        ("S2-Vol-P5-ventral-r2.0", 60, "P5-ventral"),
    ]

    print("=" * 60)
    print("Stage 2 v2 Projection Comparison")
    print("=" * 60)

    for pos_id, angle, label in positions:
        result_dir = base_dir / pos_id
        if not result_dir.exists():
            print(f"Skipping {pos_id}: directory not found")
            continue

        output_path = output_dir / f"{label}_comparison.png"
        ncc, rmse = plot_single_comparison(result_dir, angle, output_path)
        print(f"{label}: NCC={ncc:.4f}, RMSE={rmse:.4f}")

    print("=" * 60)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
