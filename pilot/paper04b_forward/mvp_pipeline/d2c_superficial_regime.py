"""D2c: Surface-space NCC filtered to superficial regime.

Based on D2b analysis, the closed-form forward is only valid for
superficial depths (distance from source <= 6-9mm).

This script recomputes NCC with:
1. Only vertices within 6mm of source
2. Only vertices within 9mm of source

These define the "superficial regime" for paper scope.
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


def compute_metrics(
    phi_mcx: np.ndarray, phi_closed: np.ndarray, mask: np.ndarray
) -> dict:
    if np.sum(mask) < 10:
        return {"ncc": 0.0, "k": 0.0, "n_valid": 0, "rmse": 0.0}

    mcx_vals = phi_mcx[mask]
    closed_vals = phi_closed[mask]

    log_mcx = np.log10(mcx_vals + 1e-20)
    log_closed = np.log10(closed_vals + 1e-20)
    ncc = np.corrcoef(log_mcx, log_closed)[0, 1]
    k = np.sum(mcx_vals) / np.sum(closed_vals)
    rmse = np.sqrt(np.mean((log_mcx - log_closed) ** 2))

    return {
        "ncc": float(ncc),
        "k": float(k),
        "n_valid": int(np.sum(mask)),
        "rmse": float(rmse),
    }


def main():
    output_dir = Path("pilot/paper04b_forward/results/d2")

    vertices_mm = np.load(output_dir / "vertices_mm.npy")
    phi_mcx = np.load(output_dir / "phi_mcx.npy")
    phi_closed = np.load(output_dir / "phi_closed.npy")
    valid = np.load(output_dir / "valid.npy")

    with open(output_dir / "d2_results.json") as f:
        meta = json.load(f)
    source_pos_mm = np.array(meta["source_pos_mm"])

    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    results = {}

    for max_dist_mm in [6, 9, 12]:
        mask = valid & (phi_mcx > 0) & (phi_closed > 0) & (dist <= max_dist_mm)
        metrics = compute_metrics(phi_mcx, phi_closed, mask)
        results[f"d{max_dist_mm}mm"] = metrics

        logger.info(
            f"Distance <= {max_dist_mm}mm: N={metrics['n_valid']}, NCC={metrics['ncc']:.4f}, k={metrics['k']:.2e}"
        )

    for percentile in [50, 75, 90]:
        threshold = np.percentile(phi_mcx[valid], percentile)
        mask = valid & (phi_mcx > threshold) & (phi_closed > 0)
        metrics = compute_metrics(phi_mcx, phi_closed, mask)
        results[f"top{100 - percentile}%"] = metrics

        logger.info(
            f"Top {100 - percentile}% fluence: N={metrics['n_valid']}, NCC={metrics['ncc']:.4f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, max_dist_mm in enumerate([6, 9, 12]):
        ax = axes[i]
        mask = valid & (phi_mcx > 0) & (phi_closed > 0) & (dist <= max_dist_mm)
        mcx_vals = phi_mcx[mask]
        closed_vals = phi_closed[mask]

        log_mcx = np.log10(mcx_vals + 1e-20)
        log_closed = np.log10(closed_vals + 1e-20)

        ax.scatter(log_closed, log_mcx, alpha=0.3, s=1)
        ax.plot([-10, 2], [-10, 2], "r--", label="y=x")
        ax.set_xlabel("log10(Closed-form)")
        ax.set_ylabel("log10(MCX)")
        ncc = results[f"d{max_dist_mm}mm"]["ncc"]
        ax.set_title(f"Distance <= {max_dist_mm}mm\nNCC={ncc:.4f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "d2c_superficial_regime.png", dpi=150)
    plt.close()

    with open(output_dir / "d2c_superficial_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("D2c: Surface-Space NCC in Superficial Regime")
    print("=" * 70)
    print(f"{'Filter':<15} {'N':<10} {'NCC':<10} {'k':<15} {'Pass?'}")
    print("-" * 70)
    for key, m in results.items():
        passed = "✓" if m["ncc"] >= 0.90 else ("⚠" if m["ncc"] >= 0.70 else "✗")
        print(
            f"{key:<15} {m['n_valid']:<10} {m['ncc']:<10.4f} {m['k']:<15.2e} {passed}"
        )
    print("=" * 70)

    print("\nConclusion:")
    if results["d6mm"]["ncc"] >= 0.90:
        print("  Superficial regime (<=6mm): NCC >= 0.90 ✓")
        print("  Physics layer is VALID within scope.")
    elif results["d9mm"]["ncc"] >= 0.90:
        print("  Extended regime (<=9mm): NCC >= 0.90 ✓")
        print("  Physics layer is VALID within extended scope.")
    else:
        print("  NCC < 0.90 even in superficial regime.")
        print("  Physics layer may have issues - need Robin BC correction.")


if __name__ == "__main__":
    main()
