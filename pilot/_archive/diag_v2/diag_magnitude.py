#!/usr/bin/env python3
"""Magnitude calibration diagnostic for Stage 2 v2.

Performs five checks on P1-dorsal (cleanest position):
1. Print raw value ranges (no max-norm)
2. Source strength normalization check
3. Pixel scatter plot (MCX vs Green) on brightest 10x10 patch
4. Scale factor identification
5. Recompute NCC + RMSE after calibration
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"


def load_projections(position_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load MCX and Green projections for a position."""
    pos_dir = RESULTS_BASE / position_id

    mcx_path = None
    green_path = None
    for f in pos_dir.iterdir():
        if f.name.startswith("mcx_a") and f.suffix == ".npy":
            mcx_path = f
        elif f.name.startswith("green_a") and f.suffix == ".npy":
            green_path = f

    if mcx_path is None or green_path is None:
        raise FileNotFoundError(f"Projections not found in {pos_dir}")

    mcx = np.load(mcx_path)
    green = np.load(green_path)

    results_path = pos_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    return mcx, green, results


def compute_basic_stats(arr: np.ndarray, name: str) -> Dict:
    """Compute basic statistics for an array."""
    nonzero = arr[arr > 0]
    return {
        "name": name,
        "shape": arr.shape,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "sum": float(arr.sum()),
        "nonzero_min": float(nonzero.min()) if len(nonzero) > 0 else 0.0,
        "nonzero_mean": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
        "n_nonzero": int(np.sum(arr > 0)),
    }


def print_stats_comparison(mcx_stats: Dict, green_stats: Dict):
    """Print side-by-side comparison of MCX vs Green stats."""
    print("\n" + "=" * 80)
    print("RAW VALUE RANGES (no max-norm)")
    print("=" * 80)
    print(f"{'Metric':<20} {'MCX':<25} {'Green':<25} {'Ratio':<15}")
    print("-" * 80)

    for key in ["min", "max", "mean", "sum", "nonzero_min", "nonzero_mean"]:
        mcx_val = mcx_stats[key]
        green_val = green_stats[key]
        ratio = mcx_val / green_val if green_val != 0 else float("inf")
        print(f"{key:<20} {mcx_val:<25.6e} {green_val:<25.6e} {ratio:<15.6e}")

    print(
        f"{'n_nonzero':<20} {mcx_stats['n_nonzero']:<25d} {green_stats['n_nonzero']:<25d}"
    )
    print("=" * 80)


def analyze_brightest_patch(
    mcx: np.ndarray, green: np.ndarray, patch_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Analyze the brightest patch in both projections."""
    combined = mcx + green
    max_idx = np.unravel_index(np.argmax(combined), combined.shape)

    cy, cx = max_idx
    h, w = mcx.shape
    half = patch_size // 2

    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    mcx_patch = mcx[y0:y1, x0:x1].flatten()
    green_patch = green[y0:y1, x0:x1].flatten()

    mask = (mcx_patch > 0) & (green_patch > 0)
    mcx_valid = mcx_patch[mask]
    green_valid = green_patch[mask]

    if len(mcx_valid) < 10:
        center_y, center_x = h // 2, w // 2
        y0 = max(0, center_y - half)
        y1 = min(h, center_y + half)
        x0 = max(0, center_x - half)
        x1 = min(w, center_x + half)
        mcx_patch = mcx[y0:y1, x0:x1].flatten()
        green_patch = green[y0:y1, x0:x1].flatten()
        mask = (mcx_patch > 0) & (green_patch > 0)
        mcx_valid = mcx_patch[mask]
        green_valid = green_patch[mask]

    stats = {
        "patch_center": (int(cy), int(cx)),
        "patch_size": (int(y1 - y0), int(x1 - x0)),
        "n_valid_pixels": int(len(mcx_valid)),
        "mcx_mean": float(mcx_valid.mean()),
        "green_mean": float(green_valid.mean()),
    }

    if len(mcx_valid) > 0 and len(green_valid) > 0:
        coeffs = np.polyfit(mcx_valid, green_valid, 1)
        stats["slope"] = float(coeffs[0])
        stats["intercept"] = float(coeffs[1])

        pred = np.polyval(coeffs, mcx_valid)
        ss_res = np.sum((green_valid - pred) ** 2)
        ss_tot = np.sum((green_valid - green_valid.mean()) ** 2)
        stats["r_squared"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats["scale_ratio"] = (
            float(mcx_valid.mean() / green_valid.mean())
            if green_valid.mean() > 0
            else float("inf")
        )

    return mcx_valid, green_valid, stats


def plot_scatter_comparison(
    mcx_valid: np.ndarray,
    green_valid: np.ndarray,
    stats: Dict,
    output_path: Path,
    position_id: str,
):
    """Create scatter plot of MCX vs Green pixel values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(mcx_valid, green_valid, alpha=0.5, s=10, label="Pixels")

    if "slope" in stats:
        x_line = np.linspace(mcx_valid.min(), mcx_valid.max(), 100)
        y_fit = stats["slope"] * x_line + stats["intercept"]
        ax.plot(
            x_line,
            y_fit,
            "r-",
            linewidth=2,
            label=f"Fit: y={stats['slope']:.4f}x+{stats['intercept']:.4f}",
        )

    max_val = max(mcx_valid.max(), green_valid.max())
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="1:1 line")

    ax.set_xlabel("MCX Projection (raw)", fontsize=12)
    ax.set_ylabel("Green Projection (raw)", fontsize=12)
    ax.set_title(
        f"{position_id}: MCX vs Green Pixel Values\n"
        f"Slope={stats.get('slope', 0):.4f}, R²={stats.get('r_squared', 0):.4f}",
        fontsize=14,
    )
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved scatter plot: {output_path}")


def compute_calibrated_metrics(
    mcx: np.ndarray, green: np.ndarray, scale_factor: float
) -> Tuple[float, float]:
    """Compute NCC and RMSE after scale calibration."""
    green_scaled = green * scale_factor

    mcx_flat = mcx.flatten().astype(np.float64)
    green_flat = green_scaled.flatten().astype(np.float64)

    mcx_mean = mcx_flat.mean()
    green_mean = green_flat.mean()

    mcx_centered = mcx_flat - mcx_mean
    green_centered = green_flat - green_mean

    num = np.dot(mcx_centered, green_centered)
    denom = np.sqrt(
        np.dot(mcx_centered, mcx_centered) * np.dot(green_centered, green_centered)
    )

    ncc = float(num / denom) if denom > 1e-10 else 0.0
    rmse = float(np.sqrt(np.mean((mcx_flat - green_flat) ** 2)))

    return ncc, rmse


def analyze_source_normalization(results: Dict) -> Dict:
    """Analyze source normalization factors.

    MCX source: pattern3d with n_source_voxels voxels
    Green source: 7-point cubature for r=2mm sphere

    Expected: MCX total = sum(pattern) * voxel_volume
              Green total = alpha * sphere_volume (with alpha=1.0 in cubature)
    """
    source_radius_mm = results.get("source_radius_mm", 2.0)
    n_source_voxels = results.get("n_source_voxels", 536)
    voxel_size_mm = 0.4

    sphere_volume = 4.0 / 3.0 * np.pi * source_radius_mm**3

    voxel_volume = voxel_size_mm**3
    mcx_source_volume = n_source_voxels * voxel_volume

    volume_ratio = mcx_source_volume / sphere_volume

    return {
        "sphere_volume_mm3": float(sphere_volume),
        "mcx_source_volume_mm3": float(mcx_source_volume),
        "n_source_voxels": int(n_source_voxels),
        "voxel_volume_mm3": float(voxel_volume),
        "volume_ratio": float(volume_ratio),
    }


def run_magnitude_diagnostic(position_id: str = "S2-Vol-P1-dorsal-r2.0") -> Dict:
    """Run full magnitude diagnostic on a position."""
    print("\n" + "=" * 80)
    print(f"MAGNITUDE CALIBRATION DIAGNOSTIC: {position_id}")
    print("=" * 80)

    mcx, green, results = load_projections(position_id)

    mcx_stats = compute_basic_stats(mcx, "MCX")
    green_stats = compute_basic_stats(green, "Green")

    print_stats_comparison(mcx_stats, green_stats)

    k_sum = (
        mcx_stats["sum"] / green_stats["sum"]
        if green_stats["sum"] > 0
        else float("inf")
    )
    k_max = (
        mcx_stats["max"] / green_stats["max"]
        if green_stats["max"] > 0
        else float("inf")
    )
    k_mean = (
        mcx_stats["mean"] / green_stats["mean"]
        if green_stats["mean"] > 0
        else float("inf")
    )

    print("\n" + "-" * 80)
    print("SCALE FACTOR ANALYSIS")
    print("-" * 80)
    print(f"  k_sum   = MCX.sum / Green.sum   = {k_sum:.6e}")
    print(f"  k_max   = MCX.max / Green.max   = {k_max:.6e}")
    print(f"  k_mean  = MCX.mean / Green.mean = {k_mean:.6e}")

    source_norm = analyze_source_normalization(results)
    print("\n" + "-" * 80)
    print("SOURCE NORMALIZATION")
    print("-" * 80)
    print(
        f"  Sphere volume (r={results['source_radius_mm']}mm): {source_norm['sphere_volume_mm3']:.4f} mm³"
    )
    print(
        f"  MCX source volume ({source_norm['n_source_voxels']} voxels): {source_norm['mcx_source_volume_mm3']:.4f} mm³"
    )
    print(f"  Volume ratio (MCX/Sphere): {source_norm['volume_ratio']:.4f}")

    mcx_valid, green_valid, patch_stats = analyze_brightest_patch(mcx, green)

    print("\n" + "-" * 80)
    print("BRIGHTEST PATCH ANALYSIS")
    print("-" * 80)
    print(f"  Patch center: {patch_stats['patch_center']}")
    print(f"  Valid pixels: {patch_stats['n_valid_pixels']}")
    print(f"  MCX mean in patch: {patch_stats['mcx_mean']:.6e}")
    print(f"  Green mean in patch: {patch_stats['green_mean']:.6e}")
    print(f"  Scale ratio (MCX/Green): {patch_stats.get('scale_ratio', 'N/A'):.6e}")
    print(f"  Linear fit slope: {patch_stats.get('slope', 'N/A'):.6f}")
    print(f"  R-squared: {patch_stats.get('r_squared', 'N/A'):.6f}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_path = output_dir / f"{position_id}_scatter.png"
    plot_scatter_comparison(
        mcx_valid, green_valid, patch_stats, scatter_path, position_id
    )

    original_ncc = results.get("ncc", 0)
    original_rmse = results.get("rmse", 0)

    print("\n" + "-" * 80)
    print("ORIGINAL METRICS (from v2)")
    print("-" * 80)
    print(f"  NCC:  {original_ncc:.6f}")
    print(f"  RMSE: {original_rmse:.6f}")

    scale_factor = 1.0 / k_sum if k_sum > 0 else 1.0
    calibrated_ncc, calibrated_rmse = compute_calibrated_metrics(
        mcx, green, scale_factor
    )

    print("\n" + "-" * 80)
    print(f"CALIBRATED METRICS (Green scaled by {scale_factor:.6e})")
    print("-" * 80)
    print(f"  NCC:  {calibrated_ncc:.6f}")
    print(f"  RMSE: {calibrated_rmse:.6f}")

    go_no_go = (
        "GO"
        if 0.8 <= k_sum <= 1.2 and patch_stats.get("r_squared", 0) >= 0.9
        else "NO-GO"
    )

    print("\n" + "=" * 80)
    print(f"GO/NO-GO: {go_no_go}")
    print("=" * 80)
    if go_no_go == "GO":
        print("  ✓ k_sum ∈ [0.8, 1.2]: Magnitude aligned")
        print("  ✓ R² ≥ 0.9: Linear relationship verified")
    else:
        if not (0.8 <= k_sum <= 1.2):
            print(f"  ✗ k_sum = {k_sum:.4f} ∉ [0.8, 1.2]: Magnitude mismatch!")
            print(
                f"    → This suggests source normalization or Lambertian coefficient issue"
            )
        if patch_stats.get("r_squared", 0) < 0.9:
            print(
                f"  ✗ R² = {patch_stats.get('r_squared', 0):.4f} < 0.9: Non-linear relationship!"
            )
            print(f"    → This suggests fundamental physics mismatch")

    diagnosis = {
        "position_id": position_id,
        "mcx_stats": mcx_stats,
        "green_stats": green_stats,
        "scale_factors": {
            "k_sum": float(k_sum),
            "k_max": float(k_max),
            "k_mean": float(k_mean),
        },
        "source_normalization": source_norm,
        "patch_analysis": patch_stats,
        "original_metrics": {
            "ncc": float(original_ncc),
            "rmse": float(original_rmse),
        },
        "calibrated_metrics": {
            "ncc": float(calibrated_ncc),
            "rmse": float(calibrated_rmse),
            "scale_factor": float(scale_factor),
        },
        "go_no_go": go_no_go,
    }

    diagnosis_path = output_dir / f"{position_id}_diagnosis.json"
    with open(diagnosis_path, "w") as f:
        json.dump(diagnosis, f, indent=2)
    logger.info(f"Saved diagnosis: {diagnosis_path}")

    return diagnosis


def run_all_positions():
    """Run magnitude diagnostic on all 5 positions."""
    positions = [
        "S2-Vol-P1-dorsal-r2.0",
        "S2-Vol-P2-left-r2.0",
        "S2-Vol-P3-right-r2.0",
        "S2-Vol-P4-dorsal-lat-r2.0",
        "S2-Vol-P5-ventral-r2.0",
    ]

    all_diagnoses = []

    for pos_id in positions:
        try:
            diagnosis = run_magnitude_diagnostic(pos_id)
            all_diagnoses.append(diagnosis)
        except Exception as e:
            logger.error(f"Failed to diagnose {pos_id}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY: ALL POSITIONS")
    print("=" * 80)
    print(f"{'Position':<30} {'k_sum':<15} {'R²':<10} {'Go/No-Go':<10}")
    print("-" * 80)
    for d in all_diagnoses:
        k_sum = d["scale_factors"]["k_sum"]
        r_sq = d["patch_analysis"].get("r_squared", 0)
        go = d["go_no_go"]
        print(f"{d['position_id']:<30} {k_sum:<15.4e} {r_sq:<10.4f} {go:<10}")

    output_dir = Path(__file__).parent / "results"
    summary_path = output_dir / "magnitude_calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_diagnoses, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    return all_diagnoses


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) > 1:
        position_id = sys.argv[1]
        run_magnitude_diagnostic(position_id)
    else:
        run_all_positions()
