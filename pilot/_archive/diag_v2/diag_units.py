#!/usr/bin/env python3
"""Deep dive into MCX vs Green unit calibration.

MCX outputs fluence in units:
  - photon weight per voxel volume, normalized by total photon count
  - For n_photons photons: fluence_mc = N_detected / (V_voxel × n_photons)
  - Units: 1/mm² (normalized by photon count)

Green function outputs:
  - G(r) = exp(-μ_eff × r) / (4π × D × r)
  - For unit source (S=1), G gives fluence in 1/mm²
  - For source strength α: fluence_green = α × G

The key: MCX pattern3d source has:
  - Pattern values p_i (typically 1.0 for uniform)
  - Each voxel emits photons proportional to p_i
  - Total source strength S = sum(p_i) × V_voxel × n_photons

To match:
  - MCX fluence needs to be multiplied by n_photons to get absolute fluence
  - Or Green needs to be scaled by (pattern_sum × V_voxel × n_photons)

This script analyzes the full chain of normalization.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"

N_PHOTONS = 1e8
VOXEL_SIZE_MM = 0.4


def load_position_data(position_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load MCX projection, Green projection, and results."""
    pos_dir = RESULTS_BASE / position_id

    mcx_path = green_path = None
    for f in pos_dir.iterdir():
        if f.name.startswith("mcx_a") and f.suffix == ".npy":
            mcx_path = f
        elif f.name.startswith("green_a") and f.suffix == ".npy":
            green_path = f

    mcx = np.load(mcx_path)
    green = np.load(green_path)

    with open(pos_dir / "results.json") as f:
        results = json.load(f)

    pattern_path = pos_dir / f"source-{position_id}.bin"
    pattern = None
    if pattern_path.exists():
        pattern = np.fromfile(pattern_path, dtype=np.float32)
        logger.info(f"Loaded pattern: {pattern.shape}, sum={pattern.sum():.4f}")

    return mcx, green, results, pattern


def analyze_normalization_chain(position_id: str):
    """Analyze the full normalization chain."""
    print("\n" + "=" * 80)
    print(f"NORMALIZATION CHAIN ANALYSIS: {position_id}")
    print("=" * 80)

    mcx, green, results, pattern = load_position_data(position_id)

    n_source_voxels = results.get("n_source_voxels", 536)
    voxel_volume = VOXEL_SIZE_MM**3
    source_radius = results.get("source_radius_mm", 2.0)

    sphere_volume = 4.0 / 3.0 * np.pi * source_radius**3

    print("\n[1] SOURCE GEOMETRY")
    print("-" * 80)
    print(f"  Source radius: {source_radius} mm")
    print(f"  Sphere volume: {sphere_volume:.4f} mm³")
    print(f"  MCX source voxels: {n_source_voxels}")
    print(f"  MCX source volume: {n_source_voxels * voxel_volume:.4f} mm³")
    print(f"  Voxel size: {VOXEL_SIZE_MM} mm")
    print(f"  Voxel volume: {voxel_volume:.4f} mm³")

    if pattern is not None:
        pattern_sum = pattern.sum()
        print(f"\n  Pattern sum: {pattern_sum:.4f}")
        print(f"  Pattern mean: {pattern.mean():.4f}")
        print(f"  Pattern max: {pattern.max():.4f}")
        print(f"  Pattern nonzero: {np.sum(pattern > 0)}")

    print("\n[2] MCX NORMALIZATION")
    print("-" * 80)
    print(f"  n_photons: {N_PHOTONS:.0e}")
    print(f"  MCX fluence unit: photon_weight / (V_voxel × n_photons)")

    mcx_sum = mcx.sum()
    mcx_max = mcx.max()
    mcx_mean_nonzero = mcx[mcx > 0].mean() if np.any(mcx > 0) else 0

    print(f"\n  MCX projection sum: {mcx_sum:.6e}")
    print(f"  MCX projection max: {mcx_max:.6e}")
    print(f"  MCX projection mean (nonzero): {mcx_mean_nonzero:.6e}")

    mcx_abs_sum = mcx_sum * N_PHOTONS
    mcx_abs_max = mcx_max * N_PHOTONS
    print(f"\n  MCX × n_photons (absolute fluence):")
    print(f"    sum: {mcx_abs_sum:.6e}")
    print(f"    max: {mcx_abs_max:.6e}")

    print("\n[3] GREEN FUNCTION NORMALIZATION")
    print("-" * 80)
    print(f"  Green function G(r) = exp(-μ_eff × r) / (4π × D × r)")
    print(f"  Unit source (α=1), output in 1/mm²")

    green_sum = green.sum()
    green_max = green.max()
    green_mean_nonzero = green[green > 0].mean() if np.any(green > 0) else 0

    print(f"\n  Green projection sum: {green_sum:.6e}")
    print(f"  Green projection max: {green_max:.6e}")
    print(f"  Green projection mean (nonzero): {green_mean_nonzero:.6e}")

    alpha = 1.0
    green_total_source = alpha * sphere_volume
    print(f"\n  For unit source α=1 over sphere:")
    print(f"    Total source strength: {green_total_source:.4f} mm³")

    print("\n[4] SCALE FACTOR ANALYSIS")
    print("-" * 80)

    k_raw = mcx_sum / green_sum if green_sum > 0 else float("inf")
    print(f"  k_raw = MCX_sum / Green_sum = {k_raw:.6e}")

    k_absolute = mcx_abs_sum / green_sum if green_sum > 0 else float("inf")
    print(f"  k_absolute = (MCX_sum × n_photons) / Green_sum = {k_absolute:.6e}")

    expected_k = N_PHOTONS * n_source_voxels * voxel_volume / sphere_volume
    print(f"\n  Expected scale factor (theory):")
    print(f"    k_expected = n_photons × V_source / V_sphere")
    print(
        f"    k_expected = {N_PHOTONS:.0e} × {n_source_voxels * voxel_volume:.4f} / {sphere_volume:.4f}"
    )
    print(f"    k_expected = {expected_k:.6e}")

    print("\n[5] HYPOTHESIS TESTING")
    print("-" * 80)

    print("  Hypothesis 1: MCX already includes photon count in output")
    print(f"    → Check if k_raw ≈ 1: {k_raw:.4e} (NO, factor {k_raw:.0f})")

    print("\n  Hypothesis 2: MCX needs × n_photons, Green needs × V_source")
    print(
        f"    → Check if k_absolute ≈ V_source: {k_absolute:.4e} vs {n_source_voxels * voxel_volume:.4f}"
    )
    print(f"    → Ratio: {k_absolute / (n_source_voxels * voxel_volume):.4f}")

    print("\n  Hypothesis 3: MCX projection picks surface voxels, not integrated")
    print("    → MCX projection: max-norm comparison hides absolute value")
    print("    → Green projection: integrated surface fluence")

    print("\n[6] PIXEL-LEVEL COMPARISON (brightest region)")
    print("-" * 80)

    combined = mcx + green * k_raw
    max_idx = np.unravel_index(np.argmax(combined), combined.shape)
    cy, cx = max_idx

    patch_size = 10
    h, w = mcx.shape
    y0, y1 = max(0, cy - patch_size // 2), min(h, cy + patch_size // 2)
    x0, x1 = max(0, cx - patch_size // 2), min(w, cx + patch_size // 2)

    mcx_patch = mcx[y0:y1, x0:x1].flatten()
    green_patch = green[y0:y1, x0:x1].flatten()

    mask = (mcx_patch > 0) & (green_patch > 0)
    mcx_v = mcx_patch[mask]
    green_v = green_patch[mask]

    if len(mcx_v) > 10:
        mcx_scaled = mcx_v * 1.0
        green_scaled = green_v * k_raw

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.scatter(mcx_v, green_v, alpha=0.5, s=10)
        ax.set_xlabel("MCX (raw)")
        ax.set_ylabel("Green (raw)")
        ax.set_title(f"Raw values\nk={k_raw:.2e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(mcx_v, green_v * k_raw, alpha=0.5, s=10)
        max_val = max(mcx_v.max(), (green_v * k_raw).max())
        ax.plot([0, max_val], [0, max_val], "r--", label="1:1")
        ax.set_xlabel("MCX (raw)")
        ax.set_ylabel(f"Green × k_raw")
        ax.set_title("After scaling Green")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ratio = mcx_v / (green_v + 1e-30)
        ax.hist(ratio, bins=50, edgecolor="black")
        ax.set_xlabel("MCX / Green")
        ax.set_ylabel("Count")
        ax.set_title(f"Pixel ratio distribution\nmedian={np.median(ratio):.2e}")
        ax.axvline(k_raw, color="r", linestyle="--", label=f"k_raw={k_raw:.2e}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{position_id}_unit_analysis.png", dpi=150)
        plt.close()
        logger.info(
            f"Saved unit analysis figure: {output_dir / f'{position_id}_unit_analysis.png'}"
        )

        print(f"  Patch center: ({cy}, {cx})")
        print(f"  Valid pixels: {len(mcx_v)}")
        print(f"  MCX range: [{mcx_v.min():.4e}, {mcx_v.max():.4e}]")
        print(f"  Green range: [{green_v.min():.4e}, {green_v.max():.4e}]")
        print(f"  Pixel ratio median: {np.median(ratio):.4e}")
        print(f"  Pixel ratio std: {np.std(ratio):.4e}")

    diagnosis = {
        "position_id": position_id,
        "source_geometry": {
            "radius_mm": source_radius,
            "sphere_volume_mm3": sphere_volume,
            "n_source_voxels": n_source_voxels,
            "mcx_source_volume_mm3": n_source_voxels * voxel_volume,
        },
        "mcx_stats": {
            "sum": float(mcx_sum),
            "max": float(mcx_max),
            "mean_nonzero": float(mcx_mean_nonzero),
        },
        "green_stats": {
            "sum": float(green_sum),
            "max": float(green_max),
            "mean_nonzero": float(green_mean_nonzero),
        },
        "scale_factors": {
            "k_raw": float(k_raw),
            "k_absolute": float(k_absolute),
            "k_expected_theory": float(expected_k),
        },
    }

    return diagnosis


def analyze_all_positions():
    """Analyze normalization for all positions."""
    positions = [
        "S2-Vol-P1-dorsal-r2.0",
        "S2-Vol-P2-left-r2.0",
        "S2-Vol-P3-right-r2.0",
        "S2-Vol-P4-dorsal-lat-r2.0",
        "S2-Vol-P5-ventral-r2.0",
    ]

    all_diagnoses = []
    for pos_id in positions:
        diag = analyze_normalization_chain(pos_id)
        all_diagnoses.append(diag)

    print("\n" + "=" * 80)
    print("SUMMARY: SCALE FACTORS")
    print("=" * 80)
    print(f"{'Position':<30} {'k_raw':<15} {'k_absolute':<15} {'k_expected':<15}")
    print("-" * 80)
    for d in all_diagnoses:
        k_raw = d["scale_factors"]["k_raw"]
        k_abs = d["scale_factors"]["k_absolute"]
        k_exp = d["scale_factors"]["k_expected_theory"]
        print(f"{d['position_id']:<30} {k_raw:<15.4e} {k_abs:<15.4e} {k_exp:<15.4e}")

    output_dir = Path(__file__).parent / "results"
    with open(output_dir / "normalization_chain_analysis.json", "w") as f:
        json.dump(all_diagnoses, f, indent=2)

    return all_diagnoses


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) > 1:
        position_id = sys.argv[1]
        analyze_normalization_chain(position_id)
    else:
        analyze_all_positions()
