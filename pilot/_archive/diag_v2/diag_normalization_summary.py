#!/usr/bin/env python3
"""
SUMMARY: MCX vs Green Scale Factor Analysis

================================================================================
OBSERVATION
================================================================================

For P1-dorsal (cleanest position):
  k_observed = MCX_sum / Green_sum = 1.21e7
  k_max = MCX_max / Green_max = 1.62e7

For all positions, k varies from 1.1e4 to 1.3e7 (3 orders of magnitude).

================================================================================
ROOT CAUSE ANALYSIS
================================================================================

1. MCX Output Units:
   - JNII header: "Fluence rate (W/mm²)"
   - MCX computes: fluence = pathlength_sum / (V_voxel × T × n_photons)
   - This is normalized by n_photons, so output represents fluence per unit photon

2. MCX Pattern3d Normalization:
   - Total photons = n_photons = 1e8
   - Distributed by pattern_value / pattern_sum
   - For uniform pattern (all 1s): each voxel emits n_photons / pattern_sum photons
   - pattern_sum = 536 for P1

3. Green Function:
   - G(r) = exp(-μ_eff × r) / (4π × D × r)
   - Unit source (α = 1)
   - 7-point cubature with weights summing to 1.0

4. The Scale Factor:
   - MCX: discrete source over pattern_sum voxels, normalized by n_photons
   - Green: continuous source over sphere volume, unit strength

   The mismatch comes from:
   a) MCX normalizes by n_photons (makes output ~independent of photon count)
   b) MCX pattern distribution affects local source density
   c) Green cubature approximates volume integral with sum of point sources

================================================================================
EMPIRICAL FINDINGS
================================================================================

Position              k_sum        k_max        Notes
--------------------------------------------------------------------------------
P1-dorsal             1.21e7       1.62e7       Cleanest, dorsal surface
P2-left               4.76e5       1.15e6       Lateral, curvature effects
P3-right              1.32e7       2.92e7       Lateral, asymmetry
P4-dorsal-lat         4.32e6       6.14e6       Oblique angle
P5-ventral            1.09e4       2.01e4       MUCH smaller - heterogeneity?

The 3-order-of-magnitude variation in k suggests:
1. MCX and Green are computing different physical quantities
2. NCC is misleading because both are max-normalized before comparison
3. P5's small k is consistent with heterogeneity reducing MCX fluence

================================================================================
CONCLUSIONS
================================================================================

1. NCC is NOT a reliable metric for absolute magnitude comparison
   - Both MCX and Green are max-normalized in the comparison
   - This hides the fundamental unit mismatch

2. The RMSE values are meaningless without proper calibration
   - RMSE spans 4 orders of magnitude (0.0007 to 6771)
   - This reflects the unit mismatch, not physical accuracy

3. For P5-ventral:
   - k = 1.09e4 is 1000× smaller than P1
   - This is consistent with high-absorption tissue (liver) in the path
   - MCX correctly models heterogeneity, Green assumes homogeneous medium

4. The "good" NCC for P1-P4 is due to:
   - Similar spatial distribution (both follow exp(-μ_eff × r) / r decay)
   - Max-normalization removes absolute magnitude differences
   - NOT because they agree on absolute fluence values

================================================================================
RECOMMENDATIONS
================================================================================

1. For Stage 2 validation:
   - Accept that MCX and Green compute different quantities
   - Use NCC as a shape-similarity metric only
   - Document the scale factor mismatch as expected behavior

2. For P5 failure analysis:
   - The small k factor (1.09e4) is evidence of heterogeneity
   - Run "homogeneous MCX" counterfactual to confirm
   - This will be the key evidence for §4.H in the paper

3. For future work:
   - Calibrate MCX output to physical units (W/mm²)
   - Scale Green output to match MCX's photon normalization
   - Or use relative comparisons throughout
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def generate_summary_report():
    """Generate a summary report of the normalization analysis."""

    results_base = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"

    positions = [
        ("S2-Vol-P1-dorsal-r2.0", "Dorsal"),
        ("S2-Vol-P2-left-r2.0", "Left"),
        ("S2-Vol-P3-right-r2.0", "Right"),
        ("S2-Vol-P4-dorsal-lat-r2.0", "Dorsal-Lateral"),
        ("S2-Vol-P5-ventral-r2.0", "Ventral"),
    ]

    print("=" * 80)
    print("STAGE 2 v2 NORMALIZATION ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n[1] SCALE FACTORS (MCX / Green)")
    print("-" * 80)
    print(f"{'Position':<25} {'k_sum':<15} {'k_max':<15} {'NCC':<10} {'RMSE':<12}")
    print("-" * 80)

    data = []
    for pos_id, name in positions:
        mcx = np.load(
            results_base
            / pos_id
            / f"mcx_a{int({'Dorsal': 0, 'Left': 90, 'Right': -90, 'Dorsal-Lateral': -30, 'Ventral': 60}[name])}.npy"
        )
        green = np.load(
            results_base
            / pos_id
            / f"green_a{int({'Dorsal': 0, 'Left': 90, 'Right': -90, 'Dorsal-Lateral': -30, 'Ventral': 60}[name])}.npy"
        )

        with open(results_base / pos_id / "results.json") as f:
            results = json.load(f)

        k_sum = mcx.sum() / green.sum()
        k_max = mcx.max() / green.max()
        ncc = results["ncc"]
        rmse = results["rmse"]

        print(f"{name:<25} {k_sum:<15.4e} {k_max:<15.4e} {ncc:<10.4f} {rmse:<12.4f}")

        data.append(
            {
                "position": name,
                "k_sum": float(k_sum),
                "k_max": float(k_max),
                "ncc": float(ncc),
                "rmse": float(rmse),
            }
        )

    print("\n[2] KEY OBSERVATIONS")
    print("-" * 80)
    print("• k_sum varies by 3 orders of magnitude (1.1e4 to 1.3e7)")
    print("• P5-ventral has k_sum = 1.09e4, 1000× smaller than P1")
    print("• NCC is high for P1-P4 (0.90-0.97) despite k mismatch")
    print("• P5 has low NCC (0.23) AND small k")

    print("\n[3] INTERPRETATION")
    print("-" * 80)
    print("• NCC measures shape similarity, not absolute magnitude")
    print("• Max-normalization hides the unit mismatch")
    print("• P5's small k suggests heterogeneity reducing MCX fluence")
    print("• P5's low NCC suggests spatial distribution mismatch")

    print("\n[4] NEXT STEPS")
    print("-" * 80)
    print("• Run P5 tissue path analysis (Main Line B)")
    print("• Run homogeneous MCX counterfactual for P5")
    print("• Document findings for Paper §4.H")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "normalization_summary.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved: {output_dir / 'normalization_summary.json'}")

    return data


if __name__ == "__main__":
    generate_summary_report()
