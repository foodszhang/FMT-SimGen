#!/usr/bin/env python3
"""
FINAL DIAGNOSTIC SUMMARY - GEOMETRY CORRECTIONS REQUIRED

================================================================================
CRITICAL FINDINGS
================================================================================

Both P2 and P5 were placed INSIDE THE LIVER due to Y position being near
the anterior where the liver is very superficial.

P5 (ventral):
- Original Y=2.4mm: liver 1.2mm below ventral → 4mm depth = INSIDE liver
- Corrected Y=10.0mm: liver deeper → 4mm depth = soft_tissue ✓
- Result: NCC improved from 0.23 → 0.84

P2 (left side):
- Original Y=2.4mm: source at X=-8mm is INSIDE liver
- P3 (right side) is symmetric but different tissue → k differs by 28×

================================================================================
ROOT CAUSE
================================================================================

The Y position (y_center = 2.4mm) is hardcoded in run_stage2_multiposition_v2.py
(line 404). This position is near the anterior end of the mouse trunk where:

1. The liver is very superficial (close to ventral surface)
2. A 4mm depth from ventral penetrates INTO the liver
3. The left side at this Y level has liver tissue

================================================================================
CORRECTIVE ACTIONS
================================================================================

1. For P5 (ventral):
   - Change Y from 2.4mm to 10.0mm or deeper
   - Re-run MCX + Green projection
   - New NCC = 0.84 (acceptable)

2. For P2 (left):
   - Need to verify correct left_x at a different Y position
   - Or accept the current result with heterogeneity explanation

3. For all positions:
   - Use Y position based on trunk center where anatomy is more uniform
   - Or verify source label before running MCX

================================================================================
UPDATED RESULTS
================================================================================

| Position | Original Y | Issue | Fix | New NCC |
|----------|------------|-------|-----|---------|
| P1-dorsal | 2.4mm | None | - | 0.97 |
| P2-left | 2.4mm | Source in liver | Need Y adjustment | TBD |
| P3-right | 2.4mm | None | - | 0.91 |
| P4-dorsal-lat | 2.4mm | None | - | 0.94 |
| P5-ventral | 2.4mm | Source in liver | Y=10mm | 0.84 |

================================================================================
IMPLICATIONS FOR PAPER
================================================================================

§4.H Failure Mode #2 (Heterogeneity):
- The original P5 "failure" was due to GEOMETRY ERROR, not heterogeneity
- The source was placed INSIDE liver, not 4mm below ventral surface
- After correction, P5 achieves NCC=0.84 (acceptable)
- This is still lower than P1 (0.97), suggesting some heterogeneity effect

The 50% liver path reported earlier was because the source was IN the liver,
not because photons passed through liver to reach the ventral surface.

RECOMMENDATION:
- Re-run all positions with Y=10mm or trunk center
- Then reassess heterogeneity effects based on correct geometry
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_final_summary():
    """Generate final summary."""

    results_base = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"
    p5_corrected_dir = Path(__file__).parent / "results" / "P5_corrected"

    print("=" * 80)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 80)

    positions = [
        ("S2-Vol-P1-dorsal-r2.0", "P1-dorsal", 0),
        ("S2-Vol-P2-left-r2.0", "P2-left", 90),
        ("S2-Vol-P3-right-r2.0", "P3-right", -90),
        ("S2-Vol-P4-dorsal-lat-r2.0", "P4-dorsal-lat", -30),
    ]

    print("\n[1] ORIGINAL STAGE 2 v2 RESULTS")
    print("-" * 80)
    print(f"{'Position':<20} {'NCC':<10} {'k_sum':<15} {'Status':<20}")
    print("-" * 80)

    for pos_id, name, angle in positions:
        with open(results_base / pos_id / "results.json") as f:
            results = json.load(f)

        mcx = np.load(results_base / pos_id / f"mcx_a{angle}.npy")
        green = np.load(results_base / pos_id / f"green_a{angle}.npy")
        k = mcx.sum() / green.sum()

        if results["ncc"] >= 0.9:
            status = "✓ PASS"
        elif results["ncc"] >= 0.8:
            status = "~ ACCEPTABLE"
        else:
            status = "✗ FAIL"

        print(f"{name:<20} {results['ncc']:<10.4f} {k:<15.4e} {status:<20}")

    # P5 original
    with open(results_base / "S2-Vol-P5-ventral-r2.0" / "results.json") as f:
        p5_orig = json.load(f)
    mcx_p5 = np.load(results_base / "S2-Vol-P5-ventral-r2.0" / "mcx_a60.npy")
    green_p5 = np.load(results_base / "S2-Vol-P5-ventral-r2.0" / "green_a60.npy")
    k_p5_orig = mcx_p5.sum() / green_p5.sum()

    print(
        f"{'P5-ventral (orig)':<20} {p5_orig['ncc']:<10.4f} {k_p5_orig:<15.4e} {'✗ FAIL (in liver)':<20}"
    )

    # P5 corrected
    if p5_corrected_dir.exists():
        with open(p5_corrected_dir / "results.json") as f:
            p5_corr = json.load(f)
        print(
            f"{'P5-ventral (fixed)':<20} {p5_corr['ncc']:<10.4f} {p5_corr['k_sum']:<15.4e} {'~ ACCEPTABLE':<20}"
        )

    print("\n[2] GEOMETRY ISSUES IDENTIFIED")
    print("-" * 80)
    print("• P2-left: Source at X=-8mm is INSIDE liver")
    print("• P5-ventral (original): Source at Z=-3.8mm is INSIDE liver")
    print("  - Root cause: Y=2.4mm is near anterior where liver is superficial")
    print("  - Fix: Use Y=10mm or trunk center")

    print("\n[3] P5 CORRECTION RESULTS")
    print("-" * 80)
    if p5_corrected_dir.exists():
        print(f"  Original NCC: 0.23 → Corrected NCC: {p5_corr['ncc']:.4f}")
        print(f"  Original k:   1.09e4 → Corrected k:   {p5_corr['k_sum']:.4e}")
        print(f"  Improvement: 3.7× in NCC")

    print("\n[4] RECOMMENDATIONS")
    print("-" * 80)
    print("1. Re-run all positions with Y=10mm or trunk center (Y=50mm)")
    print("2. Verify source label before MCX simulation")
    print("3. Update §4.H: P5 failure was geometry error, not heterogeneity")
    print("4. After correction, P5 still shows some heterogeneity effect (NCC=0.84)")

    output_dir = Path(__file__).parent / "results"
    summary = {
        "p5_original": {
            "ncc": float(p5_orig["ncc"]),
            "k": float(k_p5_orig),
            "issue": "source in liver",
        },
        "p5_corrected": {
            "ncc": float(p5_corr["ncc"]) if p5_corrected_dir.exists() else None,
            "k": float(p5_corr["k_sum"]) if p5_corrected_dir.exists() else None,
        },
        "p2_issue": "source in liver at Y=2.4mm",
        "recommendation": "use Y=10mm or trunk center for all positions",
    }

    with open(output_dir / "geometry_diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir / 'geometry_diagnostic_summary.json'}")


if __name__ == "__main__":
    generate_final_summary()
