#!/usr/bin/env python3
"""Compute expected scale factor between MCX and Green based on physics.

MCX output (from JNII header): Fluence rate (W/mm²)

For MCX Monte Carlo simulation:
- Each photon carries weight = 1 (unitless)
- Total photons launched = n_photons
- Time gate = T (seconds)
- Fluence = accumulated_weight / (V_voxel × T)
- Units: 1 / (mm³ × s) = 1 / (mm²) × (1 / (mm × s))

For pattern3d source:
- Pattern values determine spatial distribution
- Total source strength = sum(pattern) in MCX's internal units
- MCX launches n_photons photons distributed by pattern

The key normalization:
- MCX output represents fluence for source = n_photons photons over time T
- Equivalent to photon flux = n_photons / T

For Green function:
- G(r) = exp(-μ_eff × r) / (4π × D × r)
- This is fluence for unit source (α = 1 photon/s or 1 W)

Expected scale:
- MCX source flux = n_photons / T
- Green source flux = 1 (unit)
- k_expected = n_photons / T

But wait - MCX normalizes by n_photons internally!
Let me verify by checking the actual output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"

N_PHOTONS = 1e8
TIME_GATE = 5e-8  # seconds
VOXEL_SIZE = 0.4  # mm


def analyze_scale_factor():
    """Analyze the expected vs observed scale factor."""

    # Load P1 data
    mcx_proj = np.load(RESULTS_BASE / "S2-Vol-P1-dorsal-r2.0" / "mcx_a0.npy")
    green_proj = np.load(RESULTS_BASE / "S2-Vol-P1-dorsal-r2.0" / "green_a0.npy")
    fluence = np.load(RESULTS_BASE / "S2-Vol-P1-dorsal-r2.0" / "fluence.npy")

    # Load pattern
    pattern = np.fromfile(
        RESULTS_BASE / "S2-Vol-P1-dorsal-r2.0" / "source-S2-Vol-P1-dorsal-r2.0.bin",
        dtype=np.float32,
    )

    print("=" * 80)
    print("MCX vs GREEN SCALE FACTOR ANALYSIS")
    print("=" * 80)

    print("\n[1] MCX SIMULATION PARAMETERS")
    print("-" * 80)
    print(f"  n_photons = {N_PHOTONS:.0e}")
    print(f"  time_gate = {TIME_GATE:.2e} s")
    print(f"  voxel_size = {VOXEL_SIZE} mm")
    print(f"  voxel_volume = {VOXEL_SIZE**3:.4f} mm³")

    print("\n[2] PATTERN3D SOURCE")
    print("-" * 80)
    print(f"  pattern_sum = {pattern.sum():.0f}")
    print(f"  pattern_nonzero = {np.sum(pattern > 0)}")
    print(f"  pattern_mean = {pattern.mean():.4f}")

    print("\n[3] MCX OUTPUT INTERPRETATION")
    print("-" * 80)
    print("  MCX fluence = accumulated_weight / (V_voxel × T)")
    print("  where:")
    print("    accumulated_weight = sum of photon weights in voxel")
    print("    V_voxel = voxel volume")
    print("    T = time gate")
    print()
    print("  For pattern3d source:")
    print("    - Total photons launched = n_photons")
    print("    - Distributed by pattern value / pattern_sum")
    print("    - Each source voxel emits: n_photons × (value / pattern_sum)")
    print()
    print(
        f"  Effective source flux = n_photons / T = {N_PHOTONS / TIME_GATE:.2e} photons/s"
    )

    print("\n[4] GREEN FUNCTION INTERPRETATION")
    print("-" * 80)
    print("  G(r) = exp(-μ_eff × r) / (4π × D × r)")
    print("  This is fluence for unit source (α = 1)")
    print()
    print("  For 7-point cubature:")
    print("    - weights sum to 1.0")
    print("    - effective source = α × sum(weights) = 1.0")

    print("\n[5] EXPECTED SCALE FACTOR")
    print("-" * 80)

    # Theory 1: MCX output is normalized by n_photons
    # fluence_mc = (detected / n_photons) / (V_voxel × T)
    # For unit source (detected = n_photons × efficiency):
    # fluence_mc = efficiency / (V_voxel × T)
    # This would give k = 1 (if Green also uses unit source)

    # Theory 2: MCX output is NOT normalized by n_photons
    # fluence_mc = detected / (V_voxel × T)
    # For source = n_photons:
    # fluence_mc = n_photons × efficiency / (V_voxel × T)
    # This would give k = n_photons

    # Theory 3: MCX pattern3d affects normalization
    # The pattern sum might be the effective source strength
    # fluence_mc represents fluence for source = pattern_sum
    # This would give k = pattern_sum

    # Theory 4: MCX uses photon rate normalization
    # fluence_mc = fluence for source_rate = n_photons / T
    # Green uses unit source (1 photon/s)
    # This would give k = n_photons / T

    k_theory1 = 1.0
    k_theory2 = N_PHOTONS
    k_theory3 = pattern.sum()
    k_theory4 = N_PHOTONS / TIME_GATE

    print(f"  Theory 1 (MCX normalized by n_photons): k = {k_theory1:.2e}")
    print(f"  Theory 2 (MCX not normalized): k = {k_theory2:.2e}")
    print(f"  Theory 3 (MCX source = pattern_sum): k = {k_theory3:.0f}")
    print(f"  Theory 4 (MCX source_rate = n_photons/T): k = {k_theory4:.2e}")

    # Observed
    k_observed = mcx_proj.sum() / green_proj.sum()
    k_max = mcx_proj.max() / green_proj.max()

    print(f"\n  OBSERVED:")
    print(f"    k_sum = MCX_sum / Green_sum = {k_observed:.6e}")
    print(f"    k_max = MCX_max / Green_max = {k_max:.6e}")

    print("\n[6] ANALYSIS")
    print("-" * 80)

    # Check which theory is closest
    theories = {
        "Theory 1 (normalized)": k_theory1,
        "Theory 2 (not normalized)": k_theory2,
        "Theory 3 (pattern_sum)": k_theory3,
        "Theory 4 (photon rate)": k_theory4,
    }

    for name, k in theories.items():
        ratio = k_observed / k
        print(f"  {name}:")
        print(f"    k_expected = {k:.6e}")
        print(f"    k_observed / k_expected = {ratio:.6e}")
        if 0.5 < ratio < 2.0:
            print(f"    ✓ CLOSE MATCH!")
        elif 0.1 < ratio < 10.0:
            print(f"    ~ Within factor of 10")
        else:
            print(f"    ✗ Not matching")

    # The observed k is ~1.2e7
    # This is between theory 3 (536) and theory 2 (1e8)
    # Let's check if there's a factor related to the cubature

    # 7-point cubature: 7 points with equal weights = 1/7
    # Total source = sum(weights) = 1.0
    # But the points are at radius × [0, ±0.5, ±0.5, ±0.5]
    # So the cubature approximates the volume integral

    # The volume of a sphere with r=2mm is 33.5 mm³
    # The MCX source volume is 536 × 0.064 = 34.3 mm³
    # These match!

    # So the scale factor might be related to:
    # k = n_photons × (something about the Green normalization)

    # Let me check if k = n_photons / pattern_sum
    k_theory5 = N_PHOTONS / pattern.sum()
    print(f"\n  Theory 5 (n_photons / pattern_sum): k = {k_theory5:.6e}")
    print(f"    k_observed / k_expected = {k_observed / k_theory5:.6e}")

    # Or k = n_photons × T × (something)
    k_theory6 = N_PHOTONS * TIME_GATE
    print(f"\n  Theory 6 (n_photons × T): k = {k_theory6:.6e}")
    print(f"    k_observed / k_expected = {k_observed / k_theory6:.6e}")

    # Or k = n_photons / (pattern_sum × something)
    # k_observed = 1.2e7
    # n_photons = 1e8
    # pattern_sum = 536
    # n_photons / pattern_sum = 1.9e5
    # k_observed / (n_photons / pattern_sum) = 1.2e7 / 1.9e5 = 63

    factor = k_observed / (N_PHOTONS / pattern.sum())
    print(f"\n  Factor analysis:")
    print(f"    k_observed / (n_photons / pattern_sum) = {factor:.2f}")

    # 63 is close to 4π × 5 = 62.8
    # Or 2π² = 19.7 × 3.2 = 63
    # Or just a coincidence

    # Let me check if it's related to the Green function normalization
    # G(r) = exp(-μ_eff × r) / (4π × D × r)
    # The 4π factor comes from spherical symmetry

    mua = 0.08697
    mus_prime = 4.2907
    D = 1.0 / (3.0 * (mua + mus_prime))

    print(f"\n  Green function parameters:")
    print(f"    D = {D:.4f} mm")
    print(f"    4πD = {4 * np.pi * D:.4f} mm")
    print(f"    1/(4πD) = {1 / (4 * np.pi * D):.4f} /mm")

    # The observed scale factor is:
    # k = 1.2e7 = n_photons / pattern_sum × 63
    #    = n_photons / pattern_sum × (something)

    # If we think of MCX output as:
    # fluence_mc = (detected_weight) / (V_voxel × T)
    #            = (detected_photons) / (V_voxel × T)
    #            = (source_photons × efficiency) / (V_voxel × T)
    #            = (n_photons × efficiency) / (V_voxel × T)

    # And Green output as:
    # fluence_green = G(r) for unit source
    #               = efficiency / (4πD × r)  (for large r)

    # Then the ratio should be:
    # k = fluence_mc / fluence_green
    #   = (n_photons × efficiency / (V × T)) / (efficiency / (4πD × r))
    #   = n_photons × 4πD × r / (V × T)

    # For r = 4mm (source depth), V = 0.064 mm³:
    r = 4.0
    V = VOXEL_SIZE**3
    k_theory7 = N_PHOTONS * 4 * np.pi * D * r / (V * TIME_GATE)
    print(f"\n  Theory 7 (full physics):")
    print(f"    k = n_photons × 4πD × r / (V × T)")
    print(
        f"    k = {N_PHOTONS:.0e} × {4 * np.pi * D:.4f} × {r} / ({V:.4f} × {TIME_GATE:.2e})"
    )
    print(f"    k = {k_theory7:.6e}")
    print(f"    k_observed / k_expected = {k_observed / k_theory7:.6e}")

    return k_observed


if __name__ == "__main__":
    analyze_scale_factor()
