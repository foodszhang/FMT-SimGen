"""R-1: Revert fluence>0 mask, use projection-space threshold.

Tests P5-ventral @ +150° with:
1. atlas > 0 (correct semantic body mask)
2. MCX projection threshold (proj_mcx > thr)

Compares against c15abd3's 0.9212 result.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.metrics import ncc
from shared.green_surface_projection import render_green_surface_projection
from fmt_simgen.mcx_projection import project_volume_reference

VOXEL_SIZE_MM = 0.4
ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/projection_fix")

GT_POS = np.array([-0.6, 2.4, -3.8])
BEST_ANGLE = 150


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R-1: Revert fluence>0 mask, use projection-space threshold")
    print("=" * 70)

    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))
    fluence = np.load(ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"\nAtlas shape: {atlas.shape}")
    print(f"Fluence shape: {fluence.shape}")
    print(f"GT position: {GT_POS}")
    print(f"Angle: {BEST_ANGLE}°")

    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    proj_mcx, _ = project_volume_reference(
        fluence,
        BEST_ANGLE,
        voxel_size_mm=VOXEL_SIZE_MM,
        camera_distance=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
    )

    proj_green = render_green_surface_projection(
        GT_POS,
        atlas > 0,
        BEST_ANGLE,
        camera_distance_mm=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
        tissue_params=tissue_params,
        voxel_size_mm=VOXEL_SIZE_MM,
    )

    mcx_max = proj_mcx.max()
    print(f"\nMCX projection max: {mcx_max:.4e}")

    print("\n" + "=" * 70)
    print("Threshold scan")
    print("=" * 70)

    thresholds = [1e-3, 1e-5, 1e-7]
    results = []

    print(f"\n{'Threshold':<15} {'n_valid':>10} {'linear_NCC':>12}")
    print("-" * 40)

    for thr_factor in thresholds:
        mcx_thr = mcx_max * thr_factor
        valid = (proj_mcx > mcx_thr) & (proj_green > 0)
        n_valid = int(np.sum(valid))

        if n_valid < 100:
            linear_ncc = np.nan
            print(f"max * {thr_factor:<8.0e} {n_valid:>10} {'N/A':>12}")
        else:
            linear_ncc = ncc(proj_mcx[valid], proj_green[valid])
            print(f"max * {thr_factor:<8.0e} {n_valid:>10} {linear_ncc:>12.4f}")

        results.append(
            {
                "thr_factor": thr_factor,
                "mcx_thr": mcx_thr,
                "n_valid": n_valid,
                "linear_ncc": linear_ncc,
            }
        )

    print("\n" + "=" * 70)
    print("Comparison with c15abd3")
    print("=" * 70)

    print(f"\nc15abd3 (fluence>0 mask, no threshold): 0.9212")
    print(f"R-1 (atlas>0 + thr=max*1e-5): {results[1]['linear_ncc']:.4f}")
    print(f"R-1 (atlas>0 + thr=max*1e-7): {results[2]['linear_ncc']:.4f}")

    valid_green_only = proj_green > 0
    n_valid_green = int(np.sum(valid_green_only))
    if n_valid_green >= 100:
        linear_ncc_green_only = ncc(
            proj_mcx[valid_green_only], proj_green[valid_green_only]
        )
        print(f"R-1 (atlas>0 + no MCX threshold): {linear_ncc_green_only:.4f}")

    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)

    best_ncc = max(r["linear_ncc"] for r in results if not np.isnan(r["linear_ncc"]))
    if best_ncc >= 0.9:
        print(f"\n✓ R-1 NCC = {best_ncc:.4f} ≥ 0.9")
        print(
            "Forward model correct. c15abd3 just moved mask definition to fluence domain."
        )
        verdict = "PASS with threshold"
    elif best_ncc >= 0.8:
        print(f"\n⚠ R-1 best NCC = {best_ncc:.4f} (0.8 ≤ NCC < 0.9)")
        print(
            "Forward model partially correct, threshold sensitivity indicates noise issue."
        )
        verdict = "PARTIAL"
    else:
        print(f"\n✗ R-1 NCC = {best_ncc:.4f} < 0.8")
        print("c15abd3's 0.9212 is mask-tampering artifact. True NCC cannot reach 0.9.")
        verdict = "FAIL"

    report = f"""

## R-1: Revert fluence>0, threshold in projection space

- c15abd3 result (fluence>0 mask, no threshold): 0.9212
- R-1 result (atlas>0 + thr=max*1e-5): {results[1]["linear_ncc"]:.4f}
- R-1 result (atlas>0 + thr=max*1e-7): {results[2]["linear_ncc"]:.4f}
- R-1 result (atlas>0 + no MCX threshold): {linear_ncc_green_only:.4f}

### Threshold scan

| thr | mcx_thr | n_valid | linear_NCC |
|-----|---------|---------|------------|
| max * 1e-3 | {results[0]["mcx_thr"]:.4e} | {results[0]["n_valid"]} | {results[0]["linear_ncc"]:.4f} |
| max * 1e-5 | {results[1]["mcx_thr"]:.4e} | {results[1]["n_valid"]} | {results[1]["linear_ncc"]:.4f} |
| max * 1e-7 | {results[2]["mcx_thr"]:.4e} | {results[2]["n_valid"]} | {results[2]["linear_ncc"]:.4f} |

- Verdict: **{verdict}**
"""

    with open(OUTPUT_DIR / "REPORT.md", "a") as f:
        f.write(report)

    print(f"\nReport appended to: {OUTPUT_DIR / 'REPORT.md'}")

    return results


if __name__ == "__main__":
    main()
