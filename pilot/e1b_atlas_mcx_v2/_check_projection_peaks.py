#!/usr/bin/env python3
"""Check projection peaks for S1-D2mm to diagnose ±90° NCC issue."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmt_simgen.mcx_projection import project_volume_reference


def main():
    results_dir = Path(__file__).parent / "results" / "stage1" / "S1-D2mm"
    comparison_path = results_dir / "comparison.npz"

    if not comparison_path.exists():
        print(f"Error: {comparison_path} not found")
        return

    data = np.load(comparison_path)
    fluence_mcx = data["fluence_mcx"]
    fluence_green = data["fluence_green"]

    # Normalize
    fluence_mcx_norm = fluence_mcx / fluence_mcx.max()
    fluence_green_norm = fluence_green / fluence_green.max()

    # Config
    angles = [-90, -60, -30, 0, 30, 60, 90]
    detector_res = (256, 256)
    fov_mm = 50
    camera_dist = 200
    voxel_size = 0.2

    print("=" * 70)
    print("S1-D2mm Projection Peak Analysis")
    print("=" * 70)
    print(f"\nFluence volumes:")
    print(f"  MCX:   max={fluence_mcx.max():.4e}, sum={fluence_mcx.sum():.4e}")
    print(f"  Green: max={fluence_green.max():.4e}, sum={fluence_green.sum():.4e}")
    print()

    print(
        f"{'Angle':>8} | {'MCX Peak':>12} | {'Green Peak':>12} | {'MCX Sum':>12} | {'Green Sum':>12}"
    )
    print("-" * 70)

    for angle in angles:
        proj_mcx, _ = project_volume_reference(
            fluence_mcx_norm,
            angle_deg=float(angle),
            camera_distance=camera_dist,
            fov_mm=fov_mm,
            detector_resolution=detector_res,
            voxel_size_mm=voxel_size,
        )

        proj_green, _ = project_volume_reference(
            fluence_green_norm,
            angle_deg=float(angle),
            camera_distance=camera_dist,
            fov_mm=fov_mm,
            detector_resolution=detector_res,
            voxel_size_mm=voxel_size,
        )

        print(
            f"{angle:>8} | {proj_mcx.max():>12.4e} | {proj_green.max():>12.4e} | "
            f"{proj_mcx.sum():>12.4e} | {proj_green.sum():>12.4e}"
        )

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)

    # Check if ±90° peaks are significantly lower
    proj_0 = project_volume_reference(
        fluence_mcx_norm, 0, camera_dist, fov_mm, detector_res, voxel_size
    )[0]
    peak_0 = proj_0.max()

    proj_90 = project_volume_reference(
        fluence_mcx_norm, 90, camera_dist, fov_mm, detector_res, voxel_size
    )[0]
    peak_90 = proj_90.max()

    proj_m90 = project_volume_reference(
        fluence_mcx_norm, -90, camera_dist, fov_mm, detector_res, voxel_size
    )[0]
    peak_m90 = proj_m90.max()

    ratio_90 = peak_90 / peak_0 if peak_0 > 0 else 0
    ratio_m90 = peak_m90 / peak_0 if peak_0 > 0 else 0

    print(f"\nPeak at 0°:   {peak_0:.4e}")
    print(f"Peak at 90°:  {peak_90:.4e}  (ratio to 0°: {ratio_90:.4f})")
    print(f"Peak at -90°: {peak_m90:.4e} (ratio to 0°: {ratio_m90:.4f})")

    if ratio_90 < 0.01 or ratio_m90 < 0.01:
        print("\n🔴 CONCLUSION: ±90° peaks are >100x weaker than 0°")
        print("   This is NOT a bug - it's physical (shallow source viewed from side)")
        print("   Recommendation: Exclude ±90° from NCC calculation")
    elif ratio_90 < 0.1 or ratio_m90 < 0.1:
        print("\n🟡 CONCLUSION: ±90° peaks are 10-100x weaker than 0°")
        print("   Low NCC is due to weak signal, not calculation error")
        print("   Recommendation: Exclude ±90° from main analysis")
    else:
        print("\n🟢 CONCLUSION: ±90° peaks are comparable to 0°")
        print("   Low NCC suggests a real calculation issue")


if __name__ == "__main__":
    main()
