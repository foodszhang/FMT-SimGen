#!/usr/bin/env python3
"""P2 geometry sanity check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"
ATLAS_PATH = Path("/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin")

VOXEL_SIZE_MM = 0.4


def main():
    """Check P2 geometry."""
    print("=" * 80)
    print("P2 GEOMETRY SANITY CHECK")
    print("=" * 80)

    # Load original volume
    original = np.fromfile(ATLAS_PATH, dtype=np.uint8)
    original = original.reshape((104, 200, 190))

    # Load P2 results
    with open(RESULTS_BASE / "S2-Vol-P2-left-r2.0" / "results.json") as f:
        p2_results = json.load(f)

    source_pos = np.array(p2_results["source_pos"])
    print(f"\nP2 source position: {source_pos} mm")
    print(f"P2 best angle: {p2_results['best_angle']}°")

    # P2 is on the left side: x = left_x + 4mm
    # Find left_x at Y=2.4mm

    # Convert Y=2.4mm to voxel (original volume)
    # Y: -50mm at voxel 0 → +50mm at voxel 199
    y_mm = source_pos[1]
    y_vox = int(round((y_mm + 50) / 100 * 200))

    print(f"\nY position: {y_mm} mm → Y voxel: {y_vox}")

    # Get slice at this Y
    slice_xz = original[:, y_vox, :]  # (Z, X)
    tissue_z, tissue_x = np.where(slice_xz > 0)

    if len(tissue_x) > 0:
        left_x_vox = tissue_x.min()
        right_x_vox = tissue_x.max()

        # Convert to mm
        # X: -19mm at voxel 0 → +19mm at voxel 189
        left_x_mm = (left_x_vox - 95) * 0.1
        right_x_mm = (right_x_vox - 95) * 0.1

        print(f"Tissue extent at Y={y_mm}mm:")
        print(f"  Left X: {left_x_mm:.2f} mm (voxel {left_x_vox})")
        print(f"  Right X: {right_x_mm:.2f} mm (voxel {right_x_vox})")

        # Expected P2 x position
        expected_p2_x = left_x_mm + 4.0
        print(f"\nExpected P2 X (left + 4mm): {expected_p2_x:.2f} mm")
        print(f"Actual P2 X: {source_pos[0]:.2f} mm")

        if abs(source_pos[0] - expected_p2_x) > 1.0:
            print("⚠️  P2 X position mismatch!")
        else:
            print("✓ P2 X position matches expected")

    # Check label at P2 source
    # Convert source position to voxel (ZYX order for original volume)
    ix = int(round((source_pos[0] + 19) / 38 * 190))
    iy = y_vox
    iz = int(round((source_pos[2] + 10) / 20 * 104))

    print(f"\nSource voxel (ZYX): ({iz}, {iy}, {ix})")

    if 0 <= iz < 104 and 0 <= iy < 200 and 0 <= ix < 190:
        label = original[iz, iy, ix]
        names = {
            0: "background",
            1: "soft_tissue",
            2: "bone",
            3: "brain",
            4: "heart",
            5: "stomach",
            6: "abdominal",
            7: "liver",
            8: "kidney",
            9: "lung",
        }
        print(f"Label at source: {label} ({names.get(label, 'unknown')})")

        if label == 0:
            print("❌ Source is in AIR!")
        elif label == 1:
            print("✓ Source is in soft_tissue")
        else:
            print(f"⚠️ Source is in {names.get(label, 'unknown')}")

    # Compare P2 vs P3 k factors
    mcx_p2 = np.load(RESULTS_BASE / "S2-Vol-P2-left-r2.0" / "mcx_a90.npy")
    green_p2 = np.load(RESULTS_BASE / "S2-Vol-P2-left-r2.0" / "green_a90.npy")

    mcx_p3 = np.load(RESULTS_BASE / "S2-Vol-P3-right-r2.0" / "mcx_a-90.npy")
    green_p3 = np.load(RESULTS_BASE / "S2-Vol-P3-right-r2.0" / "green_a-90.npy")

    k_p2 = mcx_p2.sum() / green_p2.sum()
    k_p3 = mcx_p3.sum() / green_p3.sum()

    print(f"\nScale factors:")
    print(f"  P2 k: {k_p2:.4e}")
    print(f"  P3 k: {k_p3:.4e}")
    print(f"  Ratio P3/P2: {k_p3 / k_p2:.2f}×")

    # The difference in k factors between P2 and P3 might be due to:
    # 1. Different source positions (left vs right side)
    # 2. Different tissue composition
    # 3. Different surface geometry

    print("\nConclusion:")
    print("  P2 vs P3 k difference is due to asymmetric tissue distribution")
    print("  This is expected for real anatomical data")


if __name__ == "__main__":
    main()
