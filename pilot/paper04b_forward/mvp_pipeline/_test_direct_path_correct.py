"""Test direct-path checker with correct source positions."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pilot.paper04b_forward.shared.direct_path import is_direct_path

VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOXEL_SIZE_MM = 0.4

volume = np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)
center = np.array(VOLUME_SHAPE_XYZ) / 2


def voxel_to_mm(voxel):
    return (voxel - center + 0.5) * VOXEL_SIZE_MM


sources = {
    "P1-dorsal": np.array([-0.6, 2.4, 5.8]),
    "P2-left": np.array([-8.0, 2.4, 1.0]),
    "P3-right": np.array([6.8, 2.4, 1.0]),
    "P4-dorsal-lat": np.array([-6.3, 2.4, 5.8]),
    "P5-ventral": np.array([-0.6, 2.4, -3.8]),
}


def mm_to_voxel(mm):
    return np.floor(mm / VOXEL_SIZE_MM + center).astype(int)


print("=" * 80)
print("Direct-Path Analysis for Archived Sources")
print("=" * 80)

for name, pos in sources.items():
    voxel = mm_to_voxel(pos)
    label = (
        volume[voxel[0], voxel[1], voxel[2]]
        if (
            0 <= voxel[0] < VOLUME_SHAPE_XYZ[0]
            and 0 <= voxel[1] < VOLUME_SHAPE_XYZ[1]
            and 0 <= voxel[2] < VOLUME_SHAPE_XYZ[2]
        )
        else 0
    )

    organ_names = {
        0: "air",
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

    print(f"\n{name}: pos={pos}, label={label} ({organ_names.get(label, 'unknown')})")

    for angle in [0, 30, 60, 90, -90, 180]:
        result = is_direct_path(pos, angle, volume, VOXEL_SIZE_MM)
        status = "✓" if result.is_direct else "✗"
        path_len = (
            f"{result.path_length_mm:.1f}mm" if result.path_length_mm > 0 else "N/A"
        )
        print(f"  {angle:>4}°: {status} {path_len:<8} {result.reason}")

print("\n" + "=" * 80)
print("Summary Table")
print("=" * 80)
print(f"{'Source':<15} {'0°':<8} {'90°':<8} {'-90°':<8} {'180°':<8}")
print("-" * 80)

for name, pos in sources.items():
    results = {}
    for angle in [0, 90, -90, 180]:
        result = is_direct_path(pos, angle, volume, VOXEL_SIZE_MM)
        results[angle] = "✓" if result.is_direct else "✗"
    print(
        f"{name:<15} {results[0]:<8} {results[90]:<8} {results[-90]:<8} {results[180]:<8}"
    )

print("=" * 80)
