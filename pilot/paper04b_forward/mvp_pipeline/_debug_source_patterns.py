"""Analyze archived MCX source patterns to find actual source centers."""

import numpy as np
from pathlib import Path

VOLUME_SHAPE_XYZ = (95, 100, 52)
VOXEL_SIZE_MM = 0.4
center = np.array(VOLUME_SHAPE_XYZ) / 2


def voxel_to_mm(voxel):
    return (voxel - center + 0.5) * VOXEL_SIZE_MM


base = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

positions = {
    "P1-dorsal": "S2-Vol-P1-dorsal-r2.0",
    "P2-left": "S2-Vol-P2-left-r2.0",
    "P3-right": "S2-Vol-P3-right-r2.0",
    "P4-dorsal-lat": "S2-Vol-P4-dorsal-lat-r2.0",
    "P5-ventral": "S2-Vol-P5-ventral-r2.0",
}

volume = np.fromfile(base / "mcx_volume_downsampled_2x.bin", dtype=np.uint8).reshape(
    VOLUME_SHAPE_XYZ
)

for name, dir_name in positions.items():
    source_path = base / dir_name / f"source-{dir_name}.bin"
    if not source_path.exists():
        print(f"{name}: source file not found")
        continue

    json_path = base / dir_name / f"{dir_name}.json"
    import json

    with open(json_path) as f:
        config = json.load(f)

    pattern_shape = tuple(config["Optode"]["Source"]["Param1"])
    origin = np.array(config["Optode"]["Source"]["Pos"])

    pattern = (
        np.fromfile(source_path, dtype=np.float32)
        .reshape(pattern_shape[::-1])
        .transpose(2, 1, 0)
    )

    coords = np.where(pattern > 0)
    if len(coords[0]) == 0:
        print(f"{name}: empty pattern")
        continue

    center_voxel = np.array(
        [
            np.average(coords[0], weights=pattern[coords]),
            np.average(coords[1], weights=pattern[coords]),
            np.average(coords[2], weights=pattern[coords]),
        ]
    )

    actual_center_voxel = origin + center_voxel
    actual_center_mm = voxel_to_mm(actual_center_voxel)

    av = actual_center_voxel.astype(int)
    if (
        0 <= av[0] < VOLUME_SHAPE_XYZ[0]
        and 0 <= av[1] < VOLUME_SHAPE_XYZ[1]
        and 0 <= av[2] < VOLUME_SHAPE_XYZ[2]
    ):
        label = volume[av[0], av[1], av[2]]
    else:
        label = "OOB"

    print(f"{name}:")
    print(f"  Pattern shape: {pattern_shape}")
    print(f"  Origin voxel: {origin}")
    print(f"  Pattern center (relative): {center_voxel}")
    print(f"  Actual center voxel: {actual_center_voxel}")
    print(f"  Actual center mm: {actual_center_mm}")
    print(f"  Label at center: {label}")
    print(f"  Pattern sum: {pattern.sum():.2f}")
    print()
