"""Debug direct-path checker to understand P1-dorsal issue."""

import numpy as np
from pathlib import Path

VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOXEL_SIZE_MM = 0.4

volume = np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)

print("Volume shape:", VOLUME_SHAPE_XYZ)
print("Voxel size:", VOXEL_SIZE_MM, "mm")
print("Physical size:", [s * VOXEL_SIZE_MM for s in VOLUME_SHAPE_XYZ], "mm")

center = np.array(VOLUME_SHAPE_XYZ) / 2
print("Center voxel:", center)

source_pos = np.array([-0.6, 2.4, 5.8])
print(f"\nP1-dorsal source: {source_pos} mm")

source_voxel = np.round(source_pos / VOXEL_SIZE_MM + center).astype(int)
print(f"Source voxel: {source_voxel}")

if (
    0 <= source_voxel[0] < VOLUME_SHAPE_XYZ[0]
    and 0 <= source_voxel[1] < VOLUME_SHAPE_XYZ[1]
    and 0 <= source_voxel[2] < VOLUME_SHAPE_XYZ[2]
):
    label = volume[source_voxel[0], source_voxel[1], source_voxel[2]]
    print(f"Source label: {label}")
else:
    print("Source out of bounds!")

print("\nZ profile at source XY:")
for z in range(
    max(0, source_voxel[2] - 10), min(VOLUME_SHAPE_XYZ[2], source_voxel[2] + 10)
):
    label = volume[source_voxel[0], source_voxel[1], z]
    z_mm = (z - center[2] + 0.5) * VOXEL_SIZE_MM
    marker = " <-- source" if z == source_voxel[2] else ""
    print(f"  Z={z} ({z_mm:.1f}mm): label={label}{marker}")

print("\nDorsal surface (max Z where tissue exists):")
binary_mask = volume > 0
nz_coords = np.where(binary_mask)
dorsal_z_voxel = np.max(nz_coords[2])
dorsal_z_mm = (dorsal_z_voxel - center[2] + 0.5) * VOXEL_SIZE_MM
print(f"  Dorsal Z voxel: {dorsal_z_voxel}")
print(f"  Dorsal Z mm: {dorsal_z_mm:.1f} mm")

print("\nRay-marching from source in +Z direction (0° view):")
direction = np.array([0, 0, 1])
step_mm = 0.1
for i in range(100):
    pos_mm = source_pos + i * step_mm * direction
    voxel = np.round(pos_mm / VOXEL_SIZE_MM + center).astype(int)

    if not (
        0 <= voxel[0] < VOLUME_SHAPE_XYZ[0]
        and 0 <= voxel[1] < VOLUME_SHAPE_XYZ[1]
        and 0 <= voxel[2] < VOLUME_SHAPE_XYZ[2]
    ):
        print(f"  Step {i}: pos={pos_mm}, voxel={voxel} -> OUT OF BOUNDS")
        break

    label = volume[voxel[0], voxel[1], voxel[2]]
    if i < 10 or label == 0:
        print(f"  Step {i}: pos={pos_mm}, voxel={voxel}, label={label}")
    if label == 0 and i > 0:
        print(f"  -> EXIT at step {i}, path length = {i * step_mm:.1f} mm")
        break
