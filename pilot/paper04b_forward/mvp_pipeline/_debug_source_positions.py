"""Find correct source positions within tissue."""

import numpy as np
from pathlib import Path

VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOXEL_SIZE_MM = 0.4

volume = np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)
center = np.array(VOLUME_SHAPE_XYZ) / 2


def voxel_to_mm(voxel):
    return (voxel - center + 0.5) * VOXEL_SIZE_MM


def mm_to_voxel(mm):
    return np.round(mm / VOXEL_SIZE_MM + center).astype(int)


binary_mask = volume > 0
print("Tissue voxels:", np.sum(binary_mask))

nz_coords = np.where(binary_mask)
print(f"X range: {np.min(nz_coords[0])} - {np.max(nz_coords[0])}")
print(f"Y range: {np.min(nz_coords[1])} - {np.max(nz_coords[1])}")
print(f"Z range: {np.min(nz_coords[2])} - {np.max(nz_coords[2])}")

dorsal_z_voxel = np.max(nz_coords[2])
ventral_z_voxel = np.min(nz_coords[2])
print(
    f"\nDorsal surface Z: {dorsal_z_voxel} ({voxel_to_mm(np.array([0, 0, dorsal_z_voxel]))[2]:.1f} mm)"
)
print(
    f"Ventral surface Z: {ventral_z_voxel} ({voxel_to_mm(np.array([0, 0, ventral_z_voxel]))[2]:.1f} mm)"
)

y_slice = 56
print(f"\nAt Y slice {y_slice} ({voxel_to_mm(np.array([0, y_slice, 0]))[1]:.1f} mm):")
for z in range(VOLUME_SHAPE_XYZ[2] - 1, -1, -1):
    tissue_at_z = binary_mask[:, y_slice, z]
    if np.any(tissue_at_z):
        x_coords = np.where(tissue_at_z)[0]
        x_center = int(np.mean(x_coords))
        z_mm = voxel_to_mm(np.array([0, 0, z]))[2]
        print(
            f"  Z={z} ({z_mm:.1f}mm): X range [{np.min(x_coords)}, {np.max(x_coords)}], center X={x_center}"
        )
        break

y_slice = 56
for z in [dorsal_z_voxel - 10, dorsal_z_voxel - 5, dorsal_z_voxel]:
    if z < 0 or z >= VOLUME_SHAPE_XYZ[2]:
        continue
    tissue_at_z = binary_mask[:, y_slice, z]
    if np.any(tissue_at_z):
        x_coords = np.where(tissue_at_z)[0]
        x_center = int(np.mean(x_coords))
        x_mm = voxel_to_mm(np.array([x_center, y_slice, z]))
        label = volume[x_center, y_slice, z]
        print(f"\nPotential source at Z={z} ({x_mm[2]:.1f}mm):")
        print(f"  Voxel: ({x_center}, {y_slice}, {z})")
        print(f"  Position mm: {x_mm}")
        print(f"  Label: {label}")

print("\n\nChecking archived source positions:")
positions = {
    "P1-dorsal": np.array([-0.6, 2.4, 5.8]),
    "P2-left": np.array([-8.0, 2.4, 1.0]),
    "P3-right": np.array([6.8, 2.4, 1.0]),
    "P4-dorsal-lat": np.array([-6.3, 2.4, 5.8]),
    "P5-ventral": np.array([-0.6, 2.4, -3.8]),
}

for name, pos in positions.items():
    voxel = mm_to_voxel(pos)
    in_bounds = (
        0 <= voxel[0] < VOLUME_SHAPE_XYZ[0]
        and 0 <= voxel[1] < VOLUME_SHAPE_XYZ[1]
        and 0 <= voxel[2] < VOLUME_SHAPE_XYZ[2]
    )
    if in_bounds:
        label = volume[voxel[0], voxel[1], voxel[2]]
    else:
        label = "OUT_OF_BOUNDS"
    print(f"  {name}: pos={pos}, voxel={voxel}, label={label}")
