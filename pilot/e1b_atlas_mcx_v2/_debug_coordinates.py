#!/usr/bin/env python3
"""Debug coordinate system."""

import numpy as np

# Load atlas
atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ

print(f"Volume shape (XYZ): {volume_xyz.shape}")
print(f"Volume X range: 0-{volume_xyz.shape[0]} voxels")
print(f"Volume Y range: 0-{volume_xyz.shape[1]} voxels")
print(f"Volume Z range: 0-{volume_xyz.shape[2]} voxels")

# Voxel size
voxel_size = 0.2  # mm

# Physical extents (centered at origin)
nx, ny, nz = volume_xyz.shape
x_center = nx / 2
y_center = ny / 2
z_center = nz / 2

x_min = (0 - x_center + 0.5) * voxel_size
x_max = (nx - x_center + 0.5) * voxel_size
y_min = (0 - y_center + 0.5) * voxel_size
y_max = (ny - y_center + 0.5) * voxel_size
z_min = (0 - z_center + 0.5) * voxel_size
z_max = (nz - z_center + 0.5) * voxel_size

print(f"\nPhysical extents (mm, centered at origin):")
print(f"  X: [{x_min:.1f}, {x_max:.1f}]")
print(f"  Y: [{y_min:.1f}, {y_max:.1f}]")
print(f"  Z: [{z_min:.1f}, {z_max:.1f}]")

# Find dorsal surface (max Z with tissue)
tissue_mask = volume_xyz > 0
z_indices = np.where(tissue_mask)
if len(z_indices[2]) > 0:
    dorsal_z_voxel = z_indices[2].max()
    ventral_z_voxel = z_indices[2].min()

    dorsal_z_mm = (dorsal_z_voxel - z_center + 0.5) * voxel_size
    ventral_z_mm = (ventral_z_voxel - z_center + 0.5) * voxel_size

    print(f"\nActual tissue Z range (voxels): {ventral_z_voxel} to {dorsal_z_voxel}")
    print(f"Actual tissue Z range (mm): {ventral_z_mm:.1f} to {dorsal_z_mm:.1f}")
    print(f"  Dorsal surface (max Z): {dorsal_z_mm:.1f} mm")
    print(f"  Ventral surface (min Z): {ventral_z_mm:.1f} mm")

# Source position from config
dorsal_z_config = 15.4  # mm (from config)
source_xy = [17.0, 48.0]
depth_mm = 2.0
source_z = dorsal_z_config - depth_mm

print(f"\nConfig source position:")
print(f"  X: {source_xy[0]} mm")
print(f"  Y: {source_xy[1]} mm")
print(f"  Z: {source_z} mm (dorsal {dorsal_z_config} - depth {depth_mm})")

# Check if source is inside volume
source_x_vox = source_xy[0] / voxel_size + x_center - 0.5
source_y_vox = source_xy[1] / voxel_size + y_center - 0.5
source_z_vox = source_z / voxel_size + z_center - 0.5

print(f"\nSource voxel indices:")
print(f"  X: {source_x_vox:.1f} (0-{nx})")
print(f"  Y: {source_y_vox:.1f} (0-{ny})")
print(f"  Z: {source_z_vox:.1f} (0-{nz})")

# Check if source is inside tissue
sx = int(round(source_x_vox))
sy = int(round(source_y_vox))
sz = int(round(source_z_vox))

if 0 <= sx < nx and 0 <= sy < ny and 0 <= sz < nz:
    is_inside = volume_xyz[sx, sy, sz] > 0
    print(f"\nSource at voxel [{sx}, {sy}, {sz}]:")
    print(f"  Inside tissue: {is_inside}")
    print(f"  Tissue label: {volume_xyz[sx, sy, sz]}")
else:
    print(f"\nSource voxel [{sx}, {sy}, {sz}] is OUTSIDE volume!")

# Compare with MCX config
print(f"\nMCX config offset: [0, 30, 0] mm")
mcx_x_vox = (source_xy[0] - 0) / voxel_size
mcx_y_vox = (source_xy[1] - 30) / voxel_size
mcx_z_vox = (source_z - 0) / voxel_size
print(f"MCX source voxel: [{mcx_x_vox:.1f}, {mcx_y_vox:.1f}, {mcx_z_vox:.1f}]")
