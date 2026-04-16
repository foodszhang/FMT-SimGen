#!/usr/bin/env python3
"""Verify source position in atlas coordinates."""

import numpy as np

# Load atlas
atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
volume_xyz = volume.transpose(2, 1, 0)
atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

voxel_size = 0.2
nx, ny, nz = atlas_binary.shape
center = np.array([nx / 2, ny / 2, nz / 2])

print(f"Atlas shape: {atlas_binary.shape}")
print(f"Center (voxels): {center}")
print(f"Center (mm): {center * voxel_size}")

# MCX config source
source_config = np.array([17.0, 48.0, 8.1])  # config coords
mcx_offset = np.array([0, 30, 0])

# Convert to centered coords
source_centered = source_config - mcx_offset
print(f"\nSource (config): {source_config}")
print(f"Source (centered): {source_centered}")

# Convert to voxel indices (centered system)
source_voxel_centered = source_centered / voxel_size + center - 0.5
print(f"Source voxel (centered): {source_voxel_centered}")

# Check if inside volume
x, y, z = source_voxel_centered.astype(int)
print(f"\nSource at voxel [{x}, {y}, {z}]:")
if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
    print(f"  Inside volume: Yes")
    print(f"  Tissue label: {atlas_binary[x, y, z]}")
else:
    print(f"  Inside volume: NO!")

# Get tissue ranges
tissue_voxels = np.argwhere(atlas_binary > 0)
tissue_mm = (tissue_voxels - center + 0.5) * voxel_size

print(f"\nTissue ranges (mm, centered):")
print(f"  X: [{tissue_mm[:, 0].min():.1f}, {tissue_mm[:, 0].max():.1f}]")
print(f"  Y: [{tissue_mm[:, 1].min():.1f}, {tissue_mm[:, 1].max():.1f}]")
print(f"  Z: [{tissue_mm[:, 2].min():.1f}, {tissue_mm[:, 2].max():.1f}]")

# Find surface point directly above source
# Look for tissue voxels with X,Y near source and max Z
x_range = 5  # voxels
y_range = 5

x_min, x_max = max(0, x - x_range), min(nx, x + x_range + 1)
y_min, y_max = max(0, y - y_range), min(ny, y + y_range + 1)

region = atlas_binary[x_min:x_max, y_min:y_max, :]
if np.any(region > 0):
    # Find max Z in this region
    region_voxels = np.argwhere(region > 0)
    region_voxels[:, 0] += x_min
    region_voxels[:, 1] += y_min

    max_z_idx = np.argmax(region_voxels[:, 2])
    dorsal_voxel = region_voxels[max_z_idx]
    dorsal_mm = (dorsal_voxel - center + 0.5) * voxel_size

    print(f"\nDorsal surface near source:")
    print(f"  Voxel: {dorsal_voxel}")
    print(f"  Position (mm): {dorsal_mm}")
    print(
        f"  Distance from source: {np.linalg.norm(dorsal_mm - source_centered):.2f} mm"
    )
else:
    print(f"\nNo tissue found near source in X,Y range!")
