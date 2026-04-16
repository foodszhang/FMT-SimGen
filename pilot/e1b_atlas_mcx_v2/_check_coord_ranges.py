#!/usr/bin/env python3
import numpy as np

# Load atlas
atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
volume_xyz = volume.transpose(2, 1, 0)
atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

# Get tissue voxel coordinates
tissue_voxels = np.argwhere(atlas_binary > 0)
print(f"Number of tissue voxels: {len(tissue_voxels)}")

# Convert to mm (centered)
voxel_size = 0.2
nx, ny, nz = atlas_binary.shape
center = np.array([nx / 2, ny / 2, nz / 2])
tissue_mm = (tissue_voxels - center + 0.5) * voxel_size

print(
    f"\nTissue X range: [{tissue_mm[:, 0].min():.1f}, {tissue_mm[:, 0].max():.1f}] mm"
)
print(f"Tissue Y range: [{tissue_mm[:, 1].min():.1f}, {tissue_mm[:, 1].max():.1f}] mm")
print(f"Tissue Z range: [{tissue_mm[:, 2].min():.1f}, {tissue_mm[:, 2].max():.1f}] mm")

# Source positions
source_xy_centered = [17.0, 48.0]
print(f"\nConfig source XY: {source_xy_centered}")
x_in = tissue_mm[:, 0].min() <= source_xy_centered[0] <= tissue_mm[:, 0].max()
y_in = tissue_mm[:, 1].min() <= source_xy_centered[1] <= tissue_mm[:, 1].max()
print(
    f"  X={source_xy_centered[0]} is {'INSIDE' if x_in else 'OUTSIDE'} tissue X range"
)
print(
    f"  Y={source_xy_centered[1]} is {'INSIDE' if y_in else 'OUTSIDE'} tissue Y range"
)

# Check MCX offset
mcx_offset = [0, 30, 0]
source_xy_mcx = [
    source_xy_centered[0] - mcx_offset[0],
    source_xy_centered[1] - mcx_offset[1],
]
print(f"\nMCX offset: {mcx_offset}")
print(f"Source XY in MCX coords: {source_xy_mcx}")

# Check if source is inside in MCX coords
x_mcx_in = 0 <= (source_xy_mcx[0] / voxel_size + nx / 2) < nx
y_mcx_in = 0 <= (source_xy_mcx[1] / voxel_size + ny / 2) < ny
print(f"  X in voxel range: {x_mcx_in}")
print(f"  Y in voxel range: {y_mcx_in}")
