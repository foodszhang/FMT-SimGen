#!/usr/bin/env python3
"""Debug current run."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)

# Load atlas
atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
volume_xyz = volume.transpose(2, 1, 0)
atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

# Parameters
voxel_size = 0.2
camera_distance = 200.0
fov_mm = 50.0
detector_resolution = (256, 256)
angle = 0.0

tissue_params = {
    "mua_mm": 0.08697,
    "mus_prime_mm": 4.29071,
    "g": 0.9,
    "n": 1.37,
}

# Current source position (from run output)
source_pos = np.array([-1.0, 2.4, 8.1])

print(f"Source position: {source_pos}")

# Get surface coordinates
surface_coords, valid_mask = project_get_surface_coords(
    atlas_binary, angle, camera_distance, fov_mm, detector_resolution, voxel_size
)

# Compute Green's function
green_proj = green_infinite_point_source_on_surface(
    source_pos, surface_coords, valid_mask, tissue_params
)

# Find peak
peak_idx = np.unravel_index(np.argmax(green_proj), green_proj.shape)
peak_coord = surface_coords[peak_idx]

print(f"\nGreen function peak:")
print(f"  Pixel: {peak_idx}")
print(f"  Surface coord: {peak_coord}")
print(f"  Distance from source: {np.linalg.norm(peak_coord - source_pos):.2f} mm")

# Find surface point directly above source (same X,Y, max Z)
# Look for surface points with X,Y near source
x_tolerance = 2.0  # mm
y_tolerance = 2.0

x_diff = np.abs(surface_coords[:, :, 0] - source_pos[0])
y_diff = np.abs(surface_coords[:, :, 1] - source_pos[1])
close_xy = (x_diff < x_tolerance) & (y_diff < y_tolerance) & valid_mask

if np.any(close_xy):
    # Among these, find the one with max Z (dorsal)
    z_values = surface_coords[:, :, 2]
    z_values[~close_xy] = -np.inf
    dorsal_idx = np.unravel_index(np.argmax(z_values), z_values.shape)
    dorsal_coord = surface_coords[dorsal_idx]

    print(f"\nDorsal surface near source:")
    print(f"  Pixel: {dorsal_idx}")
    print(f"  Surface coord: {dorsal_coord}")
    print(f"  Distance from source: {np.linalg.norm(dorsal_coord - source_pos):.2f} mm")
    print(f"  Green value: {green_proj[dorsal_idx]:.6e}")
else:
    print(f"\nNo surface points found near X={source_pos[0]}, Y={source_pos[1]}")

# Compare with actual peak
print(f"\nActual Green peak:")
print(f"  Pixel: {peak_idx}")
print(f"  Green value: {green_proj[peak_idx]:.6e}")
