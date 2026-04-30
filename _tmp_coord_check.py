#!/usr/bin/env python3
"""Understand the coordinate mismatch between DE and MCX."""
import numpy as np
import json

# MCX volume is 190x200x104 @ 0.2mm
# The source pattern has Pos=[1, 106, 0] which means:
#   pattern voxel [0,0,0] corresponds to volume voxel [1, 106, 0]
#   pattern voxel [i,j,k] corresponds to volume voxel [1+i, 106+j, k]

src = np.fromfile('data/default/samples/sample_0000/source-sample_0000.bin', dtype=np.float32)
src_3d = src.reshape((167, 94, 104))

# Source hot spot in pattern-local voxel coords
threshold = np.percentile(src_3d[src_3d > 0], 99)
hot_mask = src_3d > threshold
hot_pattern_voxel = np.argwhere(hot_mask).mean(axis=0)
print(f"Hot spot in pattern-local voxel coords: {hot_pattern_voxel}")

# Convert to volume voxel coords (add Pos offset)
Pos = np.array([1, 106, 0])  # from MCX JSON
hot_volume_voxel = hot_pattern_voxel + Pos
print(f"Hot spot in volume voxel coords: {hot_volume_voxel}")

# Convert to physical mm
voxel_size = 0.2
hot_physical = hot_volume_voxel * voxel_size
print(f"Hot spot in physical mm (relative to volume origin): {hot_physical}")

# DE tumor Focus 1 center in voxel coords
de_center = np.array([11.0, 32.0, 10.0])
de_voxel = de_center / voxel_size
print(f"\nDE Focus 1 center in voxel coords: {de_voxel}")

# Distance between DE center and MCX hot spot
diff = de_center - hot_physical
print(f"Discrepancy: DE center - MCX hot spot = {diff} mm")
print(f"L2 distance: {np.linalg.norm(diff):.2f} mm")

# THE PROBLEM: The MCX source pattern is a LOCAL subset of the volume
# It's placed at Pos=[1,106,0] within the volume. This is the offset.
# In the visualize script, we were treating the source pattern as if it's
# the entire volume (starting at 0,0,0), which is wrong.

# The source pattern covers Y=[106,200] voxels = Y=[21.2, 40.0] mm
# DE tumors are at Y ~ 32 mm - they fall within this range. OK.

# But the script renders the source pattern AS-IF its at Y=[0,94] voxels
# which corresponds to Y=[0, 18.8] mm physical - WRONG.

# The correct rendering should add the Pos offset to get the actual
# position in the volume.

print(f"\n=== What the visualization was showing (WRONG) ===")
print(f"Source pattern origin as rendered: (0, 0, 0) mm")
print(f"Hot spot as rendered: {hot_pattern_voxel * voxel_size} mm")

print(f"\n=== What it should show (CORRECT) ===")
print(f"Source pattern origin: {Pos * voxel_size} mm")
print(f"Hot spot: {hot_physical} mm")
