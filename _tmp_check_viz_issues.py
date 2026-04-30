#!/usr/bin/env python3
"""Check if visualization is showing mesh at all."""
import numpy as np
import json

# Load mesh
mesh = np.load('output/shared/mesh.npz')
nodes = mesh['nodes']
surface_faces = mesh['surface_faces']

# DE gt_voxels
gt_voxels = np.load('data/default/samples/sample_0000/gt_voxels.npy')

print(f"=== Mesh ===")
print(f"nodes: {nodes.shape[0]}")
print(f"surface_faces: {surface_faces.shape[0]}")

# The DE visualization shows gt_voxels as a point cloud
# But gt_voxels is at voxel resolution (190x200x104 = 3.9M points)
# While mesh has only 52k nodes

# Key insight: gt_voxels is NOT the mesh - it's a voxel representation
# The visualization should show:
# 1. The MESH surface (from mesh.npz surface_faces)
# 2. The TUMOR CORE (thresholded gt_voxels > 0.5)
# 3. Optionally: the mesh nodes colored by gt_nodes value

print(f"\n=== gt_voxels content ===")
print(f"gt_voxels shape: {gt_voxels.shape} (voxel grid)")
print(f"non-zero: {(gt_voxels>0).sum()} ({100*(gt_voxels>0).mean():.1f}% of volume)")
print(f">0.5: {(gt_voxels>0.5).sum()}")
print(f">0.1: {(gt_voxels>0.1).sum()}")

# The visualization as currently written shows ALL of gt_voxels>0
# which is 1.86M voxels = 47% of volume
# This covers most of the body, not just the tumor

# The CORRECT rendering should threshold to show only tumor core
threshold = 0.5
tumor_core = gt_voxels > threshold
print(f"\n=== If we threshold at {threshold} ===")
print(f"Tumor core voxels: {tumor_core.sum()}")
if tumor_core.any():
    c = np.argwhere(tumor_core).mean(axis=0) * 0.2
    print(f"Tumor core centroid: {c} mm")

# Now check where the mesh surface nodes are vs gt_voxels
# Surface nodes should be at the body boundary
print(f"\n=== Mesh surface node range ===")
print(f"X: [{nodes[:,0].min():.1f}, {nodes[:,0].max():.1f}]")
print(f"Y: [{nodes[:,1].min():.1f}, {nodes[:,1].max():.1f}]")
print(f"Z: [{nodes[:,1].min():.1f}, {nodes[:,1].max():.1f}]")

# gt_voxels volume range
print(f"\n=== gt_voxels volume range ===")
print(f"X: [0, {gt_voxels.shape[0]*0.2:.1f}] mm")
print(f"Y: [0, {gt_voxels.shape[1]*0.2:.1f}] mm")
print(f"Z: [0, {gt_voxels.shape[2]*0.2:.1f}] mm")
