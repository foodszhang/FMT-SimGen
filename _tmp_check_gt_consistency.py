#!/usr/bin/env python3
"""Check gt_voxels vs gt_nodes consistency."""
import numpy as np
import json

# Load mesh
mesh = np.load('output/shared/mesh.npz')
nodes = mesh['nodes']
elements = mesh['elements']
tissue_labels = mesh['tissue_labels']
surface_faces = mesh['surface_faces']

# Load DE sample
tp = json.load(open('data/default/samples/sample_0000/tumor_params.json'))
gt_nodes = np.load('data/default/samples/sample_0000/gt_nodes.npy')
gt_voxels = np.load('data/default/samples/sample_0000/gt_voxels.npy')

print(f"=== Shapes ===")
print(f"mesh nodes: {nodes.shape}")
print(f"mesh elements: {elements.shape}")
print(f"gt_nodes: {gt_nodes.shape} (should match surface nodes)")
print(f"gt_voxels: {gt_voxels.shape}")

# gt_nodes is the ground truth at FEM surface nodes
# gt_voxels is the ground truth at voxel centers
# They should be consistent in the tumor region

# Sample tumor Focus 1 center
center = np.array([11.0, 32.0, 10.0])  # mm

# Find surface node closest to tumor center
dists = np.linalg.norm(nodes - center, axis=1)
nearest_node_idx = np.argmin(dists)
nearest_node = nodes[nearest_node_idx]
print(f"\n=== Focus 1 center: {center} mm ===")
print(f"Nearest surface node: idx={nearest_node_idx}, pos={nearest_node}, dist={dists[nearest_node_idx]:.3f}mm")
print(f"gt_nodes at nearest node: {gt_nodes[nearest_node_idx]:.4f}")

# Find voxel closest to tumor center
voxel_idx = center / 0.2
voxel_idx_int = np.round(voxel_idx).astype(int)
print(f"\nNearest voxel (int): {voxel_idx_int}")
print(f"gt_voxels at nearest voxel: {gt_voxels[voxel_idx_int[0], voxel_idx_int[1], voxel_idx_int[2]]:.4f}")

# Check: at the tumor center, gt_nodes should be ~1.0 (Gaussian peak)
print(f"\n=== At tumor center ===")
print(f"Expected: gt_nodes ~ 1.0, gt_voxels ~ 1.0")
print(f"Actual: node={gt_nodes[nearest_node_idx]:.4f}, voxel={gt_voxels[voxel_idx_int[0], voxel_idx_int[1], voxel_idx_int[2]]:.4f}")

# Now check the relationship more carefully:
# gt_voxels is defined at voxel centers: physical point (x,y,z) maps to voxel index floor(x/0.2)
# For center=(11, 32, 10)mm → voxel=(55, 160, 50)
# At that voxel: gt_voxels[55, 160, 50] should equal the Gaussian value at that point

# Compute expected Gaussian at this point for focus 1
# Focus 1: center=[11.0, 32.0, 10.0], radius/sigma=2.576mm
sigma = 2.576
d = np.linalg.norm(center - center)  # 0 at center
expected = np.exp(-0.5 * (d/sigma)**2)
print(f"\nExpected Gaussian at focus 1 center: {expected:.4f}")
print(f"Actual gt_voxels at voxel [55,160,50]: {gt_voxels[55, 160, 50]:.4f}")
print(f"Actual gt_nodes at nearest node: {gt_nodes[nearest_node_idx]:.4f}")

# Check max values
print(f"\n=== Max values ===")
print(f"gt_nodes max: {gt_nodes.max():.4f} at node {np.argmax(gt_nodes)}")
print(f"gt_voxels max: {gt_voxels.max():.4f} at voxel {np.unravel_index(np.argmax(gt_voxels), gt_voxels.shape)}")

# Where is gt_nodes max physically?
max_node_idx = np.argmax(gt_nodes)
max_node_pos = nodes[max_node_idx]
print(f"gt_nodes max node position: {max_node_pos} mm")

# Where is gt_voxels max physically?
max_vox_idx = np.unravel_index(np.argmax(gt_voxels), gt_voxels.shape)
max_vox_pos = np.array(max_vox_idx) * 0.2 + 0.1  # voxel center
print(f"gt_voxels max voxel position: {max_vox_pos} mm")

print(f"\n=== Tumor center positions from foci ===")
for i, f in enumerate(tp.get('foci', [])):
    c = np.array(f['center'])
    print(f"  Focus {i}: {c} mm → voxel {c/0.2}")
