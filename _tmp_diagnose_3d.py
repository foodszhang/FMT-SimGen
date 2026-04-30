#!/usr/bin/env python3
"""Diagnose why 3D visualization looks wrong."""
import numpy as np
import json

# 1. DE tumor params: what are the actual centers and sigmas?
tp = json.load(open('data/default/samples/sample_0000/tumor_params.json'))
print("=== DE Tumor Parameters ===")
for i, f in enumerate(tp.get('foci', [])):
    c = f['center']
    r = f.get('radius') or max(f.get('rx',0), f.get('ry',0), f.get('rz',0))
    print(f"  Focus {i}: center={c}, sigma/radius={r:.3f}mm")

# 2. MCX source pattern Pos (origin) from JSON
mcx_cfg = json.load(open('data/default/samples/sample_0000/sample_0000.json'))
src_cfg = mcx_cfg['Optode']['Source']
print(f"\n=== MCX Source Config ===")
print(f"  Pos (origin voxel): {src_cfg['Pos']}")
print(f"  Pattern shape: Nx={src_cfg['Pattern']['Nx']}, Ny={src_cfg['Pattern']['Ny']}, Nz={src_cfg['Pattern']['Nz']}")

# The source pattern origin in the MCX volume
src_origin = src_cfg['Pos']  # [1, 106, 0] in voxel coords
print(f"  Source origin voxel: {src_origin}")
print(f"  Source origin physical (mm): {[p*0.2 for p in src_origin]}")

# 3. Check if source is being placed at correct physical position
# In MCX: physical = voxel_index * voxel_size
# But source pattern is offset by Pos from the volume corner
# So the pattern bounding box in volume coords is:
#   [Pos[0], Pos[0]+Nx) x [Pos[1], Pos[1]+Ny) x [Pos[2], Pos[2]+Nz)
# In physical mm: [Pos[0]*0.2, (Pos[0]+Nx)*0.2] etc.

# 4. Load source and check where the "hot spot" actually is
src = np.fromfile('data/default/samples/sample_0000/source-sample_0000.bin', dtype=np.float32)
src_3d = src.reshape((167, 94, 104))
print(f"\n=== MCX Source Pattern Stats ===")
print(f"  shape: {src_3d.shape}")
print(f"  max: {src_3d.max():.4f}, min: {src_3d.min():.6e}")
print(f"  nonzero count: {(src_3d > 0).sum()}, total: {src_3d.size}")

# Find hot spot (top 1% values)
threshold = np.percentile(src_3d[src_3d > 0], 99)
hot_spot = src_3d > threshold
hot_coords = np.argwhere(hot_spot)
print(f"  hot spot (>p99={threshold:.4f}): {hot_spot.sum()} voxels")
if len(hot_coords) > 0:
    c = hot_coords.mean(axis=0)
    print(f"  hot spot centroid voxel: {c}")
    # Physical position relative to MCX volume origin
    physical = c * 0.2
    print(f"  hot spot centroid physical (mm): {physical}")
    # Plus offset from Pos
    physical += np.array(src_origin) * 0.2
    print(f"  hot spot absolute physical (mm): {physical}")

# 5. DE gt_voxels: what should the "tumor region" be?
gt = np.load('data/default/samples/sample_0000/gt_voxels.npy')
print(f"\n=== DE gt_voxels Stats ===")
print(f"  shape: {gt.shape}")
print(f"  nonzero: {(gt>0).sum()} ({100*(gt>0).mean():.1f}%)")

# The gt_voxels is continuous Gaussian values - threshold at ~0.5 to see tumor core
gt_threshold = 0.5
gt_tumor_core = gt > gt_threshold
print(f"  tumor core (>0.5): {gt_tumor_core.sum()} voxels")
if gt_tumor_core.any():
    c = np.argwhere(gt_tumor_core).mean(axis=0) * 0.2
    print(f"  tumor core centroid physical (mm): {c}")

# 6. Check the source pattern vs gt_voxels alignment
# MCX source pattern origin is at Pos=[1,106,0]
# This is (1, 106, 0) voxel in the MCX volume
# The source pattern spans [1, 168] x [106, 200] x [0, 104]
# Physical origin of MCX volume is (0,0,0) in trunk-local coords
# So the source physical position starts at [0.2, 21.2, 0.0] mm
print(f"\n=== Coordinate Alignment ===")
print(f"  MCX volume: 190x200x104 @ 0.2mm, origin=(0,0,0)")
print(f"  MCX source pattern origin voxel: {src_origin}")
print(f"  MCX source pattern origin physical: {[p*0.2 for p in src_origin]} mm")
print(f"  This means source pattern starts at X={1*0.2}, Y={106*0.2}, Z=0")
print(f"  DE tumor centers:")
for i, f in enumerate(tp.get('foci', [])):
    print(f"    Focus {i}: {f['center']} mm")
