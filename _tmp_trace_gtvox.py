#!/usr/bin/env python3
"""Trace gt_voxels generation to find Z offset bug."""
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from fmt_simgen.sampling.dual_sampler import DualSampler, VoxelGridConfig
from fmt_simgen.tumor.tumor_generator import TumorGenerator

# Setup - replicate what DatasetBuilder does
sample_dir = "data/small_uniform_5samples/samples/sample_0000"
tp = json.load(open(f"{sample_dir}/tumor_params.json"))

# Reconstruct TumorSample from tumor_params (mimicking what builder does)
from fmt_simgen.tumor.tumor_generator import TumorSample, AnalyticFocus, ShapeType

foci = []
for f in tp['foci']:
    center = np.array(f['center'], dtype=np.float64)
    shape = ShapeType(f['shape'])
    params = f['params']
    focus = AnalyticFocus(center=center, shape=shape, params=params)
    foci.append(focus)

tumor_sample = TumorSample(foci=foci)

# Check focus centers
print("=== Focus Centers ===")
for i, focus in enumerate(tumor_sample.foci):
    print(f"  Focus {i}: {focus.center} mm")
    print(f"    Expected Z: {focus.center[2]:.2f}mm")

# Check DualSampler grid
voxel_grid_config = VoxelGridConfig(
    shape=(190, 200, 104),
    spacing=0.2,
    offset=np.array([0.0, 0.0, 0.0])
)
dual_sampler = DualSampler(nodes=np.zeros((100, 3)), voxel_grid_config=voxel_grid_config)

# Get voxel grid coordinates
vx = dual_sampler._voxel_coords
print(f"\n=== Voxel Grid ===")
print(f"Total voxels: {len(vx)}")
print(f"Grid extent: X=[{vx[:,0].min():.2f}, {vx[:,0].max():.2f}], Y=[{vx[:,1].min():.2f}, {vx[:,1].max():.2f}], Z=[{vx[:,2].min():.2f}, {vx[:,2].max():.2f}]")

# Sample at voxel grid
gt_voxels_computed = dual_sampler.sample_to_voxels(tumor_sample)
print(f"\n=== Computed gt_voxels ===")
print(f"Shape: {gt_voxels_computed.shape}")
print(f"Max: {gt_voxels_computed.max():.4f}")

# Find max location
max_loc = np.unravel_index(np.argmax(gt_voxels_computed), gt_voxels_computed.shape)
print(f"Max at (X,Y,Z)=({max_loc[0]},{max_loc[1]},{max_loc[2]})")
print(f"Max at XYZ mm: X={max_loc[0]*0.2:.2f}, Y={max_loc[1]*0.2:.2f}, Z={max_loc[2]*0.2:.2f}")

# Compare with saved gt_voxels
gt_voxels_saved = np.load(f"{sample_dir}/gt_voxels.npy")
max_loc_saved = np.unravel_index(np.argmax(gt_voxels_saved), gt_voxels_saved.shape)
print(f"\n=== Saved gt_voxels ===")
print(f"Shape: {gt_voxels_saved.shape}")
print(f"Max: {gt_voxels_saved.max():.4f}")
print(f"Max at (X,Y,Z)=({max_loc_saved[0]},{max_loc_saved[1]},{max_loc_saved[2]})")
print(f"Max at XYZ mm: X={max_loc_saved[0]*0.2:.2f}, Y={max_loc_saved[1]*0.2:.2f}, Z={max_loc_saved[2]*0.2:.2f}")

# Check if computed matches saved
diff = np.abs(gt_voxels_computed - gt_voxels_saved).max()
print(f"\nMax difference between computed and saved: {diff:.6f}")

if diff > 0.01:
    print("WARNING: Computed and saved gt_voxels differ significantly!")
    print("The saved file was likely generated with a BUGGY version.")
    print("Need to REGENERATE samples.")

# Now check what the coordinate looks like for the focus Z
print("\n=== Focus evaluation check ===")
for i, focus in enumerate(tumor_sample.foci):
    # Evaluate at focus center
    val_at_center = focus.evaluate(focus.center.reshape(1, 3))
    print(f"Focus {i}: value at center {focus.center} = {val_at_center[0]:.4f} (should be 1.0 for uniform)")

    # Evaluate at some nearby grid points
    cx, cy, cz = focus.center
    test_points = np.array([
        [cx, cy, cz],          # center
        [cx, cy, cz + 1.0],   # +1mm Z
        [cx, cy, cz - 1.0],   # -1mm Z
    ])
    vals = focus.evaluate(test_points)
    print(f"  Test values: center={vals[0]:.4f}, +Z={vals[1]:.4f}, -Z={vals[2]:.4f}")

# Evaluate at voxel grid points near focus 0
print("\n=== Evaluation at grid points near focus 0 ===")
focus0 = tumor_sample.foci[0].center
print(f"Focus 0 center: {focus0}")

# Find closest grid point to focus 0 center
dists = np.linalg.norm(dual_sampler._voxel_coords - focus0, axis=1)
closest_idx = np.argmin(dists)
closest_pt = dual_sampler._voxel_coords[closest_idx]
print(f"Closest grid point: {closest_pt} mm (dist={dists[closest_idx]:.3f}mm)")
print(f"Closest grid voxel index: {np.unravel_index(closest_idx, (190, 200, 104))}")

# Check value at closest point
val_at_closest = tumor_sample.evaluate(closest_pt.reshape(1, 3))
print(f"Value at closest grid point: {val_at_closest[0]:.4f}")

# Check value at several Z positions
print("\n=== Focus 0 Z-profile along Y=31mm, X=7mm ===")
test_z = np.arange(0, 20.5, 0.5)  # Z from 0 to 20mm in 0.5mm steps
test_points = np.column_stack([np.full_like(test_z, focus0[0]), np.full_like(test_z, focus0[1]), test_z])
test_vals = tumor_sample.evaluate(test_points)
for z, v in zip(test_z, test_vals):
    if v > 0.01:
        print(f"  Z={z:.1f}mm: {v:.4f}")