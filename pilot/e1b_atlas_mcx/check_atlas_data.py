#!/usr/bin/env python3
"""Check available atlas data for E1b experiment."""

import numpy as np
from pathlib import Path

base_dir = Path("/home/foods/pro/FMT-SimGen/output/shared")

print("=" * 60)
print("Checking Atlas Data Availability")
print("=" * 60)

# Check mesh
mesh_path = base_dir / "mesh.npz"
if mesh_path.exists():
    mesh = np.load(mesh_path)
    print(f"\n✓ mesh.npz found")
    print(f"  Keys: {list(mesh.keys())}")
    if "nodes" in mesh:
        print(f"  Nodes: {mesh['nodes'].shape}")
    if "elements" in mesh:
        print(f"  Elements: {mesh['elements'].shape}")
else:
    print("\n✗ mesh.npz not found")

# Check atlas_full
atlas_path = base_dir / "atlas_full.npz"
if atlas_path.exists():
    atlas = np.load(atlas_path)
    print(f"\n✓ atlas_full.npz found")
    print(f"  Keys: {list(atlas.keys())}")
    for key in atlas.keys():
        arr = atlas[key]
        if hasattr(arr, "shape"):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
else:
    print("\n✗ atlas_full.npz not found")

# Check MCX volume
mcx_vol_path = base_dir / "mcx_volume_trunk.bin"
if mcx_vol_path.exists():
    print(f"\n✓ mcx_volume_trunk.bin found")
    print(f"  Size: {mcx_vol_path.stat().st_size / (1024**2):.1f} MB")
else:
    print("\n✗ mcx_volume_trunk.bin not found")

# Check surface data from E1d
e1d_surface_path = Path(
    "/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface/results/gt_atlas/A1_atlas_self_consistent_shallow_gt.npz"
)
if e1d_surface_path.exists():
    surface = np.load(e1d_surface_path)
    print(f"\n✓ E1d surface data found")
    print(f"  Keys: {list(surface.keys())}")
    for key in surface.keys():
        arr = surface[key]
        if hasattr(arr, "shape"):
            print(f"  {key}: shape={arr.shape}")
else:
    print("\n✗ E1d surface data not found")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
