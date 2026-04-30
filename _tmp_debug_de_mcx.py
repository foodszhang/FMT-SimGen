#!/usr/bin/env python3
"""Debug DE vs MCX surface projection comparison."""
import numpy as np
import json
import yaml
import jdata as jd
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD
import matplotlib.pyplot as plt

sample_dir = "data/small_uniform_5samples/samples/sample_0000"

# Load DE surface data
mesh = np.load("output/shared/mesh.npz")
nodes = mesh["nodes"]
surface_idx = mesh["surface_node_indices"]
surface_nodes = nodes[surface_idx]
b = np.load(f"{sample_dir}/measurement_b.npy")

print(f"=== DE Surface ===")
print(f"Surface nodes: {len(surface_nodes)}")
print(f"b shape: {b.shape}, range: [{b.min():.6f}, {b.max():.6f}]")
print(f"b non-zero: {(b > 0).sum()} / {len(b)}")

# Where are the non-zero b values physically?
nz_mask = b > 0
nz_nodes = surface_nodes[nz_mask]
print(f"\nNon-zero b nodes: {len(nz_nodes)}")
if len(nz_nodes) > 0:
    print(f"  X: [{nz_nodes[:,0].min():.2f}, {nz_nodes[:,0].max():.2f}] mm")
    print(f"  Y: [{nz_nodes[:,1].min():.2f}, {nz_nodes[:,1].max():.2f}] mm")
    print(f"  Z: [{nz_nodes[:,2].min():.2f}, {nz_nodes[:,2].max():.2f}] mm")

# Load MCX volume
jnii_path = list(f"{sample_dir}" for _ in [1])[0]
jnii_files = list(f"{sample_dir}".split())
print(f"\nJNII files: {[f for f in __import__('pathlib').Path(sample_dir).glob('*.jnii')]}")

# Check the projection directly
cam_cfg = yaml.safe_load(open("config/default.yaml"))["view_config"]
cam = TurntableCamera(cam_cfg)

# Load MCX volume and project just the surface layer
data = jd.load(f"{sample_dir}/sample_0000.jnii")
nifti = data["NIFTIData"][:, :, :, 0, 0]
mcx_vol = np.transpose(nifti, (2, 1, 0))
print(f"\nMCX vol shape: {mcx_vol.shape}, max: {mcx_vol.max():.2f}")

# Create a surface-only volume (just the boundary voxels with mcx values)
surf_vol = np.zeros_like(mcx_vol)
for x in [0, mcx_vol.shape[0]-1]:
    surf_vol[x, :, :] = mcx_vol[x, :, :]
for y in [0, mcx_vol.shape[1]-1]:
    surf_vol[:, y, :] = mcx_vol[:, y, :]
for z in [0, mcx_vol.shape[2]-1]:
    surf_vol[:, :, z] = mcx_vol[:, :, z]
print(f"MCX surface voxels non-zero: {(surf_vol > 0).sum()}")

# Project surface-only MCX at angle 0
proj_surf, depth = project_volume_reference(
    surf_vol, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
)
print(f"\nMCX surface-only projection at 0°: shape={proj_surf.shape}, range=[{proj_surf.min():.6f}, {proj_surf.max():.6f}]")

# Now let's also project full MCX volume normalized
mcx_norm = mcx_vol / max(mcx_vol.max(), 1e-6)
proj_full, _ = project_volume_reference(
    mcx_norm, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
)
print(f"MCX full-norm projection at 0°: range=[{proj_full.min():.6f}, {proj_full.max():.6f}]")

# Check: what does the DE surface look like when projected?
# DE measurement_b is at surface nodes - we need to put them in a volume first
de_vol = np.zeros_like(mcx_vol)
voxel_size = 0.2
# For each surface node, find the nearest voxel
for i, (nd, bv) in enumerate(zip(surface_nodes, b)):
    vx = int(round((nd[0] - 0.0) / voxel_size))
    vy = int(round((nd[1] - 0.0) / voxel_size))
    vz = int(round((nd[2] - 0.0) / voxel_size))
    if 0 <= vx < mcx_vol.shape[0] and 0 <= vy < mcx_vol.shape[1] and 0 <= vz < mcx_vol.shape[2]:
        if bv > 0:
            de_vol[vx, vy, vz] = max(de_vol[vx, vy, vz], bv)

print(f"\nDE volume non-zero voxels: {(de_vol > 0).sum()}")
proj_de_direct, _ = project_volume_reference(
    de_vol, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
)
print(f"DE direct projection at 0°: range=[{proj_de_direct.min():.6f}, {proj_de_direct.max():.6f}]")

# Show side by side
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(proj_surf, cmap="hot")
axes[0].set_title("MCX surface only @ 0°")
axes[1].imshow(proj_full, cmap="hot")
axes[1].set_title("MCX full (norm) @ 0°")
axes[2].imshow(proj_de_direct, cmap="hot")
axes[2].set_title("DE direct (b→vol) @ 0°")
diff = proj_de_direct - proj_surf
v = max(abs(diff.min()), abs(diff.max()), 0.01)
axes[3].imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
axes[3].set_title("Diff DE - MCX surf")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/verification/_debug_de_mcx.png", dpi=150)
plt.close()
print("\nSaved debug image: output/verification/_debug_de_mcx.png")