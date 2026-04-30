#!/usr/bin/env python3
"""Deep comparison of DE measurement_b vs MCX surface fluence."""
import numpy as np
import json
import yaml
import jdata as jd
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD
import matplotlib.pyplot as plt

sample_dir = "data/small_uniform_5samples/samples/sample_0000"

# Load all data
mesh = np.load("output/shared/mesh.npz")
surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
b = np.load(f"{sample_dir}/measurement_b.npy")

tp = json.load(open(f"{sample_dir}/tumor_params.json"))
print(f"Tumor foci: {tp['foci']}")
print(f"Source type: {tp['source_type']}")

# Load MCX volume and extract surface fluence values at the same physical positions
data = jd.load(f"{sample_dir}/sample_0000.jnii")
nifti = data["NIFTIData"][:, :, :, 0, 0]
mcx_vol = np.transpose(nifti, (2, 1, 0))
voxel_size = 0.2

# For each DE surface node, find the MCX fluence value at that physical position
# by nearest-voxel lookup
mcx_at_de_nodes = []
de_node_phys = surface_nodes  # trunk-local mm

for nd in de_node_phys:
    vx = int(round(nd[0] / voxel_size))
    vy = int(round(nd[1] / voxel_size))
    vz = int(round(nd[2] / voxel_size))
    # Clamp to volume bounds
    vx = max(0, min(vx, mcx_vol.shape[0]-1))
    vy = max(0, min(vy, mcx_vol.shape[1]-1))
    vz = max(0, min(vz, mcx_vol.shape[2]-1))
    mcx_at_de_nodes.append(mcx_vol[vx, vy, vz])

mcx_at_de_nodes = np.array(mcx_at_de_nodes)

print(f"\n=== DE measurement_b vs MCX at same physical positions ===")
print(f"DE b:      min={b.min():.6f}, max={b.max():.6f}, mean={b.mean():.6f}")
print(f"MCX@DE_nd: min={mcx_at_de_nodes.min():.2f}, max={mcx_at_de_nodes.max():.2f}, mean={mcx_at_de_nodes.mean():.2f}")

# Correlation between DE b and MCX at same positions
nz = (b > 0) & (mcx_at_de_nodes > 0)
print(f"\nNon-zero both: {nz.sum()} / {len(b)}")
if nz.sum() > 100:
    corr = np.corrcoef(b[nz], mcx_at_de_nodes[nz])[0, 1]
    print(f"Correlation (non-zero): {corr:.4f}")

# Visual comparison at matching positions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter: DE b vs MCX at same physical position
ax = axes[0]
ax.scatter(b, mcx_at_de_nodes, alpha=0.1, s=1)
ax.set_xlabel("DE measurement_b")
ax.set_ylabel("MCX fluence at DE node pos")
ax.set_title("DE b vs MCX@DE_node_pos")
ax.set_xscale("log")
ax.set_yscale("log")

# Show where non-zero b is on surface (Y-Z plane slice at X median of body)
ax = axes[1]
# Pick X index near center of body
x_idx = 80  # around X=16mm
slice_de = b  # all surface nodes at once isn't easy for 2D

# Better: compare projections at a specific angle
cam_cfg = yaml.safe_load(open("config/default.yaml"))["view_config"]
cam = TurntableCamera(cam_cfg)

# Project MCX surface-only at 0 degrees
surf_vol = np.zeros_like(mcx_vol)
surf_vol[0, :, :] = mcx_vol[0, :, :]
surf_vol[-1, :, :] = mcx_vol[-1, :, :]
surf_vol[:, 0, :] = mcx_vol[:, 0, :]
surf_vol[:, -1, :] = mcx_vol[:, -1, :]
surf_vol[:, :, 0] = mcx_vol[:, :, 0]
surf_vol[:, :, -1] = mcx_vol[:, :, -1]

proj_mcx_surf, _ = project_volume_reference(
    surf_vol, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
)

# For DE: put b values at surface node positions, project
de_vol = np.zeros_like(mcx_vol)
for nd, bv in zip(de_node_phys, b):
    vx = int(round(nd[0] / voxel_size))
    vy = int(round(nd[1] / voxel_size))
    vz = int(round(nd[2] / voxel_size))
    if 0 <= vx < mcx_vol.shape[0] and 0 <= vy < mcx_vol.shape[1] and 0 <= vz < mcx_vol.shape[2]:
        de_vol[vx, vy, vz] = max(de_vol[vx, vy, vz], bv)

proj_de_direct, _ = project_volume_reference(
    de_vol, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
)

ax = axes[1]
ax.imshow(proj_de_direct, cmap="hot")
ax.set_title("DE surface b → vol → proj @ 0°")

ax = axes[2]
ax.imshow(proj_mcx_surf, cmap="hot")
ax.set_title("MCX surface fluence → proj @ 0°")

plt.tight_layout()
plt.savefig("output/verification/_debug_de_mcx_aligned.png", dpi=150)
plt.close()
print("\nSaved: output/verification/_debug_de_mcx_aligned.png")

# Key question: are DE and MCX actually computing the same physical quantity?
# DE forward: diffusion equation with uniform internal source → surface fluence
# MCX: Monte Carlo with uniform internal source → surface fluence
# They SHOULD agree in pattern (though MCX has noise)

# Let's check: at the tumor positions, what are the surface node b values?
tumor_pos = np.array(tp['foci'][0]['center'])  # Focus 0: [7.0, 31.4, 6.2]
# Find surface nodes near the tumor
dists = np.linalg.norm(surface_nodes - tumor_pos, axis=1)
nearest_surface_idx = np.argsort(dists)[:10]
print(f"\n=== Surface nodes nearest to tumor focus 0 at {tumor_pos} ===")
print(f"Distances: {dists[nearest_surface_idx]}")
print(f"DE b at these nodes: {b[nearest_surface_idx]}")
# MCX at same positions
for idx in nearest_surface_idx[:3]:
    nd = surface_nodes[idx]
    vx = int(round(nd[0]/voxel_size))
    vy = int(round(nd[1]/voxel_size))
    vz = int(round(nd[2]/voxel_size))
    print(f"  node phys={nd}, MCX@node={mcx_vol[vx,vy,vz]:.2f}")