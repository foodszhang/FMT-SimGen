#!/usr/bin/env python3
"""Deep diagnostic of DE vs MCX alignment issues."""
import json
import numpy as np
import jdata as jd
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD
import yaml, matplotlib.pyplot as plt

sample_dir = "data/small_uniform_5samples/samples/sample_0000"
voxel_size = 0.2

# Load all data
mesh = np.load("output/shared/mesh.npz")
surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
b = np.load(f"{sample_dir}/measurement_b.npy")
tp = json.load(open(f"{sample_dir}/tumor_params.json"))
gt_voxels = np.load(f"{sample_dir}/gt_voxels.npy")

data = jd.load(f"{sample_dir}/sample_0000.jnii")
nifti = data["NIFTIData"][:, :, :, 0, 0]
mcx_vol = np.transpose(nifti, (2, 1, 0))

print("=== Data Shapes ===")
print(f"surface_nodes: {surface_nodes.shape}")
print(f"b: {b.shape}, range [{b.min():.6f}, {b.max():.6f}]")
print(f"gt_voxels: {gt_voxels.shape}, range [{gt_voxels.min():.6f}, {gt_voxels.max():.6f}]")
print(f"mcx_vol: {mcx_vol.shape}, range [{mcx_vol.min():.2f}, {mcx_vol.max():.2f}]")

# Check tumor positions
print("\n=== Tumor Foci ===")
for i, f in enumerate(tp.get('foci', [])):
    c = np.array(f['center'])
    print(f"  Focus {i}: {c} mm = voxels {c/voxel_size}")

# Check where gt_voxels has high values (tumor core)
tumor_core = gt_voxels > 0.5
if tumor_core.any():
    xi, yi, zi = np.where(tumor_core)
    print(f"\n=== GT Voxels Tumor Core (>0.5) ===")
    print(f"  Count: {tumor_core.sum()}")
    print(f"  X: [{xi.min()}, {xi.max()}] = [{xi.min()*voxel_size:.2f}, {xi.max()*voxel_size:.2f}] mm")
    print(f"  Y: [{yi.min()}, {yi.max()}] = [{yi.min()*voxel_size:.2f}, {yi.max()*voxel_size:.2f}] mm")
    print(f"  Z: [{zi.min()}, {zi.max()}] = [{zi.min()*voxel_size:.2f}, {zi.max()*voxel_size:.2f}] mm")
    center_vox = np.array([xi.mean(), yi.mean(), zi.mean()])
    print(f"  Centroid: {center_vox} voxels = {center_vox * voxel_size} mm")

# Check where DE surface b is non-zero
nz_b = b > 0.01
print(f"\n=== DE Surface b (non-zero >0.01) ===")
print(f"  Count: {nz_b.sum()} / {len(b)}")
if nz_b.sum() > 0:
    nz_nodes = surface_nodes[nz_b]
    print(f"  X: [{nz_nodes[:,0].min():.2f}, {nz_nodes[:,0].max():.2f}] mm")
    print(f"  Y: [{nz_nodes[:,1].min():.2f}, {nz_nodes[:,1].max():.2f}] mm")
    print(f"  Z: [{nz_nodes[:,2].min():.2f}, {nz_nodes[:,2].max():.2f}] mm")

# Check MCX source
mcx_json = json.load(open(f"{sample_dir}/sample_0000.json"))
Pos = np.array(mcx_json['Optode']['Source']['Pos'])
pattern_shape = tuple(mcx_json['Optode']['Source']['Pattern'].values())
src = np.fromfile(f"{sample_dir}/source-sample_0000.bin", dtype=np.float32)
src_3d = src.reshape(pattern_shape)
print(f"\n=== MCX Source ===")
print(f"  Pos (voxels): {Pos}")
print(f"  Pattern shape: {pattern_shape}")
print(f"  Source extent in volume voxels: X=[{Pos[0]}, {Pos[0]+pattern_shape[0]}), Y=[{Pos[1]}, {Pos[1]+pattern_shape[1]}), Z=[{Pos[2]}, {Pos[2]+pattern_shape[2]})")
print(f"  Source extent physical mm: X=[{Pos[0]*voxel_size:.2f}, {(Pos[0]+pattern_shape[0])*voxel_size:.2f}), Y=[{Pos[1]*voxel_size:.2f}, {(Pos[1]+pattern_shape[1])*voxel_size:.2f}), Z=[{Pos[2]*voxel_size:.2f}, {(Pos[2]+pattern_shape[2])*voxel_size:.2f})")

# Check where source=1.0 voxels are
src_binary = src_3d == 1.0
if src_binary.any():
    tx, ty, tz = np.where(src_binary)
    print(f"\n  Source=1 voxels in pattern-local coords:")
    print(f"    X: [{tx.min()}, {tx.max()}] = [{(tx.min()+Pos[0])*voxel_size:.2f}, {(tx.max()+Pos[0])*voxel_size:.2f}] mm")
    print(f"    Y: [{ty.min()}, {ty.max()}] = [{(ty.min()+Pos[1])*voxel_size:.2f}, {(ty.max()+Pos[1])*voxel_size:.2f}] mm")
    print(f"    Z: [{tz.min()}, {tz.max()}] = [{(tz.min()+Pos[2])*voxel_size:.2f}, {(tz.max()+Pos[2])*voxel_size:.2f}] mm")

# Check where MCX fluence is non-zero (high values)
mcx_high = mcx_vol > np.percentile(mcx_vol[mcx_vol > 0], 99)
if mcx_high.any():
    xi, yi, zi = np.where(mcx_high)
    print(f"\n=== MCX Fluence Top 1% ===")
    print(f"  Count: {mcx_high.sum()}")
    print(f"  X: [{xi.min()*voxel_size:.2f}, {xi.max()*voxel_size:.2f}] mm")
    print(f"  Y: [{yi.min()*voxel_size:.2f}, {yi.max()*voxel_size:.2f}] mm")
    print(f"  Z: [{zi.min()*voxel_size:.2f}, {zi.max()*voxel_size:.2f}] mm")

# Now compare projections at 0 degrees
cam_cfg = yaml.safe_load(open("config/default.yaml"))["view_config"]
cam = TurntableCamera(cam_cfg)

# Project gt_voxels directly (not interpolated - just thresholded)
gt_vol = (gt_voxels > 0.5).astype(np.float32)
proj_gt, _ = project_volume_reference(
    gt_vol, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
)
print(f"\n=== Projections at 0° ===")
print(f"gt_voxels>0.5 projected: max={proj_gt.max():.4f}")

# Project MCX surface only
surf_vol = np.zeros_like(mcx_vol)
for edge in [0, -1]:
    surf_vol[edge, :, :] = mcx_vol[edge, :, :]
    surf_vol[:, edge, :] = mcx_vol[:, edge, :]
    surf_vol[:, :, edge] = mcx_vol[:, :, edge]
mcx_surf_norm = surf_vol / max(surf_vol.max(), 1e-6)
proj_mcx_surf, _ = project_volume_reference(
    mcx_surf_norm, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
)
print(f"MCX surface projected (norm): max={proj_mcx_surf.max():.4f}")

# Project MCX full (normalized)
mcx_norm = mcx_vol / max(mcx_vol.max(), 1e-6)
proj_mcx_full, _ = project_volume_reference(
    mcx_norm, 0.0, cam.camera_distance_mm, cam.fov_mm,
    cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
)
print(f"MCX full projected (norm): max={proj_mcx_full.max():.4f}")

# Show side by side
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(proj_gt, cmap="hot")
axes[0].set_title("gt_voxels>0.5 projection")
axes[1].imshow(proj_mcx_surf, cmap="hot")
axes[1].set_title("MCX surface projection")
axes[2].imshow(proj_mcx_full, cmap="hot")
axes[2].set_title("MCX full (norm) projection")
axes[3].imshow(proj_gt - proj_mcx_surf, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
axes[3].set_title("Diff: gt - MCX surf")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/verification/_diag_proj.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: output/verification/_diag_proj.png")

# Also check the IDW interpolation issue - how many surface voxels get values?
print("\n=== IDW Interpolation Check ===")
from scripts.de_surface_to_mcx_projection import interpolate_de_to_mcx_surface, identify_surface_voxels
de_interp_vol = interpolate_de_to_mcx_surface(
    surface_nodes, b, mcx_vol.shape, voxel_size=0.2, radius=3.0
)
print(f"DE interpolated: max={de_interp_vol.max():.6f}, non-zero={np.count_nonzero(de_interp_vol)}")

# What fraction of surface voxels got interpolated?
surface_mask = identify_surface_voxels(mcx_vol.shape)
surf_with_values = np.count_nonzero(de_interp_vol * surface_mask.astype(bool))
surf_total = np.count_nonzero(surface_mask)
print(f"Surface voxels with DE values: {surf_with_values} / {surf_total} ({100*surf_with_values/surf_total:.1f}%)")

# The issue might be that DE surface nodes are in mesh-local coords but MCX is in trunk-local
# Check if nodes are in the same coordinate range
print("\n=== Coordinate Range Check ===")
print(f"Mesh surface nodes: X[{surface_nodes[:,0].min():.2f}, {surface_nodes[:,0].max():.2f}], Y[{surface_nodes[:,1].min():.2f}, {surface_nodes[:,1].max():.2f}], Z[{surface_nodes[:,2].min():.2f}, {surface_nodes[:,2].max():.2f}]")
print(f"MCX vol extent (0.2*shape): X[0, {mcx_vol.shape[0]*voxel_size:.2f}], Y[0, {mcx_vol.shape[1]*voxel_size:.2f}], Z[0, {mcx_vol.shape[2]*voxel_size:.2f}]")