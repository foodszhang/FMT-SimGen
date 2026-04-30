#!/usr/bin/env python3
"""Proper surface-to-surface comparison: DE measurement_b vs MCX surface fluence.

The key comparison is:
- DE surface: measurement_b (FEM forward solution at mesh surface nodes)
- MCX surface: MCX fluence at volume boundary voxels, evaluated at same physical positions

Both represent surface exiting fluence due to internal source.
"""
import numpy as np
import json
import yaml
import jdata as jd
import matplotlib.pyplot as plt
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD
from fmt_simgen.mcx_projection import project_volume_reference

sample_dir = "data/small_uniform_5samples/samples/sample_0000"
voxel_size = 0.2

# Load DE surface data
mesh = np.load("output/shared/mesh.npz")
surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
b = np.load(f"{sample_dir}/measurement_b.npy")  # DE surface fluence

# Load MCX fluence volume
data = jd.load(f"{sample_dir}/sample_0000.jnii")
nifti = data["NIFTIData"][:, :, :, 0, 0]
mcx_vol = np.transpose(nifti, (2, 1, 0))  # (X=190, Y=200, Z=104)

# Load tumor params
tp = json.load(open(f"{sample_dir}/tumor_params.json"))

print(f"=== Data ===")
print(f"DE surface nodes: {len(surface_nodes)}, b range [{b.min():.4f}, {b.max():.4f}]")
print(f"MCX vol: {mcx_vol.shape}, max fluence {mcx_vol.max():.2f}")
print(f"Tumor foci: {len(tp['foci'])}, source_type={tp['source_type']}")

# ============================================================
# Step 1: Compare at SAME PHYSICAL POSITIONS
# For each surface node, get DE b and MCX fluence at that position
# ============================================================
mcx_at_de_positions = []
for nd in surface_nodes:
    vx = int(np.clip(round(nd[0] / voxel_size), 0, mcx_vol.shape[0]-1))
    vy = int(np.clip(round(nd[1] / voxel_size), 0, mcx_vol.shape[1]-1))
    vz = int(np.clip(round(nd[2] / voxel_size), 0, mcx_vol.shape[2]-1))
    mcx_at_de_positions.append(mcx_vol[vx, vy, vz])
mcx_at_de_positions = np.array(mcx_at_de_positions)

# Normalize MCX to 0-1 for comparison (only pattern matters, not absolute scale)
mcx_norm_at_de = mcx_at_de_positions / max(mcx_at_de_positions.max(), 1e-9)
b_norm = b / max(b.max(), 1e-9)

# Correlation
nz = (b > 0.001) & (mcx_at_de_positions > 0)
corr = np.corrcoef(b[nz], mcx_norm_at_de[nz])[0, 1] if nz.sum() > 100 else 0
print(f"\n=== Position-wise comparison (normalized) ===")
print(f"Correlation at surface positions: {corr:.4f} (n={nz.sum()})")

# ============================================================
# Step 2: Visualize the comparison
# ============================================================
cam_cfg = yaml.safe_load(open("config/default.yaml"))["view_config"]
cam = TurntableCamera(cam_cfg)
angles = [-90, -60, -30, 0, 30, 60, 90]

fig, axes = plt.subplots(4, len(angles), figsize=(3*len(angles), 12))

for col, angle in enumerate(angles):
    # DE surface b projected (b values at surface nodes → volume → project)
    de_vol = np.zeros_like(mcx_vol)
    for nd, bv in zip(surface_nodes, b):
        vx = int(np.clip(round(nd[0]/voxel_size), 0, mcx_vol.shape[0]-1))
        vy = int(np.clip(round(nd[1]/voxel_size), 0, mcx_vol.shape[1]-1))
        vz = int(np.clip(round(nd[2]/voxel_size), 0, mcx_vol.shape[2]-1))
        de_vol[vx, vy, vz] = max(de_vol[vx, vy, vz], bv)

    proj_de, _ = project_volume_reference(
        de_vol, float(angle), cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
    )

    # MCX surface-only projection (just boundary voxels)
    surf_vol = np.zeros_like(mcx_vol)
    surf_vol[0, :, :] = mcx_vol[0, :, :]
    surf_vol[-1, :, :] = mcx_vol[-1, :, :]
    surf_vol[:, 0, :] = mcx_vol[:, 0, :]
    surf_vol[:, -1, :] = mcx_vol[:, -1, :]
    surf_vol[:, :, 0] = mcx_vol[:, :, 0]
    surf_vol[:, :, -1] = mcx_vol[:, :, -1]

    proj_mcx_surf, _ = project_volume_reference(
        surf_vol, float(angle), cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, voxel_size, VOLUME_CENTER_WORLD
    )
    proj_mcx_surf_norm = proj_mcx_surf / max(proj_mcx_surf.max(), 1e-9)

    axes[0, col].imshow(proj_de, cmap="hot")
    axes[0, col].set_title(f"{angle}°")
    axes[0, col].axis("off")

    axes[1, col].imshow(proj_mcx_surf_norm, cmap="hot")
    axes[1, col].axis("off")

    diff = proj_de - proj_mcx_surf_norm
    v = max(abs(diff.min()), abs(diff.max()), 0.05)
    axes[2, col].imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[2, col].axis("off")

    # Row 3: line profile across center
    h = proj_de.shape[0] // 2
    axes[3, col].plot(proj_de[h, :], label="DE", color="orange")
    axes[3, col].plot(proj_mcx_surf_norm[h, :] * proj_de.max(), label="MCX (scaled)", color="blue", alpha=0.7)
    axes[3, col].legend(fontsize=6)
    axes[3, col].axis("off")

for row, label in enumerate(["DE surface→vol→proj", "MCX surface→proj (norm)", "Difference", "Profile @ center"]):
    axes[row, 0].set_ylabel(label, fontsize=8, rotation=90, labelpad=3)

fig.suptitle(f"Surface-to-Surface: DE b vs MCX surface fluence (sample_0000)\n"
             f"Correlation at positions: {corr:.4f}", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/verification/_debug_surface_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: output/verification/_debug_surface_surface.png")

# ============================================================
# Step 3: Check if the issue is that MCX surface is measuring
# EXITANCE (fluence * absorption) rather than just fluence.
# The DE forward solution gives the RADIANCE at the surface.
# ============================================================
print("\n=== Surface voxel analysis ===")
# MCX surface voxels
for boundary, name in [(0, "X_min"), (-1, "X_max"), (0, "Y_min"), (-1, "Y_max"), (0, "Z_min"), (-1, "Z_max")]:
    pass  # skip for now

# Key diagnostic: scatter plot at matching positions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(b, mcx_norm_at_de, alpha=0.1, s=2)
ax.set_xlabel("DE b (normalized)")
ax.set_ylabel("MCX fluence at DE pos (normalized)")
ax.set_title(f"DE b vs MCX@DE_pos (corr={corr:.4f})")
ax.plot([0, 1], [0, 1], 'r--', label="y=x")
ax.legend()

ax = axes[1]
# Log scale
mask = (b > 0) & (mcx_norm_at_de > 0)
ax.scatter(b[mask], mcx_norm_at_de[mask], alpha=0.1, s=2)
ax.set_xlabel("DE b (normalized)")
ax.set_ylabel("MCX fluence at DE pos (normalized)")
ax.set_title("Same, log scale")
ax.set_xscale("log")
ax.set_yscale("log")

plt.tight_layout()
plt.savefig("output/verification/_debug_scatter.png", dpi=150)
plt.close()
print("Saved: output/verification/_debug_scatter.png")