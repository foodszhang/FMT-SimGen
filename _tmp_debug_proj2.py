#!/usr/bin/env python3
"""Better comparison: DE surface fluence vs MCX at matching projection angles."""
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
surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
b = np.load(f"{sample_dir}/measurement_b.npy")

# Camera
cam_cfg = yaml.safe_load(open("config/default.yaml"))["view_config"]
cam = TurntableCamera(cam_cfg)

# Load MCX volume
data = jd.load(f"{sample_dir}/sample_0000.jnii")
nifti = data["NIFTIData"][:, :, :, 0, 0]
mcx_vol = np.transpose(nifti, (2, 1, 0))
mcx_norm = mcx_vol / max(mcx_vol.max(), 1e-6)

# Tumor params
tp = json.load(open(f"{sample_dir}/tumor_params.json"))
print(f"Tumor: {tp.get('num_foci')} foci, source_type={tp.get('source_type')}")
for i, f in enumerate(tp.get('foci', [])):
    print(f"  Focus {i}: center={f['center']}, radius={f.get('radius')}")

# Key insight: both DE and MCX surface fluence result from the same physics.
# We want to verify that the SURFACE PATTERN (spatial distribution on body surface)
# is similar between DE and MCX.

# Strategy: project MCX but ONLY show the surface layer (boundary voxels that face the camera)
# This is what the camera actually sees from each angle.
# DE measurement_b is the surface fluence at the actual mesh surface nodes.

# Let's do a side-by-side at each angle:
# - Row 1: MCX full volume projection (what MCX predicts the camera sees)
# - Row 2: MCX surface-only projection (just boundary voxels)
# - Row 3: DE surface as point cloud projected from surface node positions
# - Row 4: DE b values shown as surface nodes in 3D scatter

angles = [-90, -60, -30, 0, 30, 60, 90]
n = len(angles)

fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

for col, angle in enumerate(angles):
    # Full MCX volume projection
    proj_mcx_full, _ = project_volume_reference(
        mcx_norm, float(angle), cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
    )
    ax = axes[0, col]
    vmax = proj_mcx_full.max() * 0.7
    ax.imshow(proj_mcx_full, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"MCX full {angle}°")
    ax.axis("off")

    # MCX surface-only projection (boundary voxels)
    surf_vol = np.zeros_like(mcx_vol)
    surf_vol[0, :, :] = mcx_vol[0, :, :]
    surf_vol[-1, :, :] = mcx_vol[-1, :, :]
    surf_vol[:, 0, :] = mcx_vol[:, 0, :]
    surf_vol[:, -1, :] = mcx_vol[:, -1, :]
    surf_vol[:, :, 0] = mcx_vol[:, :, 0]
    surf_vol[:, :, -1] = mcx_vol[:, :, -1]
    surf_max = surf_vol.max()
    surf_norm = surf_vol / max(surf_max, 1e-6) if surf_max > 0 else surf_vol

    proj_mcx_surf, _ = project_volume_reference(
        surf_norm, float(angle), cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
    )
    ax = axes[1, col]
    vmax = proj_mcx_surf.max() * 0.7
    ax.imshow(proj_mcx_surf, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"MCX surf {angle}°")
    ax.axis("off")

axes[0, 0].set_ylabel("MCX full vol", fontsize=10)
axes[1, 0].set_ylabel("MCX surf only", fontsize=10)

fig.suptitle(f"MCX projections (sample_0000)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/verification/_debug_mcx_proj.png", dpi=150)
plt.close()
print("Saved: output/verification/_debug_mcx_proj.png")

# Now let's understand the DE measurement_b spatial distribution
# Plot surface nodes colored by b value in 3D
fig = plt.figure(figsize=(12, 4))

# Show measurement_b values at their 3D positions (surface nodes)
ax = fig.add_subplot(131, projection='3d')
# Color by b value
scatter = ax.scatter(surface_nodes[:, 0], surface_nodes[:, 1], surface_nodes[:, 2],
                     c=b, cmap="hot", s=1, alpha=0.5)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("DE surface nodes colored by b value")
plt.colorbar(scatter, ax=ax, shrink=0.5)

# Show only nodes with b > 0.1 (high signal)
ax = fig.add_subplot(132, projection='3d')
high_mask = b > 0.1
ax.scatter(surface_nodes[high_mask, 0], surface_nodes[high_mask, 1], surface_nodes[high_mask, 2],
           c=b[high_mask], cmap="hot", s=5, alpha=0.8)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title(f"DE surface nodes with b > 0.1 ({high_mask.sum()} nodes)")

# Show distribution of b values
ax = fig.add_subplot(133)
ax.hist(b[b > 0], bins=50, color="orange", alpha=0.7)
ax.set_xlabel("b value (fluence)")
ax.set_ylabel("Count")
ax.set_title(f"DE measurement_b distribution (non-zero: {(b>0).sum()})")

plt.tight_layout()
plt.savefig("output/verification/_debug_de_surface.png", dpi=150)
plt.close()
print("Saved: output/verification/_debug_de_surface.png")