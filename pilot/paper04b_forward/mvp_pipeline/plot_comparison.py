"""Generate full comparison plots with original values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "e1b_atlas_mcx_v2"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)
from source_quadrature import sample_uniform

# Archived parameters
VOXEL_SIZE_MM = 0.4
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)
SOURCE_POS = np.array([-0.6, 2.4, 5.8])
SOURCE_RADIUS = 2.0
TISSUE_PARAMS = {"mua_mm": 0.08697, "mus_prime_mm": 4.2907}

# Load archived results
archived_green = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/green_a0.npy"
)
archived_mcx = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/mcx_a0.npy"
)
archived_fluence = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/fluence.npy"
)

# Compute Green on fluence mask surface
fluence_threshold = archived_fluence.max() * 1e-6
fluence_mask = archived_fluence > fluence_threshold

surface_coords, valid_mask = project_get_surface_coords(
    fluence_mask, 0, CAMERA_DISTANCE_MM, FOV_MM, DETECTOR_RESOLUTION, VOXEL_SIZE_MM
)

axes = np.array([SOURCE_RADIUS, SOURCE_RADIUS, SOURCE_RADIUS])
points, weights = sample_uniform(
    center=SOURCE_POS, axes=axes, alpha=1.0, scheme="7-point"
)

my_green = np.zeros(DETECTOR_RESOLUTION[::-1], dtype=np.float32)
for pt, w in zip(points, weights):
    proj_i = green_infinite_point_source_on_surface(
        pt, surface_coords, valid_mask, TISSUE_PARAMS
    )
    my_green += w * proj_i

# Create figure - 2 rows x 3 cols
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Original values (linear scale)
ax = axes[0, 0]
im = ax.imshow(archived_mcx, cmap="hot")
ax.set_title(f"Archived MCX\npeak={archived_mcx.max():.2e}")
plt.colorbar(im, ax=ax, label="Fluence")

ax = axes[0, 1]
im = ax.imshow(archived_green, cmap="hot")
ax.set_title(f"Archived Green\npeak={archived_green.max():.2e}")
plt.colorbar(im, ax=ax, label="G(r)")

ax = axes[0, 2]
im = ax.imshow(my_green, cmap="hot")
ax.set_title(f"My Green (fluence surface)\npeak={my_green.max():.2e}")
plt.colorbar(im, ax=ax, label="G(r)")

# Row 2: Log scale and scatter
ax = axes[1, 0]
im = ax.imshow(np.log10(archived_mcx + 1e-10), cmap="hot")
ax.set_title("MCX (log10)")
plt.colorbar(im, ax=ax, label="log10(Fluence)")

ax = axes[1, 1]
im = ax.imshow(np.log10(archived_green + 1e-10), cmap="hot")
ax.set_title("Green (log10)")
plt.colorbar(im, ax=ax, label="log10(G)")

# Scatter plot with both linear and log
ax = axes[1, 2]
valid = (my_green > 0) & (archived_mcx > 0)
ax.scatter(my_green[valid][::5], archived_mcx[valid][::5], alpha=0.1, s=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Green")
ax.set_ylabel("MCX")
ax.plot([1e-5, 1e-2], [1e-5, 1e-2], "r--", alpha=0.5, label="y=x")
ncc = np.corrcoef(
    np.log10(my_green[valid] + 1e-10), np.log10(archived_mcx[valid] + 1e-10)
)[0, 1]
k = archived_mcx[valid].sum() / my_green[valid].sum()
ax.set_title(f"NCC(log)={ncc:.4f}, k={k:.2e}")
ax.legend()

plt.tight_layout()
output_path = "pilot/paper04b_forward/mvp_pipeline/results/m1_full_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to: {output_path}")
print(f"NCC (log space): {ncc:.4f}")
print(f"k (sum ratio): {k:.2e}")

# Print summary
print(f"\\n=== Summary ===")
print(
    f"MCX: peak={archived_mcx.max():.2e}, sum={archived_mcx.sum():.2e}, non-zero={np.sum(archived_mcx > 0)}"
)
print(
    f"Green: peak={my_green.max():.2e}, sum={my_green.sum():.2e}, non-zero={np.sum(my_green > 0)}"
)
