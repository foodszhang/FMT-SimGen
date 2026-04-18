"""Debug script to verify MCX vs Green comparison using archived volume."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "e1b_atlas_mcx_v2"))

import numpy as np
from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)
from source_quadrature import sample_uniform
from fmt_simgen.mcx_projection import project_volume_reference
import jdata as jd

# Archived volume parameters
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
VOLUME_SHAPE_ZYX = (52, 100, 95)
VOXEL_SIZE_MM = 0.4
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)

# Source parameters (from archived config)
SOURCE_POS = np.array([-0.6, 2.4, 5.8])
SOURCE_RADIUS = 2.0
ANGLE_DEG = 0

# Tissue params
TISSUE_PARAMS = {
    "mua_mm": 0.08697,
    "mus_prime_mm": 4.2907,
}

# Load volume
volume = np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_ZYX)
binary_mask = (volume > 0).astype(np.uint8)
print(f"Volume shape (ZYX): {volume.shape}, tissue: {np.sum(binary_mask)}")

# Compute surface coords
surface_coords, valid_mask = project_get_surface_coords(
    binary_mask,
    ANGLE_DEG,
    CAMERA_DISTANCE_MM,
    FOV_MM,
    DETECTOR_RESOLUTION,
    VOXEL_SIZE_MM,
)
print(f"Valid surface pixels: {np.sum(valid_mask)}")

# Compute Green (7-point)
axes = np.array([SOURCE_RADIUS, SOURCE_RADIUS, SOURCE_RADIUS])
points, weights = sample_uniform(
    center=SOURCE_POS, axes=axes, alpha=1.0, scheme="7-point"
)

green_proj = np.zeros(DETECTOR_RESOLUTION[::-1], dtype=np.float32)
for pt, w in zip(points, weights):
    proj_i = green_infinite_point_source_on_surface(
        pt, surface_coords, valid_mask, TISSUE_PARAMS
    )
    green_proj += w * proj_i

print(
    f"My Green: peak={green_proj.max():.4e}, sum={green_proj.sum():.4e}, non-zero={np.sum(green_proj > 0)}"
)

# Load archived Green
archived_green = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/green_a0.npy"
)
print(
    f"Archived Green: peak={archived_green.max():.4e}, sum={archived_green.sum():.4e}, non-zero={np.sum(archived_green > 0)}"
)

# Load archived MCX fluence and project
archived_fluence = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/fluence.npy"
)
print(f"Archived fluence shape: {archived_fluence.shape}")

# Project fluence (fluence is in XYZ order)
mcx_proj, _ = project_volume_reference(
    archived_fluence,
    ANGLE_DEG,
    CAMERA_DISTANCE_MM,
    FOV_MM,
    DETECTOR_RESOLUTION,
    VOXEL_SIZE_MM,
)
print(f"Projected MCX: peak={mcx_proj.max():.2e}, non-zero={np.sum(mcx_proj > 0)}")

# Load archived MCX projection
archived_mcx = np.load(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/mcx_a0.npy"
)
print(
    f"Archived MCX: peak={archived_mcx.max():.2e}, non-zero={np.sum(archived_mcx > 0)}"
)

# NCC
valid = (green_proj > 0) & (archived_mcx > 0)
ncc_my_green = np.corrcoef(green_proj[valid].flatten(), archived_mcx[valid].flatten())[
    0, 1
]
ncc_archived_green = np.corrcoef(
    archived_green[valid].flatten(), archived_mcx[valid].flatten()
)[0, 1]
print(f"\nNCC (my Green vs archived MCX): {ncc_my_green:.4f}")
print(f"NCC (archived Green vs archived MCX): {ncc_archived_green:.4f}")
