#!/usr/bin/env python3
"""Debug projection differences."""

import numpy as np
import jdata as jd
from pathlib import Path

# Import from working script
import sys

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import rotation_matrix_y


def project_mcx(
    fluence_xyz,
    angle_deg,
    camera_distance_mm,
    fov_mm,
    detector_resolution,
    voxel_size_mm,
):
    """Project MCX 3D fluence to 2D."""
    nx, ny, nz = fluence_xyz.shape
    width, height = detector_resolution

    nonzero = np.argwhere(fluence_xyz > 0)
    if len(nonzero) == 0:
        return np.zeros((height, width), dtype=np.float32)

    center = np.array([nx / 2, ny / 2, nz / 2])
    coords_mm = (nonzero.astype(np.float32) - center + 0.5) * voxel_size_mm
    values = fluence_xyz[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]

    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        coords_rot = coords_mm @ R.T
    else:
        coords_rot = coords_mm

    cam_x = coords_rot[:, 0]
    cam_y = coords_rot[:, 1]
    depths = camera_distance_mm - coords_rot[:, 2]

    half_w = fov_mm / 2
    half_h = fov_mm / 2
    px_size_x = fov_mm / width
    px_size_y = fov_mm / height

    projection = np.zeros((height, width), dtype=np.float32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    half_voxel = voxel_size_mm / 2

    for idx in range(len(cam_x)):
        px, py = cam_x[idx], cam_y[idx]
        d = depths[idx]

        if abs(px) > half_w or abs(py) > half_h or d < 0:
            continue

        u_start = int((px - half_voxel + half_w) / px_size_x)
        u_end = int((px + half_voxel + half_w) / px_size_x)
        v_start = int((py - half_voxel + half_h) / px_size_y)
        v_end = int((py + half_voxel + half_h) / px_size_y)

        u_start = max(0, u_start)
        u_end = min(width - 1, u_end)
        v_start = max(0, v_start)
        v_end = min(height - 1, v_end)

        for pu in range(u_start, u_end + 1):
            for pv in range(v_start, v_end + 1):
                if d < depth_map[pv, pu]:
                    depth_map[pv, pu] = d
                    projection[pv, pu] = values[idx]

    return projection


def main():
    # Load both MCX outputs
    work_data = jd.loadjd("results/stage1_5_surface/S1.5-D4mm/S1.5-D4mm.jnii")
    work_fluence = np.array(work_data["NIFTIData"], dtype=np.float32)[
        ..., 0, 0
    ].transpose(2, 1, 0)

    broken_data = jd.loadjd("results/multiposition/P1-dorsal/P1-dorsal.jnii")
    broken_fluence = np.array(broken_data["NIFTIData"], dtype=np.float32)[
        ..., 0, 0
    ].transpose(2, 1, 0)

    print(f"Work fluence: {work_fluence.shape}, max={work_fluence.max():.4e}")
    print(f"Broken fluence: {broken_fluence.shape}, max={broken_fluence.max():.4e}")

    # Project both at 0° using same code
    camera_distance = 200.0
    fov_mm = 50.0
    resolution = (256, 256)
    voxel_size = 0.2

    work_proj = project_mcx(
        work_fluence, 0.0, camera_distance, fov_mm, resolution, voxel_size
    )
    broken_proj = project_mcx(
        broken_fluence, 0.0, camera_distance, fov_mm, resolution, voxel_size
    )

    print(
        f"\nWork projection: max={work_proj.max():.4e}, peak at {np.unravel_index(np.argmax(work_proj), work_proj.shape)}"
    )
    print(
        f"Broken projection: max={broken_proj.max():.4e}, peak at {np.unravel_index(np.argmax(broken_proj), broken_proj.shape)}"
    )

    # Load saved projections
    work_saved = np.load("results/stage1_5_surface/S1.5-D4mm/mcx_projection_a0.npy")
    broken_saved = np.load("results/multiposition/P1-dorsal/mcx_a0.npy")

    print(
        f"\nWork saved: max={work_saved.max():.4e}, peak at {np.unravel_index(np.argmax(work_saved), work_saved.shape)}"
    )
    print(
        f"Broken saved: max={broken_saved.max():.4e}, peak at {np.unravel_index(np.argmax(broken_saved), broken_saved.shape)}"
    )

    # Check if saved matches recomputed
    work_diff = np.abs(work_proj - work_saved).max()
    broken_diff = np.abs(broken_proj - broken_saved).max()
    print(f"\nDifference between recomputed and saved:")
    print(f"  Work: {work_diff:.4e}")
    print(f"  Broken: {broken_diff:.4e}")


if __name__ == "__main__":
    main()
