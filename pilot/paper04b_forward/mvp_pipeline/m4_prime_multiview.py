"""M4' Multi-view: Position recovery with multiple camera angles."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.green_surface_projection import rotation_matrix_y

logging.basicConfig(level=logging.WARNING)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)

SOFT_TISSUE_LABEL = 1
AIR_LABEL = 0


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def is_direct_path_vertex(source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm):
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return True
    direction = direction / distance
    step_mm = 0.1
    n_steps = int(distance / step_mm)

    for i in range(1, n_steps + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = mm_to_voxel(pos_mm)
        if not (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        ):
            break
        if volume_labels[voxel[0], voxel[1], voxel[2]] not in {
            AIR_LABEL,
            SOFT_TISSUE_LABEL,
        }:
            return False
    return True


def project_vertices_to_camera(
    vertices_mm, angle_deg, camera_dist_mm, fov_mm, resolution, voxel_size_mm
):
    R = rotation_matrix_y(angle_deg)
    rotated = vertices_mm @ R.T
    cam_x, cam_y, depths = rotated[:, 0], rotated[:, 1], camera_dist_mm - rotated[:, 2]

    half_w, half_h = fov_mm / 2, fov_mm / 2
    px_size = fov_mm / resolution

    u = ((cam_x + half_w) / px_size).astype(int)
    v = ((cam_y + half_h) / px_size).astype(int)

    valid = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution) & (depths > 0)
    return u, v, depths, valid


def loss_multiview(
    params, vertices, measurements, scales, view_angles, direct_masks, camera_params
):
    source_pos = params[:3]
    total_loss = 0.0
    n_valid = 0

    for i, angle in enumerate(view_angles):
        measurement = measurements[i]
        scale = scales[i]
        direct_mask = direct_masks[i]

        u, v, depths, valid_proj = project_vertices_to_camera(
            vertices,
            angle,
            camera_params["distance"],
            camera_params["fov"],
            camera_params["resolution"],
            VOXEL_SIZE_MM,
        )

        vertex_valid = direct_mask & valid_proj & (measurement > 0)
        if np.sum(vertex_valid) < 10:
            continue

        dx = vertices[:, 0] - source_pos[0]
        dy = vertices[:, 1] - source_pos[1]
        dz = vertices[:, 2] - source_pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 0.01)
        forward = G_inf(r, OPTICAL).astype(np.float32)

        log_meas = np.log10(measurement[vertex_valid] + 1e-20)
        log_fwd = np.log10(scale * forward[vertex_valid] + 1e-20)
        total_loss += np.sum((log_meas - log_fwd) ** 2)
        n_valid += np.sum(vertex_valid)

    return total_loss / max(n_valid, 1)


def run_multiview_inversion(
    gt_pos, init_pos, vertices, volume, fluence, direct_mask_all, n_views, output_dir
):
    camera_params = {"distance": 200.0, "fov": 50.0, "resolution": 256}

    center = np.array(fluence.shape) / 2

    if n_views == 1:
        view_angles = [0]
    elif n_views == 2:
        view_angles = [0, -90]
    else:
        view_angles = [0, 90, -90]

    measurements = []
    scales = []
    direct_masks = []

    for angle in view_angles:
        R = rotation_matrix_y(angle)

        u, v, depths, valid_proj = project_vertices_to_camera(
            vertices,
            angle,
            camera_params["distance"],
            camera_params["fov"],
            camera_params["resolution"],
            VOXEL_SIZE_MM,
        )

        meas = np.zeros(len(vertices), dtype=np.float32)
        verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)

        for i in range(len(vertices)):
            if valid_proj[i]:
                vx, vy, vz = verts_voxel[i]
                if (
                    0 <= vx < fluence.shape[0]
                    and 0 <= vy < fluence.shape[1]
                    and 0 <= vz < fluence.shape[2]
                ):
                    meas[i] = fluence[vx, vy, vz]

        measurements.append(meas)

        valid = direct_mask_all & valid_proj & (meas > 0)
        dx = vertices[:, 0] - gt_pos[0]
        dy = vertices[:, 1] - gt_pos[1]
        dz = vertices[:, 2] - gt_pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        forward_gt = G_inf(np.maximum(r, 0.01), OPTICAL)

        if np.sum(valid) > 10:
            scales.append(float(np.sum(meas[valid]) / np.sum(forward_gt[valid])))
        else:
            scales.append(1e6)

        direct_masks.append(direct_mask_all & valid_proj)

    result = minimize(
        loss_multiview,
        init_pos.copy(),
        args=(vertices, measurements, scales, view_angles, direct_masks, camera_params),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )

    recovered = result.x[:3]
    pos_err = np.linalg.norm(recovered - gt_pos)

    return {
        "gt": gt_pos.tolist(),
        "init": init_pos.tolist(),
        "recovered": recovered.tolist(),
        "error_mm": float(pos_err),
        "n_views": n_views,
        "views": view_angles,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="pilot/paper04b_forward/results/m4_prime_multiview"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)

    np.random.seed(42)

    positions = {"P5-ventral": np.array([-0.6, 2.4, -3.8])}
    fluence = np.load(
        "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P5-ventral-r2.0/fluence.npy"
    )

    results = []

    for pos_name, gt_pos in positions.items():
        is_direct = np.zeros(n_vertices, dtype=bool)
        for i in range(n_vertices):
            is_direct[i] = is_direct_path_vertex(
                gt_pos, vertices[i], volume, VOXEL_SIZE_MM
            )

        for n_views in [1, 2, 3]:
            for trial in range(3):
                init_pos = gt_pos + np.random.randn(3) * 1.0
                r = run_multiview_inversion(
                    gt_pos,
                    init_pos,
                    vertices,
                    volume,
                    fluence,
                    is_direct,
                    n_views,
                    output_dir,
                )
                r["trial"] = trial
                results.append(r)
                print(
                    f"P5-ventral, {n_views} views, trial {trial}: pos_err = {r['error_mm']:.3f} mm"
                )

    import csv

    with open(output_dir / "m4_prime_multiview.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["position", "n_views", "trial", "error_mm", "views"]
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "position": "P5-ventral",
                    "n_views": r["n_views"],
                    "trial": r["trial"],
                    "error_mm": f"{r['error_mm']:.3f}",
                    "views": str(r["views"]),
                }
            )

    with open(output_dir / "m4_prime_multiview.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
