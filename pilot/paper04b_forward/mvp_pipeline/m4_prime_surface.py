"""M4' Surface-space inversion with vertex visibility sensitivity.

Architecture: E-F style surface-space fitting (not per-view camera projection).
n_views controls vertex visibility: union of (direct_geo ∩ visible_in_top_K_views).
"""

from __future__ import annotations

import csv
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
from shared.direct_path import is_direct_path_vertex, get_direct_views_for_source_v2

logging.basicConfig(level=logging.WARNING)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

CANDIDATE_ANGLES = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]
CAMERA_PARAMS = {"distance": 200.0, "fov": 50.0, "resolution": (256, 256)}


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])


def project_vertices_to_camera(vertices, angle_deg, camera_params, voxel_size_mm):
    R = rotation_matrix_y(angle_deg)
    rotated = vertices @ R.T
    cam_x, cam_y = rotated[:, 0], rotated[:, 1]
    depths = camera_params["distance"] - rotated[:, 2]

    fov = camera_params["fov"]
    res = camera_params["resolution"]
    px_size = fov / res[0]
    half_w, half_h = fov / 2, fov / 2

    u = ((cam_x + half_w) / px_size).astype(int)
    v = ((cam_y + half_h) / px_size).astype(int)

    valid = (u >= 0) & (u < res[0]) & (v >= 0) & (v < res[1]) & (depths > 0)
    return u, v, depths, valid


def sample_fluence_at_vertices(fluence, vertices, voxel_size_mm):
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices / voxel_size_mm + center).astype(int)
    phi = np.zeros(len(vertices), dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi[i] = fluence[vx, vy, vz]
    return phi


def loss_surface(params, vertices, phi_mcx, direct_mask, optical):
    """Surface-space loss with single scale."""
    source_pos = params[:3]
    r = np.linalg.norm(vertices - source_pos, axis=1)
    forward = G_inf(np.maximum(r, 0.01), optical).astype(np.float32)

    valid = direct_mask & (phi_mcx > 0) & (forward > 0)
    if np.sum(valid) < 50:
        return 1e10

    scale = float(np.sum(phi_mcx[valid]) / np.sum(forward[valid]))
    log_meas = np.log10(phi_mcx[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    return float(np.mean((log_meas - log_fwd) ** 2))


def run_inversion(gt_pos, init_pos, vertices, phi_mcx, direct_mask, optical):
    result = minimize(
        loss_surface,
        init_pos.copy(),
        args=(vertices, phi_mcx, direct_mask, optical),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )
    recovered = result.x[:3]
    pos_err = float(np.linalg.norm(recovered - gt_pos))
    return pos_err, recovered.tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="pilot/paper04b_forward/results/m4_prime_surface"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)

    positions = {
        "P1-dorsal": {
            "pos": np.array([-1.6, 2.4, 5.8]),
            "archive": "S2-Vol-P1-dorsal-r2.0",
        },
        "P5-ventral": {
            "pos": np.array([-0.6, 2.4, -3.8]),
            "archive": "S2-Vol-P5-ventral-r2.0",
        },
        "P2-left": {
            "pos": np.array([-8.0, 1.9, 1.0]),
            "archive": "S2-Vol-P2-left-r2.0",
        },
    }

    results = []
    visibility_levels = [1, 2, 3, "all"]
    n_seeds = 5

    for pos_name, info in positions.items():
        gt_pos = info["pos"]
        fluence = np.load(ARCHIVE_BASE / info["archive"] / "fluence.npy")
        phi_mcx = sample_fluence_at_vertices(fluence, vertices, VOXEL_SIZE_MM)

        is_direct_geo = np.array(
            [is_direct_path_vertex(gt_pos, v, volume, VOXEL_SIZE_MM) for v in vertices]
        )

        direct_views = get_direct_views_for_source_v2(
            gt_pos, CANDIDATE_ANGLES, vertices, volume, VOXEL_SIZE_MM, CAMERA_PARAMS
        )

        print(
            f"\n{pos_name}: {len(direct_views)} direct views: {[a for a, _ in direct_views[:5]]}"
        )

        for K in visibility_levels:
            if K == "all":
                direct_mask = is_direct_geo
                n_verts_used = int(np.sum(direct_mask))
                views_used = "all"
            else:
                if len(direct_views) < K:
                    continue
                top_K_angles = [a for a, _ in direct_views[:K]]

                visible_union = np.zeros(n_vertices, dtype=bool)
                for angle in top_K_angles:
                    _, _, _, valid_proj = project_vertices_to_camera(
                        vertices, angle, CAMERA_PARAMS, VOXEL_SIZE_MM
                    )
                    visible_union |= valid_proj

                direct_mask = is_direct_geo & visible_union
                n_verts_used = int(np.sum(direct_mask))
                views_used = str(top_K_angles)

            if n_verts_used < 100:
                print(f"  K={K}: skipped (only {n_verts_used} vertices)")
                continue

            errors = []
            for seed in range(n_seeds):
                np.random.seed(seed)
                init_pos = gt_pos + np.random.randn(3) * 0.5

                pos_err, recovered = run_inversion(
                    gt_pos, init_pos, vertices, phi_mcx, direct_mask, OPTICAL
                )
                errors.append(pos_err)

                results.append(
                    {
                        "position": pos_name,
                        "K": str(K),
                        "seed": seed,
                        "pos_err_mm": pos_err,
                        "n_verts": n_verts_used,
                        "views": views_used,
                    }
                )

            mean_err = np.mean(errors)
            std_err = np.std(errors)
            print(
                f"  K={K}: {mean_err:.2f} ± {std_err:.2f} mm (n_verts={n_verts_used})"
            )

    with open(output_dir / "m4_prime_surface.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["position", "K", "seed", "pos_err_mm", "n_verts", "views"]
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    with open(output_dir / "m4_prime_surface.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
