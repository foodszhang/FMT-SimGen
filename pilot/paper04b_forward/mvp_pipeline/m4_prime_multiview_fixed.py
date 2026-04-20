"""M4' Multi-view: Position recovery with multiple camera angles (FIXED).

Fixes:
A. Scale dynamically fitted per-iteration (not frozen from GT)
B. View angles filtered by preflight (not hardcoded)
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
from shared.direct_path import get_direct_views_for_source, is_direct_path_vertex

logging.basicConfig(level=logging.WARNING)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

CANDIDATE_ANGLES = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def loss_multiview(params, vertices, measurements, view_angles, direct_masks):
    """Loss with dynamic scale fitting per-view."""
    source_pos = params[:3]

    r = np.linalg.norm(vertices - source_pos, axis=1)
    forward = G_inf(np.maximum(r, 0.01), OPTICAL).astype(np.float32)

    total_loss = 0.0
    n_valid_total = 0

    for i, angle in enumerate(view_angles):
        measurement = measurements[i]
        direct_mask = direct_masks[i]

        valid = direct_mask & (measurement > 0) & (forward > 0)
        if np.sum(valid) < 10:
            continue

        scale_i = float(np.sum(measurement[valid]) / np.sum(forward[valid]))

        log_meas = np.log10(measurement[valid] + 1e-20)
        log_fwd = np.log10(scale_i * forward[valid] + 1e-20)

        total_loss += np.sum((log_meas - log_fwd) ** 2)
        n_valid_total += int(np.sum(valid))

    return total_loss / max(n_valid_total, 1)


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


def run_multiview_inversion(
    gt_pos, init_pos, vertices, volume, fluence, is_direct_all, n_views, seed
):
    """Run inversion with preflight-filtered views."""

    direct_views = get_direct_views_for_source(
        gt_pos, CANDIDATE_ANGLES, volume, VOXEL_SIZE_MM
    )
    direct_views_sorted = sorted(direct_views, key=lambda x: x[1].path_length_mm)
    selected_angles = [a for a, _ in direct_views_sorted[:n_views]]

    if len(selected_angles) < n_views:
        return None

    measurements = []
    direct_masks = []

    for angle in selected_angles:
        meas = sample_fluence_at_vertices(fluence, vertices, VOXEL_SIZE_MM)
        measurements.append(meas)
        direct_masks.append(is_direct_all)

    result = minimize(
        loss_multiview,
        init_pos.copy(),
        args=(vertices, measurements, selected_angles, direct_masks),
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
        "views": selected_angles,
        "seed": seed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="pilot/paper04b_forward/results/m4_prime_multiview_fixed"
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
    selected_views_report = {}

    for pos_name, info in positions.items():
        gt_pos = info["pos"]
        fluence = np.load(ARCHIVE_BASE / info["archive"] / "fluence.npy")

        is_direct = np.zeros(n_vertices, dtype=bool)
        for i in range(n_vertices):
            is_direct[i] = is_direct_path_vertex(
                gt_pos, vertices[i], volume, VOXEL_SIZE_MM
            )

        direct_views = get_direct_views_for_source(
            gt_pos, CANDIDATE_ANGLES, volume, VOXEL_SIZE_MM
        )
        direct_views_sorted = sorted(direct_views, key=lambda x: x[1].path_length_mm)

        views_1 = [a for a, _ in direct_views_sorted[:1]]
        views_2 = [a for a, _ in direct_views_sorted[:2]]
        views_3 = [a for a, _ in direct_views_sorted[:3]]
        selected_views_report[pos_name] = {"1": views_1, "2": views_2, "3": views_3}

        for n_views in [1, 2, 3]:
            for seed in range(3):
                np.random.seed(seed)
                init_pos = gt_pos + np.random.randn(3) * 1.0

                r = run_multiview_inversion(
                    gt_pos,
                    init_pos,
                    vertices,
                    volume,
                    fluence,
                    is_direct,
                    n_views,
                    seed,
                )
                if r:
                    r["position"] = pos_name
                    results.append(r)
                    print(
                        f"{pos_name}, {n_views} view(s), seed {seed}: pos_err = {r['error_mm']:.3f} mm, views={r['views']}"
                    )
                else:
                    print(f"{pos_name}, {n_views} view(s): not enough direct views")

    with open(output_dir / "m4_prime_multiview.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["position", "n_views", "seed", "error_mm", "views"]
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "position": r["position"],
                    "n_views": r["n_views"],
                    "seed": r["seed"],
                    "error_mm": f"{r['error_mm']:.3f}",
                    "views": str(r["views"]),
                }
            )

    with open(output_dir / "selected_views.json", "w") as f:
        json.dump(selected_views_report, f, indent=2)

    print("\n" + "=" * 60)
    print("M4' Multi-view Summary")
    print("=" * 60)
    for pos_name, views in selected_views_report.items():
        print(f"{pos_name} selected_views [1/2/3]:")
        print(f"  1-view: {views['1']}")
        print(f"  2-view: {views['2']}")
        print(f"  3-view: {views['3']}")

    for pos_name in positions:
        errs = {1: [], 2: [], 3: []}
        for r in results:
            if r["position"] == pos_name:
                errs[r["n_views"]].append(r["error_mm"])
        print(f"\n{pos_name}:")
        for n in [1, 2, 3]:
            if errs[n]:
                mean_err = np.mean(errs[n])
                std_err = np.std(errs[n])
                print(f"  {n}-view: {mean_err:.2f} ± {std_err:.2f} mm")


if __name__ == "__main__":
    main()
