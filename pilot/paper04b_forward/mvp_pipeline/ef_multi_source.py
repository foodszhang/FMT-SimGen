"""E-F: Multi-source scale-up."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

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


def forward_multi_source(positions, vertices, optical):
    total = np.zeros(len(vertices), dtype=np.float32)
    for pos in positions:
        dx = vertices[:, 0] - pos[0]
        dy = vertices[:, 1] - pos[1]
        dz = vertices[:, 2] - pos[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 0.01)
        total += G_inf(r, optical).astype(np.float32)
    return total


def loss_multi_source(params, vertices, measurement, direct_mask, n_sources, optical):
    positions = params.reshape(n_sources, 3)
    forward = forward_multi_source(positions, vertices, optical)

    valid = direct_mask & (measurement > 0) & (forward > 0)
    if np.sum(valid) < 10:
        return 1e10

    scale = np.sum(measurement[valid]) / np.sum(forward[valid])
    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    return float(np.mean((log_meas - log_fwd) ** 2))


def run_multi_source_inversion(
    gt_positions, init_positions, vertices, measurement, direct_mask, optical
):
    n_sources = len(gt_positions)
    init_params = np.array(init_positions).flatten()

    result = minimize(
        loss_multi_source,
        init_params,
        args=(vertices, measurement, direct_mask, n_sources, optical),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )

    recovered = result.x.reshape(n_sources, 3)

    errors = [np.linalg.norm(recovered[i] - gt_positions[i]) for i in range(n_sources)]

    return {
        "n_sources": n_sources,
        "mean_pos_err_mm": float(np.mean(errors)),
        "max_pos_err_mm": float(max(errors)),
        "errors_mm": errors,
    }


def main():
    output_dir = Path("pilot/paper04b_forward/results/ef_multi_source")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    center = np.array([95, 100, 52]) / 2

    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    phi_mcx = np.zeros(n_vertices, dtype=np.float32)
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx[i] = fluence[vx, vy, vz]

    results = []

    for n_sources in [1, 2, 3, 5]:
        np.random.seed(42)

        gt_positions = [np.array([-0.6, 2.4, -3.8])]
        spacing = 4.0
        for i in range(1, n_sources):
            offset = np.array([(i % 2) * spacing - spacing / 2, (i // 2) * spacing, 0])
            gt_positions.append(gt_positions[0] + offset)

        all_direct = np.zeros(n_vertices, dtype=bool)
        for pos in gt_positions:
            for i in range(n_vertices):
                if is_direct_path_vertex(pos, vertices[i], volume, VOXEL_SIZE_MM):
                    all_direct[i] = True

        init_positions = [p + np.random.randn(3) * 0.5 for p in gt_positions]

        forward_gt = forward_multi_source(gt_positions, vertices, OPTICAL)

        r = run_multi_source_inversion(
            gt_positions, init_positions, vertices, phi_mcx, all_direct, OPTICAL
        )
        r["n_direct_vertices"] = int(np.sum(all_direct))
        results.append(r)
        print(
            f"N={n_sources}: mean_err={r['mean_pos_err_mm']:.3f}mm, max_err={r['max_pos_err_mm']:.3f}mm, n_direct={r['n_direct_vertices']}"
        )

    with open(output_dir / "ef_multi_source.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_sources",
                "mean_pos_err_mm",
                "max_pos_err_mm",
                "n_direct_vertices",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    k: f"{v:.3f}" if isinstance(v, float) else v
                    for k, v in r.items()
                    if k in w.fieldnames
                }
            )


if __name__ == "__main__":
    main()
