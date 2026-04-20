"""E-G: Optical prior perturbation."""

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


def forward_with_mua_musp(pos, mu_a, mus_p, vertices):
    D = 1.0 / (3.0 * (mu_a + mus_p))
    delta = np.sqrt(D / mu_a)
    dx = vertices[:, 0] - pos[0]
    dy = vertices[:, 1] - pos[1]
    dz = vertices[:, 2] - pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    return np.exp(-r / delta) / (4.0 * np.pi * D * r)


def loss_with_fixed_optical(params, vertices, measurement, direct_mask, mu_a, mus_p):
    pos = params[:3]
    forward = forward_with_mua_musp(pos, mu_a, mus_p, vertices)

    valid = direct_mask & (measurement > 0) & (forward > 0)
    if np.sum(valid) < 10:
        return 1e10

    scale = np.sum(measurement[valid]) / np.sum(forward[valid])
    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    return float(np.mean((log_meas - log_fwd) ** 2))


def run_with_perturbed_optical(
    gt_pos,
    mu_a_factor,
    musp_factor,
    vertices,
    measurement,
    direct_mask,
    gt_mua,
    gt_musp,
):
    perturbed_mua = gt_mua * mu_a_factor
    perturbed_musp = gt_musp * musp_factor

    init_pos = gt_pos + np.random.randn(3) * 0.5

    result = minimize(
        loss_with_fixed_optical,
        init_pos,
        args=(vertices, measurement, direct_mask, perturbed_mua, perturbed_musp),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )

    recovered_pos = result.x[:3]
    pos_err = np.linalg.norm(recovered_pos - gt_pos)

    return {
        "mu_a_factor": mu_a_factor,
        "musp_factor": musp_factor,
        "pos_err_mm": float(pos_err),
    }


def main():
    output_dir = Path("pilot/paper04b_forward/results/eg_optical_prior")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    center = np.array([95, 100, 52]) / 2

    gt_pos = np.array([-0.6, 2.4, -3.8])
    gt_mua = 0.087
    gt_musp = 4.3

    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    phi_mcx = np.zeros(n_vertices, dtype=np.float32)
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx[i] = fluence[vx, vy, vz]

    is_direct = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        is_direct[i] = is_direct_path_vertex(gt_pos, vertices[i], volume, VOXEL_SIZE_MM)

    results = []

    for mu_a_factor in [0.5, 1.0, 2.0]:
        for musp_factor in [0.5, 1.0, 2.0]:
            for seed in range(5):
                np.random.seed(seed)
                r = run_with_perturbed_optical(
                    gt_pos,
                    mu_a_factor,
                    musp_factor,
                    vertices,
                    phi_mcx,
                    is_direct,
                    gt_mua,
                    gt_musp,
                )
                r["seed"] = seed
                results.append(r)
                print(
                    f"μ_a×{mu_a_factor}, μs'×{musp_factor}, seed {seed}: pos_err={r['pos_err_mm']:.3f}mm"
                )

    with open(output_dir / "eg_optical_prior.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["mu_a_factor", "musp_factor", "seed", "pos_err_mm"]
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
