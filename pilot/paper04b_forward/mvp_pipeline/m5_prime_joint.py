"""M5' + E-E: Joint μ + source inversion with multi-view."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def forward_with_params(pos, mu_a, mus_p, vertices):
    D = 1.0 / (3.0 * (mu_a + mus_p))
    delta = np.sqrt(D / mu_a)

    dx = vertices[:, 0] - pos[0]
    dy = vertices[:, 1] - pos[1]
    dz = vertices[:, 2] - pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)

    return np.exp(-r / delta) / (4.0 * np.pi * D * r)


def loss_joint(params, vertices, measurement, direct_mask):
    pos = params[:3]
    mu_a = params[3]
    mus_p = params[4]

    if mu_a < 0.01 or mus_p < 1.0:
        return 1e10

    forward = forward_with_params(pos, mu_a, mus_p, vertices)

    valid = direct_mask & (measurement > 0) & (forward > 0)
    if np.sum(valid) < 10:
        return 1e10

    scale = np.sum(measurement[valid]) / np.sum(forward[valid])
    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    return float(np.mean((log_meas - log_fwd) ** 2))


def run_joint_inversion(
    gt_pos,
    gt_mua,
    gt_musp,
    init_pos,
    init_mua,
    init_musp,
    vertices,
    measurement,
    direct_mask,
):
    init_params = np.array([*init_pos, init_mua, init_musp])

    result = minimize(
        loss_joint,
        init_params,
        args=(vertices, measurement, direct_mask),
        method="L-BFGS-B",
        bounds=[(-20, 20), (-20, 20), (-20, 20), (0.01, 1.0), (0.5, 20.0)],
        options={"maxiter": 200},
    )

    recovered_pos = result.x[:3]
    recovered_mua = result.x[3]
    recovered_musp = result.x[4]

    pos_err = np.linalg.norm(recovered_pos - gt_pos)
    mua_err = abs(recovered_mua - gt_mua) / gt_mua
    musp_err = abs(recovered_musp - gt_musp) / gt_musp

    return {
        "pos_err_mm": float(pos_err),
        "mua_err_pct": float(mua_err * 100),
        "musp_err_pct": float(musp_err * 100),
        "recovered_pos": recovered_pos.tolist(),
        "recovered_mua": float(recovered_mua),
        "recovered_musp": float(recovered_musp),
    }


def main():
    output_dir = Path("pilot/paper04b_forward/results/m5_prime_joint")
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
    np.random.seed(42)

    for seed in range(5):
        np.random.seed(seed)
        init_pos = gt_pos + np.random.randn(3) * 1.0
        init_mua = gt_mua * (1 + np.random.randn() * 0.2)
        init_musp = gt_musp * (1 + np.random.randn() * 0.2)

        r = run_joint_inversion(
            gt_pos,
            gt_mua,
            gt_musp,
            init_pos,
            init_mua,
            init_musp,
            vertices,
            phi_mcx,
            is_direct,
        )
        r["seed"] = seed
        results.append(r)
        print(
            f"Seed {seed}: pos_err={r['pos_err_mm']:.3f}mm, mua_err={r['mua_err_pct']:.1f}%, musp_err={r['musp_err_pct']:.1f}%"
        )

    with open(output_dir / "m5_prime_joint.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["seed", "pos_err_mm", "mua_err_pct", "musp_err_pct"]
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

    with open(output_dir / "m5_prime_joint.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
