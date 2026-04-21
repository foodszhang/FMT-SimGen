"""Phase A-2: Gradient check for m4_prime loss_and_grad."""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def sample_fluence_at_vertices(fluence, vertices_mm, voxel_size_mm):
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices_mm / voxel_size_mm + center).astype(int)
    phi = np.zeros(len(vertices_mm), dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi[i] = fluence[vx, vy, vz]
    return phi


def compute_forward_and_grad(source_pos, vertices, optical):
    from shared.green import G_inf

    dx = vertices[:, 0] - source_pos[0]
    dy = vertices[:, 1] - source_pos[1]
    dz = vertices[:, 2] - source_pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)

    G = G_inf(r, optical).astype(np.float64)

    factor = (-1.0 / optical.delta - 1.0 / r) / r
    dG_dx = G * factor * (-dx)
    dG_dy = G * factor * (-dy)
    dG_dz = G * factor * (-dz)

    return G, np.column_stack([dG_dx, dG_dy, dG_dz])


def loss_fn(params, vertices, measurement, direct_mask, scale, optical):
    source_pos = params[:3]
    forward, _ = compute_forward_and_grad(source_pos, vertices, optical)

    valid = direct_mask & (measurement > 1e-10) & (forward > 1e-10)
    n_valid = np.sum(valid)

    if n_valid < 10:
        return 1e10

    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    diff = log_meas - log_fwd
    loss = float(np.mean(diff**2))

    return loss


def grad_fn(params, vertices, measurement, direct_mask, scale, optical):
    source_pos = params[:3]
    forward, grad_forward = compute_forward_and_grad(source_pos, vertices, optical)

    valid = direct_mask & (measurement > 1e-10) & (forward > 1e-10)
    n_valid = np.sum(valid)

    if n_valid < 10:
        return np.zeros(3)

    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    diff = log_meas - log_fwd

    d_loss_d_logfwd = -2.0 * diff / n_valid
    d_logfwd_d_fwd = 1.0 / (scale * forward[valid] * np.log(10))
    d_fwd_d_pos = grad_forward[valid]

    grad = np.zeros(3)
    for i in range(3):
        grad[i] = np.sum(d_loss_d_logfwd * d_logfwd_d_fwd * scale * d_fwd_d_pos[:, i])

    return grad


def numeric_grad(params, vertices, measurement, direct_mask, scale, optical, eps=1e-5):
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        f_plus = loss_fn(
            params_plus, vertices, measurement, direct_mask, scale, optical
        )
        f_minus = loss_fn(
            params_minus, vertices, measurement, direct_mask, scale, optical
        )
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def main():
    print("=" * 70)
    print("Phase A-2: m4_prime gradient check")
    print("=" * 70)

    volume = load_volume()
    binary_mask = volume > 0

    print("Extracting surface vertices...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    print(f"Total vertices: {n_vertices}")

    gt_pos = np.array([-0.6, 2.4, -3.8])

    fluence_path = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"
    print(f"Loading fluence from {fluence_path}")
    fluence = np.load(fluence_path)
    measurement = sample_fluence_at_vertices(fluence, vertices_mm, VOXEL_SIZE_MM)

    valid_mask = measurement > 0
    forward_gt, _ = compute_forward_and_grad(gt_pos, vertices_mm, OPTICAL)
    scale = float(np.sum(measurement[valid_mask]) / np.sum(forward_gt[valid_mask]))
    print(f"Scale factor (from GT): {scale:.2e}")

    np.random.seed(0)

    body_center = np.array([0.0, 0.0, 0.0])
    body_radius = 15.0

    random_positions = []
    for _ in range(10):
        while True:
            pos = body_center + (np.random.rand(3) - 0.5) * 2 * body_radius
            if np.linalg.norm(pos - body_center) < body_radius:
                random_positions.append(pos)
                break

    results = []

    for i, x in enumerate(random_positions):
        x_params = x.astype(np.float64)

        l2_diff = check_grad(
            loss_fn,
            grad_fn,
            x_params,
            vertices_mm,
            measurement,
            valid_mask,
            scale,
            OPTICAL,
        )

        g_analytic = grad_fn(
            x_params, vertices_mm, measurement, valid_mask, scale, OPTICAL
        )
        g_numeric = numeric_grad(
            x_params, vertices_mm, measurement, valid_mask, scale, OPTICAL
        )

        rel_err = np.linalg.norm(g_analytic - g_numeric) / (
            np.linalg.norm(g_numeric) + 1e-12
        )
        cos_sim = np.dot(g_analytic, g_numeric) / (
            np.linalg.norm(g_analytic) * np.linalg.norm(g_numeric) + 1e-12
        )

        results.append(
            {
                "x": x.tolist(),
                "l2_diff": float(l2_diff),
                "rel_err": float(rel_err),
                "cos_sim": float(cos_sim),
                "grad_analytic_norm": float(np.linalg.norm(g_analytic)),
                "grad_numeric_norm": float(np.linalg.norm(g_numeric)),
            }
        )

        print(
            f"[{i}] rel_err={rel_err:.2e}, cos_sim={cos_sim:.4f}, "
            f"|g_a|={np.linalg.norm(g_analytic):.2e}, "
            f"|g_n|={np.linalg.norm(g_numeric):.2e}"
        )

    output_dir = Path("pilot/paper04b_forward/results/ef_vs_m4_diag")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "m4prime_grad_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nOutput: {output_dir / 'm4prime_grad_check.csv'}")

    max_rel_err = max(r["rel_err"] for r in results)
    min_cos_sim = min(r["cos_sim"] for r in results)
    n_bad = sum(1 for r in results if r["rel_err"] > 1e-3)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  max rel_err: {max_rel_err:.2e}")
    print(f"  min cos_sim: {min_cos_sim:.4f}")
    print(f"  rel_err > 1e-3: {n_bad} / 10")
    print("=" * 70)


if __name__ == "__main__":
    main()
