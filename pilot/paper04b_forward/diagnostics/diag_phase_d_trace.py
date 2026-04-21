"""Phase D: Optimizer trace for M4' Phase C vs E-F N=1."""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.metrics import scale_factor_logmse

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


def run_m4prime_trace(gt_pos, init_pos, vertices, measurement, direct_mask, output_csv):
    trace = []

    def loss_only(params):
        source_pos = params[:3]
        forward, _ = compute_forward_and_grad(source_pos, vertices, OPTICAL)
        valid = direct_mask & (measurement > 1e-10) & (forward > 1e-10)
        n_valid = np.sum(valid)
        if n_valid < 10:
            return 1e10
        scale = scale_factor_logmse(measurement[valid], forward[valid])
        log_meas = np.log10(measurement[valid] + 1e-20)
        log_fwd = np.log10(scale * forward[valid] + 1e-20)
        diff = log_meas - log_fwd
        loss = float(np.mean(diff**2))
        return loss

    def callback(xk):
        loss = loss_only(xk)
        err = np.linalg.norm(xk - gt_pos)
        trace.append(
            {
                "iter": len(trace),
                "loss": loss,
                "pos_x": xk[0],
                "pos_y": xk[1],
                "pos_z": xk[2],
                "err_to_gt": err,
            }
        )

    result = minimize(
        loss_only,
        init_pos.copy(),
        method="L-BFGS-B",
        jac=False,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
        callback=callback,
    )

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["iter", "loss", "pos_x", "pos_y", "pos_z", "err_to_gt"]
        )
        w.writeheader()
        for row in trace:
            w.writerow(
                {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()}
            )

    return result, trace


def forward_single_source(pos, vertices, optical):
    dx = vertices[:, 0] - pos[0]
    dy = vertices[:, 1] - pos[1]
    dz = vertices[:, 2] - pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    return G_inf(r, optical).astype(np.float32)


def run_ef_trace(gt_pos, init_pos, vertices, measurement, direct_mask, output_csv):
    trace = []

    def loss_single_source(params):
        pos = params[:3]
        forward = forward_single_source(pos, vertices, OPTICAL)
        valid = direct_mask & (measurement > 0) & (forward > 0)
        if np.sum(valid) < 10:
            return 1e10
        scale = scale_factor_logmse(measurement[valid], forward[valid])
        log_meas = np.log10(measurement[valid] + 1e-20)
        log_fwd = np.log10(scale * forward[valid] + 1e-20)
        return float(np.mean((log_meas - log_fwd) ** 2))

    def callback(xk):
        loss = loss_single_source(xk)
        err = np.linalg.norm(xk - gt_pos)
        trace.append(
            {
                "iter": len(trace),
                "loss": loss,
                "pos_x": xk[0],
                "pos_y": xk[1],
                "pos_z": xk[2],
                "err_to_gt": err,
            }
        )

    result = minimize(
        loss_single_source,
        init_pos.copy(),
        method="L-BFGS-B",
        options={"maxiter": 200},
        callback=callback,
    )

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["iter", "loss", "pos_x", "pos_y", "pos_z", "err_to_gt"]
        )
        w.writeheader()
        for row in trace:
            w.writerow(
                {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()}
            )

    return result, trace


def main():
    print("=" * 70)
    print("Phase D: Optimizer trace for M4' Phase C vs E-F N=1")
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

    print("Computing direct-path vertices...")
    is_direct = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        is_direct[i] = is_direct_path_vertex(
            gt_pos, vertices_mm[i], volume, VOXEL_SIZE_MM
        )
    n_direct = int(np.sum(is_direct))
    print(f"Direct-path vertices: {n_direct}")

    np.random.seed(42)
    init_pos_m4prime = gt_pos + np.random.randn(3) * 0.5
    print(f"M4' init_pos: {init_pos_m4prime}")

    np.random.seed(42)
    init_pos_ef = gt_pos + np.random.randn(3) * 0.5
    print(f"E-F init_pos: {init_pos_ef}")

    output_dir = Path("pilot/paper04b_forward/results/ef_vs_m4_diag")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning M4' Phase C...")
    result_m4, trace_m4 = run_m4prime_trace(
        gt_pos,
        init_pos_m4prime,
        vertices_mm,
        measurement,
        is_direct,
        output_dir / "m4prime_trace.csv",
    )

    print("\nRunning E-F N=1...")
    result_ef, trace_ef = run_ef_trace(
        gt_pos,
        init_pos_ef,
        vertices_mm,
        measurement,
        is_direct,
        output_dir / "ef_trace.csv",
    )

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    print(f"\nM4' Phase C:")
    print(f"  Total iterations: {result_m4.nit}")
    print(f"  Converged: {result_m4.success}")
    print(f"  Message: {result_m4.message}")
    print(f"  Final loss: {result_m4.fun:.6f}")
    print(f"  Final |pos - gt|: {np.linalg.norm(result_m4.x - gt_pos):.3f} mm")

    first_good_m4 = next(
        (i for i, t in enumerate(trace_m4) if t["err_to_gt"] < 1.0), None
    )
    print(f"  First |pos - gt| < 1mm at iter: {first_good_m4}")

    print(f"\nE-F N=1:")
    print(f"  Total iterations: {result_ef.nit}")
    print(f"  Converged: {result_ef.success}")
    print(f"  Message: {result_ef.message}")
    print(f"  Final loss: {result_ef.fun:.6f}")
    print(f"  Final |pos - gt|: {np.linalg.norm(result_ef.x - gt_pos):.3f} mm")

    first_good_ef = next(
        (i for i, t in enumerate(trace_ef) if t["err_to_gt"] < 1.0), None
    )
    print(f"  First |pos - gt| < 1mm at iter: {first_good_ef}")

    print(f"\nTrace files:")
    print(f"  {output_dir / 'm4prime_trace.csv'}")
    print(f"  {output_dir / 'ef_trace.csv'}")


if __name__ == "__main__":
    main()
