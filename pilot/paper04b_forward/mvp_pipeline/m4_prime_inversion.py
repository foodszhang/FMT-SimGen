"""M4': N=1 inversion with direct-path constraint.

Fixed version: Use gradient-based optimization with proper scaling.
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

SOFT_TISSUE_LABEL = 1
AIR_LABEL = 0


def load_volume() -> np.ndarray:
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    return volume.reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(
    binary_mask: np.ndarray, voxel_size_mm: float
) -> np.ndarray:
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def is_direct_path_vertex(
    source_pos_mm: np.ndarray,
    vertex_pos_mm: np.ndarray,
    volume_labels: np.ndarray,
    voxel_size_mm: float,
) -> bool:
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

        label = volume_labels[voxel[0], voxel[1], voxel[2]]
        if label not in {AIR_LABEL, SOFT_TISSUE_LABEL}:
            return False

    return True


def sample_fluence_at_vertices(
    fluence: np.ndarray, vertices_mm: np.ndarray, voxel_size_mm: float
) -> np.ndarray:
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


def compute_forward_and_grad(
    source_pos: np.ndarray, vertices: np.ndarray, optical
) -> tuple:
    """Compute forward fluence and gradient at vertices.

    G(r) = exp(-r/delta) / (4*pi*D*r)
    dG/dx_i = G(r) * (-1/delta - 1/r) * (x_i - s_i) / r
    """
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


def loss_and_grad(
    params: np.ndarray,
    vertices: np.ndarray,
    measurement: np.ndarray,
    direct_mask: np.ndarray,
    scale: float,
    optical,
) -> tuple:
    """Compute loss and gradient."""
    source_pos = params[:3]

    forward, grad_forward = compute_forward_and_grad(source_pos, vertices, optical)

    valid = direct_mask & (measurement > 1e-10) & (forward > 1e-10)
    n_valid = np.sum(valid)

    if n_valid < 10:
        return 1e10, np.zeros(3)

    log_meas = np.log10(measurement[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    diff = log_meas - log_fwd
    loss = float(np.mean(diff**2))

    d_loss_d_logfwd = -2.0 * diff / n_valid
    d_logfwd_d_fwd = 1.0 / (scale * forward[valid] * np.log(10))
    d_fwd_d_pos = grad_forward[valid]

    grad = np.zeros(3)
    for i in range(3):
        grad[i] = np.sum(d_loss_d_logfwd * d_logfwd_d_fwd * scale * d_fwd_d_pos[:, i])

    return loss, grad


def run_inversion(
    gt_pos: np.ndarray,
    init_pos: np.ndarray,
    vertices: np.ndarray,
    measurement: np.ndarray,
    direct_mask: np.ndarray,
    scale: float,
) -> dict:
    """Run inversion from init_pos to recover gt_pos."""

    result = minimize(
        loss_and_grad,
        init_pos.copy(),
        args=(vertices, measurement, direct_mask, scale, OPTICAL),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},
    )

    recovered_pos = result.x[:3]
    position_error = np.linalg.norm(recovered_pos - gt_pos)

    forward, _ = compute_forward_and_grad(recovered_pos, vertices, OPTICAL)
    valid = direct_mask & (measurement > 1e-10) & (forward > 1e-10)
    if np.sum(valid) > 10:
        log_meas = np.log10(measurement[valid] + 1e-20)
        log_fwd = np.log10(scale * forward[valid] + 1e-20)
        final_ncc = np.corrcoef(log_meas, log_fwd)[0, 1]
    else:
        final_ncc = 0.0

    return {
        "gt_position_mm": gt_pos.tolist(),
        "init_position_mm": init_pos.tolist(),
        "recovered_position_mm": recovered_pos.tolist(),
        "position_error_mm": float(position_error),
        "final_ncc": float(final_ncc),
        "n_iterations": int(result.nit),
        "success": result.success,
        "final_loss": float(result.fun),
    }


def main():
    parser = argparse.ArgumentParser(description="M4': N=1 inversion")
    parser.add_argument(
        "--output", type=str, default="pilot/paper04b_forward/results/m4_prime"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    binary_mask = volume > 0

    logger.info("Extracting surface vertices...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    logger.info(f"Total vertices: {n_vertices}")

    gt_positions = {
        "P1-dorsal": np.array([-1.6, 2.4, 5.8]),
        "P5-ventral": np.array([-0.6, 2.4, -3.8]),
    }

    results = {
        "voxel_size_mm": VOXEL_SIZE_MM,
        "n_vertices_total": n_vertices,
        "seed": args.seed,
        "inversions": {},
    }

    for pos_name, gt_pos in gt_positions.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Position: {pos_name}")
        logger.info(f"GT position: {gt_pos} mm")
        logger.info(f"{'=' * 60}")

        logger.info("Computing direct-path vertices...")
        is_direct = np.zeros(n_vertices, dtype=bool)
        for i in range(0, n_vertices, 10000):
            end_i = min(i + 10000, n_vertices)
            for j in range(i, end_i):
                is_direct[j] = is_direct_path_vertex(
                    gt_pos, vertices_mm[j], volume, VOXEL_SIZE_MM
                )

        n_direct = int(np.sum(is_direct))
        logger.info(f"Direct-path vertices: {n_direct}")

        if pos_name == "P1-dorsal":
            fluence_path = ARCHIVE_BASE / "S2-Vol-P1-dorsal-r2.0" / "fluence.npy"
        else:
            fluence_path = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"

        logger.info(f"Loading measurement from {fluence_path}")
        fluence = np.load(fluence_path)
        measurement = sample_fluence_at_vertices(fluence, vertices_mm, VOXEL_SIZE_MM)

        valid = is_direct & (measurement > 0)
        forward_gt, _ = compute_forward_and_grad(gt_pos, vertices_mm, OPTICAL)
        scale = float(np.sum(measurement[valid]) / np.sum(forward_gt[valid]))
        logger.info(f"Scale factor: {scale:.2e}")

        init_errors = [0.5, 1.0, 2.0]
        pos_results = []

        for init_error in init_errors:
            init_pos = gt_pos + np.random.randn(3) * init_error
            init_err_actual = np.linalg.norm(init_pos - gt_pos)
            logger.info(
                f"  Init error target: {init_error}mm, actual: {init_err_actual:.2f}mm"
            )

            inv_result = run_inversion(
                gt_pos, init_pos, vertices_mm, measurement, is_direct, scale
            )
            pos_results.append(inv_result)

            logger.info(f"    Recovered: {inv_result['recovered_position_mm']}")
            logger.info(f"    Error: {inv_result['position_error_mm']:.3f}mm")
            logger.info(f"    Final NCC: {inv_result['final_ncc']:.4f}")
            logger.info(f"    Iterations: {inv_result['n_iterations']}")

        results["inversions"][pos_name] = {
            "gt_position_mm": gt_pos.tolist(),
            "n_direct_vertices": n_direct,
            "scale_factor": scale,
            "trials": pos_results,
        }

    with open(output_dir / "m4_prime_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("M4': N=1 Inversion Results")
    print("=" * 75)
    print(
        f"{'Position':<15} {'Init Err':<12} {'Pos Error':<12} {'Final NCC':<12} {'Pass'}"
    )
    print("-" * 75)

    for pos_name, data in results["inversions"].items():
        for i, trial in enumerate(data["trials"]):
            init_err = init_errors[i]
            pos_err = trial["position_error_mm"]
            ncc = trial["final_ncc"]
            passed = "✓" if pos_err < 1.0 else "✗"
            print(
                f"{pos_name:<15} {init_err:<12.1f} {pos_err:<12.3f} {ncc:<12.4f} {passed}"
            )

    print("=" * 75)

    all_pass = all(
        trial["position_error_mm"] < 1.0
        for data in results["inversions"].values()
        for trial in data["trials"]
    )
    if all_pass:
        print("All inversions pass (position error < 1mm) ✓")
    else:
        print("Some inversions fail - check results")


if __name__ == "__main__":
    main()
