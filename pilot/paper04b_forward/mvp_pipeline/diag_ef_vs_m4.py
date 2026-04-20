"""Diagnostic: Compare E-F and M4' surface-space inversion on P5-ventral.

Goal: Understand why E-F gets 0.41mm but M4' gets ~2mm.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.direct_path import is_direct_path_vertex

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


def loss_ef_style(params, vertices, phi_mcx, direct_mask, optical):
    """E-F style loss (single source)."""
    source_pos = params[:3]
    r = np.linalg.norm(vertices - source_pos, axis=1)
    forward = G_inf(np.maximum(r, 0.01), optical).astype(np.float32)

    valid = direct_mask & (phi_mcx > 0) & (forward > 0)
    if np.sum(valid) < 10:
        return 1e10

    scale = float(np.sum(phi_mcx[valid]) / np.sum(forward[valid]))
    log_meas = np.log10(phi_mcx[valid] + 1e-20)
    log_fwd = np.log10(scale * forward[valid] + 1e-20)

    return float(np.mean((log_meas - log_fwd) ** 2))


def main():
    print("=" * 70)
    print("Diagnostic: E-F vs M4' surface-space on P5-ventral")
    print("=" * 70)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    print(f"Total vertices: {n_vertices}")

    gt_pos = np.array([-0.6, 2.4, -3.8])
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")
    phi_mcx = sample_fluence_at_vertices(fluence, vertices, VOXEL_SIZE_MM)

    print(f"GT position: {gt_pos}")
    print(f"phi_mcx range: [{phi_mcx.min():.2e}, {phi_mcx.max():.2e}]")

    is_direct_geo = np.array(
        [is_direct_path_vertex(gt_pos, v, volume, VOXEL_SIZE_MM) for v in vertices]
    )
    n_direct = int(np.sum(is_direct_geo))
    print(f"Direct vertices (geometric): {n_direct}")

    valid_phi = is_direct_geo & (phi_mcx > 0)
    n_valid = int(np.sum(valid_phi))
    print(f"Direct vertices with phi > 0: {n_valid}")

    forward_gt = G_inf(np.linalg.norm(vertices - gt_pos, axis=1), OPTICAL).astype(
        np.float32
    )
    valid_fwd = is_direct_geo & (forward_gt > 0)
    print(f"Direct vertices with forward > 0: {int(np.sum(valid_fwd))}")

    direct_mask = is_direct_geo & (phi_mcx > 0) & (forward_gt > 0)
    n_mask = int(np.sum(direct_mask))
    print(f"Final mask (direct & phi>0 & fwd>0): {n_mask}")

    if n_mask < 100:
        print("WARNING: Very few vertices in mask!")
        return

    print("\nRunning optimization with same mask as E-F...")

    np.random.seed(42)
    init_pos = gt_pos + np.random.randn(3) * 0.5
    print(f"Init position: {init_pos}")

    result = minimize(
        loss_ef_style,
        init_pos.copy(),
        args=(vertices, phi_mcx, direct_mask, OPTICAL),
        method="L-BFGS-B",
        options={"maxiter": 200},
    )

    recovered = result.x[:3]
    pos_err = float(np.linalg.norm(recovered - gt_pos))

    print(f"\nResult:")
    print(f"  Recovered: {recovered}")
    print(f"  GT:        {gt_pos}")
    print(f"  Error:     {pos_err:.3f} mm")
    print(f"  Success:   {result.success}")
    print(f"  Message:   {result.message}")

    print("\nChecking NCC at GT position...")
    forward_gt_valid = forward_gt[direct_mask]
    phi_valid = phi_mcx[direct_mask]
    scale_gt = float(np.sum(phi_valid) / np.sum(forward_gt_valid))
    log_meas = np.log10(phi_valid + 1e-20)
    log_fwd = np.log10(scale_gt * forward_gt_valid + 1e-20)
    mse_gt = float(np.mean((log_meas - log_fwd) ** 2))
    print(f"  Scale at GT: {scale_gt:.2e}")
    print(f"  MSE at GT:   {mse_gt:.6f}")

    print("\nChecking NCC at init position...")
    forward_init = G_inf(np.linalg.norm(vertices - init_pos, axis=1), OPTICAL).astype(
        np.float32
    )
    forward_init_valid = forward_init[direct_mask]
    scale_init = float(np.sum(phi_valid) / np.sum(forward_init_valid))
    log_fwd_init = np.log10(scale_init * forward_init_valid + 1e-20)
    mse_init = float(np.mean((log_meas - log_fwd_init) ** 2))
    print(f"  Scale at init: {scale_init:.2e}")
    print(f"  MSE at init:   {mse_init:.6f}")

    print("\nChecking NCC at recovered position...")
    forward_rec = G_inf(np.linalg.norm(vertices - recovered, axis=1), OPTICAL).astype(
        np.float32
    )
    forward_rec_valid = forward_rec[direct_mask]
    scale_rec = float(np.sum(phi_valid) / np.sum(forward_rec_valid))
    log_fwd_rec = np.log10(scale_rec * forward_rec_valid + 1e-20)
    mse_rec = float(np.mean((log_meas - log_fwd_rec) ** 2))
    print(f"  Scale at recovered: {scale_rec:.2e}")
    print(f"  MSE at recovered:   {mse_rec:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  This should match E-F result (0.41mm for n_sources=1)")
    print(f"  If error is ~2mm, there's a bug to find.")
    print("=" * 70)


if __name__ == "__main__":
    main()
