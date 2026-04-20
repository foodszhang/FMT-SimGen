"""Verify M4' with E-F style init (seed=42, sigma=0.5)."""

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


def loss_surface(params, vertices, phi_mcx, direct_mask, optical):
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


def main():
    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)

    positions = {
        "P1-dorsal": np.array([-1.6, 2.4, 5.8]),
        "P5-ventral": np.array([-0.6, 2.4, -3.8]),
        "P2-left": np.array([-8.0, 1.9, 1.0]),
    }

    print("=" * 70)
    print("M4' with E-F style init (seed=42, sigma=0.5)")
    print("=" * 70)

    for pos_name, gt_pos in positions.items():
        fluence = np.load(ARCHIVE_BASE / f"S2-Vol-{pos_name}-r2.0" / "fluence.npy")
        phi_mcx = sample_fluence_at_vertices(fluence, vertices, VOXEL_SIZE_MM)

        is_direct_geo = np.array(
            [is_direct_path_vertex(gt_pos, v, volume, VOXEL_SIZE_MM) for v in vertices]
        )

        np.random.seed(42)
        init_pos = gt_pos + np.random.randn(3) * 0.5

        result = minimize(
            loss_surface,
            init_pos.copy(),
            args=(vertices, phi_mcx, is_direct_geo, OPTICAL),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        recovered = result.x[:3]
        pos_err = float(np.linalg.norm(recovered - gt_pos))

        print(
            f"{pos_name}: {pos_err:.3f} mm (init_err={np.linalg.norm(init_pos - gt_pos):.3f})"
        )

    print("\n" + "=" * 70)
    print("Now with sigma=1.0 (M4' original):")
    print("=" * 70)

    for pos_name, gt_pos in positions.items():
        fluence = np.load(ARCHIVE_BASE / f"S2-Vol-{pos_name}-r2.0" / "fluence.npy")
        phi_mcx = sample_fluence_at_vertices(fluence, vertices, VOXEL_SIZE_MM)

        is_direct_geo = np.array(
            [is_direct_path_vertex(gt_pos, v, volume, VOXEL_SIZE_MM) for v in vertices]
        )

        errors = []
        for seed in range(5):
            np.random.seed(seed)
            init_pos = gt_pos + np.random.randn(3) * 1.0

            result = minimize(
                loss_surface,
                init_pos.copy(),
                args=(vertices, phi_mcx, is_direct_geo, OPTICAL),
                method="L-BFGS-B",
                options={"maxiter": 200},
            )

            recovered = result.x[:3]
            pos_err = float(np.linalg.norm(recovered - gt_pos))
            errors.append(pos_err)

        print(f"{pos_name}: {np.mean(errors):.2f} ± {np.std(errors):.2f} mm")


if __name__ == "__main__":
    main()
