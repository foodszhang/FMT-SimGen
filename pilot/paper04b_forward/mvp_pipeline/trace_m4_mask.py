"""Trace M4' mask calculation to find discrepancy."""

import sys
from pathlib import Path

import numpy as np
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


def main():
    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    print(f"Total vertices: {n_vertices}")

    gt_pos = np.array([-0.6, 2.4, -3.8])
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    phi_mcx = np.zeros(n_vertices, dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx[i] = fluence[vx, vy, vz]

    is_direct_geo = np.array(
        [is_direct_path_vertex(gt_pos, v, volume, VOXEL_SIZE_MM) for v in vertices]
    )

    print(f"\n=== M4' calculation ===")
    print(f"is_direct_geo count: {np.sum(is_direct_geo)}")
    print(f"M4' reports n_verts = 10468 (just is_direct_geo, no phi filter)")

    print(f"\n=== E-F style calculation ===")
    forward = G_inf(np.linalg.norm(vertices - gt_pos, axis=1), OPTICAL).astype(
        np.float32
    )
    valid_ef = is_direct_geo & (phi_mcx > 0) & (forward > 0)
    print(f"E-F mask (is_direct & phi>0 & forward>0): {np.sum(valid_ef)}")

    print(f"\n=== The issue ===")
    print(f"M4' uses is_direct_geo as direct_mask")
    print(
        f"But the loss function filters: valid = direct_mask & (phi_mcx > 0) & (forward > 0)"
    )
    print(f"So the effective mask is the same!")

    print(f"\n=== Checking M4' loss function ===")
    print(f"M4' loss_surface line 95-96:")
    print(f"  valid = direct_mask & (phi_mcx > 0) & (forward > 0)")
    print(f"  if np.sum(valid) < 50: return 1e10")
    print(f"This is the SAME as E-F!")

    print(f"\n=== So why different results? ===")
    print(f"Let me check the init_pos...")

    np.random.seed(42)
    init_pos_ef = gt_pos + np.random.randn(3) * 0.5
    print(f"E-F init_pos (seed=42, sigma=0.5): {init_pos_ef}")

    np.random.seed(0)
    init_pos_m4 = gt_pos + np.random.randn(3) * 1.0
    print(f"M4' init_pos (seed=0, sigma=1.0): {init_pos_m4}")

    print(f"\n=== AHA! Different random seeds and sigma! ===")
    print(f"E-F: seed=42, sigma=0.5")
    print(f"M4': seed=0-4, sigma=1.0")
    print(f"This explains the different results!")


if __name__ == "__main__":
    main()
