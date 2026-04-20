"""E-D: Multi-view equivariance."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.direct_path import is_direct_path

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


def compute_ncc(phi_mcx, phi_closed, mask):
    valid = mask & (phi_mcx > 0) & (phi_closed > 0)
    if np.sum(valid) < 10:
        return 0.0
    log_mcx = np.log10(phi_mcx[valid] + 1e-20)
    log_closed = np.log10(phi_closed[valid] + 1e-20)
    return float(np.corrcoef(log_mcx, log_closed)[0, 1])


def main():
    output_dir = Path("pilot/paper04b_forward/results/ed_equivariance")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    center = np.array([95, 100, 52]) / 2

    source_pos = np.array([-0.6, 2.4, -3.8])  # P5-ventral
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    phi_mcx = np.zeros(n_vertices, dtype=np.float32)
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx[i] = fluence[vx, vy, vz]

    dx = vertices[:, 0] - source_pos[0]
    dy = vertices[:, 1] - source_pos[1]
    dz = vertices[:, 2] - source_pos[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    phi_closed = G_inf(np.maximum(r, 0.01), OPTICAL).astype(np.float32)

    is_direct = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        is_direct[i] = is_direct_path_vertex(
            source_pos, vertices[i], volume, VOXEL_SIZE_MM
        )

    results = []
    ncc_values = []

    for angle in [0, 30, 60, 90, -30, -60, -90]:
        view_result = is_direct_path(source_pos, angle, volume, VOXEL_SIZE_MM)
        is_direct_view = view_result.is_direct

        ncc_direct = compute_ncc(phi_mcx, phi_closed, is_direct)

        results.append(
            {
                "angle_deg": angle,
                "is_direct_view": is_direct_view,
                "ncc_direct": ncc_direct,
            }
        )
        ncc_values.append(ncc_direct)
        print(f"Angle {angle:>4}°: direct_view={is_direct_view}, NCC={ncc_direct:.4f}")

    spread = max(ncc_values) - min(ncc_values)
    mean_ncc = np.mean(ncc_values)
    print(f"\nSpread: {spread:.4f}, Mean: {mean_ncc:.4f}")

    with open(output_dir / "ed_equivariance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["angle_deg", "is_direct_view", "ncc_direct"])
        w.writeheader()
        for r in results:
            w.writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            )


if __name__ == "__main__":
    main()
