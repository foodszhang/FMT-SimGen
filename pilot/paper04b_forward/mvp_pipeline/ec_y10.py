"""E-C: P2/P3/P4 at Y=10 with direct-path filtering."""

from __future__ import annotations

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
    output_dir = Path("pilot/paper04b_forward/results/ec_y10")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    center = np.array([95, 100, 52]) / 2

    positions = {
        "P1-dorsal": {
            "pos": np.array([-0.6, 2.4, 5.8]),
            "archive": "S2-Vol-P1-dorsal-r2.0",
        },
        "P2-left": {
            "pos": np.array([-8.0, 2.4, 1.0]),
            "archive": "S2-Vol-P2-left-r2.0",
        },
        "P3-right": {
            "pos": np.array([6.8, 2.4, 1.0]),
            "archive": "S2-Vol-P3-right-r2.0",
        },
        "P4-dorsal-lat": {
            "pos": np.array([-6.3, 2.4, 5.8]),
            "archive": "S2-Vol-P4-dorsal-lat-r2.0",
        },
        "P5-ventral": {
            "pos": np.array([-0.6, 2.4, -3.8]),
            "archive": "S2-Vol-P5-ventral-r2.0",
        },
    }

    results = []

    for pos_name, info in positions.items():
        source_pos = info["pos"]
        fluence_path = ARCHIVE_BASE / info["archive"] / "fluence.npy"

        if not fluence_path.exists():
            print(f"{pos_name}: fluence not found, skipping")
            continue

        fluence = np.load(fluence_path)

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

        ncc_all = compute_ncc(phi_mcx, phi_closed, np.ones(n_vertices, dtype=bool))
        ncc_direct = compute_ncc(phi_mcx, phi_closed, is_direct)
        n_direct = int(np.sum(is_direct))

        results.append(
            {
                "position": pos_name,
                "pos_mm": source_pos.tolist(),
                "n_direct_vertices": n_direct,
                "ncc_all": ncc_all,
                "ncc_direct": ncc_direct,
            }
        )
        print(
            f"{pos_name}: N_direct={n_direct}, NCC_all={ncc_all:.4f}, NCC_direct={ncc_direct:.4f}"
        )

    with open(output_dir / "ec_y10.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["position", "n_direct_vertices", "ncc_all", "ncc_direct"]
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k in w.fieldnames
                }
            )

    with open(output_dir / "ec_y10.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
