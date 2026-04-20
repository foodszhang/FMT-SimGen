"""M6': Figure 1 teaser data + §4.H two-regime table."""

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
    output_dir = Path("pilot/paper04b_forward/results/m6_teaser")
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)
    n_vertices = len(vertices)
    center = np.array([95, 100, 52]) / 2

    # §4.H: Two-regime table
    # Regime A: P1-dorsal (dorsal source, dorsal view = direct)
    # Regime B: P5-ventral (ventral source, ventral view = direct)
    # Regime C: P5-ventral 0° (through-organ) = OUT OF SCOPE

    regimes = []

    # Regime A: P1-dorsal at 0° (but actually this is not direct-path for vertex)
    source_p1 = np.array([-0.6, 2.4, 5.8])
    fluence_p1 = np.load(ARCHIVE_BASE / "S2-Vol-P1-dorsal-r2.0" / "fluence.npy")

    phi_mcx_p1 = np.zeros(n_vertices, dtype=np.float32)
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx_p1[i] = fluence_p1[vx, vy, vz]

    dx = vertices[:, 0] - source_p1[0]
    dy = vertices[:, 1] - source_p1[1]
    dz = vertices[:, 2] - source_p1[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    phi_closed_p1 = G_inf(np.maximum(r, 0.01), OPTICAL).astype(np.float32)

    is_direct_p1 = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        is_direct_p1[i] = is_direct_path_vertex(
            source_p1, vertices[i], volume, VOXEL_SIZE_MM
        )

    ncc_direct_p1 = compute_ncc(phi_mcx_p1, phi_closed_p1, is_direct_p1)
    regimes.append(
        {
            "regime": "A (P1-dorsal, direct-path)",
            "source": "P1-dorsal",
            "view": "N/A (vertex-level)",
            "n_direct": int(np.sum(is_direct_p1)),
            "ncc_direct": ncc_direct_p1,
        }
    )

    # Regime B: P5-ventral
    source_p5 = np.array([-0.6, 2.4, -3.8])
    fluence_p5 = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    phi_mcx_p5 = np.zeros(n_vertices, dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx_p5[i] = fluence_p5[vx, vy, vz]

    dx = vertices[:, 0] - source_p5[0]
    dy = vertices[:, 1] - source_p5[1]
    dz = vertices[:, 2] - source_p5[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    phi_closed_p5 = G_inf(np.maximum(r, 0.01), OPTICAL).astype(np.float32)

    is_direct_p5 = np.zeros(n_vertices, dtype=bool)
    for i in range(n_vertices):
        is_direct_p5[i] = is_direct_path_vertex(
            source_p5, vertices[i], volume, VOXEL_SIZE_MM
        )

    ncc_direct_p5 = compute_ncc(phi_mcx_p5, phi_closed_p5, is_direct_p5)
    regimes.append(
        {
            "regime": "B (P5-ventral, direct-path)",
            "source": "P5-ventral",
            "view": "N/A (vertex-level)",
            "n_direct": int(np.sum(is_direct_p5)),
            "ncc_direct": ncc_direct_p5,
        }
    )

    # Regime C: Out of scope
    regimes.append(
        {
            "regime": "C (through-organ)",
            "source": "N/A",
            "view": "N/A",
            "n_direct": 0,
            "ncc_direct": "OUT OF SCOPE",
        }
    )

    print("§4.H Two-Regime Table:")
    for r in regimes:
        ncc_str = (
            f"{r['ncc_direct']:.4f}"
            if isinstance(r["ncc_direct"], float)
            else r["ncc_direct"]
        )
        print(f"  {r['regime']}: N_direct={r['n_direct']}, NCC={ncc_str}")

    with open(output_dir / "h_two_regime.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["regime", "source", "view", "n_direct", "ncc_direct"]
        )
        w.writeheader()
        for r in regimes:
            w.writerow(r)

    # Figure 1 teaser summary
    fig1_data = {
        "panel1": "Direct-path measurement subset",
        "panel2": "Inversion trajectory (see m4_prime_multiview)",
        "panel3": "Initial vs Final vs GT",
        "ncc_baseline": ncc_direct_p5,
        "position_error_mm": 0.414,  # From E-F N=1
    }

    with open(output_dir / "m6_teaser.json", "w") as f:
        json.dump(fig1_data, f, indent=2)

    print(f"\nM6' Figure 1: NCC_baseline={ncc_direct_p5:.4f}, pos_err_min=0.414mm")


if __name__ == "__main__":
    main()
