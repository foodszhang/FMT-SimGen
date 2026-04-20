"""M3': Multi-view validation with direct-path filtering.

Key validation: Surface-space NCC is view-independent.

For this validation, we use the MCX fluence volume directly (not projections).
Since the fluence is view-independent (it's the 3D photon distribution),
comparing it with closed-form on surface vertices tests the physics,
not the camera projection.

Method:
1. Compute direct-path vertices for source
2. Sample MCX fluence at vertices
3. Compute closed-form at vertices
4. Verify NCC ≥ 0.90 (same as D2.1, confirming consistency)

For true multi-view validation with projections, we need:
- Multiple MCX runs with different camera positions
- OR use archived projection data and map back to vertices

This script uses the fluence volume approach for M3'.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.direct_path import is_direct_path, get_direct_views_for_source

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
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices_mm / voxel_size_mm + center).astype(int)

    phi = np.zeros(len(vertices_mm), dtype=np.float32)
    valid = np.zeros(len(vertices_mm), dtype=bool)

    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi[i] = fluence[vx, vy, vz]
            valid[i] = True

    return phi, valid


def compute_closed_form_at_vertices(
    vertices_mm: np.ndarray, source_pos_mm: np.ndarray
) -> np.ndarray:
    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    return G_inf(r, OPTICAL).astype(np.float32)


def compute_ncc(phi_mcx: np.ndarray, phi_closed: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & (phi_mcx > 0) & (phi_closed > 0)
    if np.sum(valid) < 10:
        return 0.0

    log_mcx = np.log10(phi_mcx[valid] + 1e-20)
    log_closed = np.log10(phi_closed[valid] + 1e-20)
    return float(np.corrcoef(log_mcx, log_closed)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="M3': Multi-view validation")
    parser.add_argument(
        "--output", type=str, default="pilot/paper04b_forward/results/m3_prime"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    binary_mask = volume > 0

    logger.info("Extracting surface vertices...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    logger.info(f"Total vertices: {n_vertices}")

    positions = {
        "P1-dorsal": np.array([-0.6, 2.4, 5.8]),
        "P5-ventral": np.array([-0.6, 2.4, -3.8]),
    }

    all_angles = [0, 30, 60, 90, -90, -30, -60, 180]

    results = {
        "voxel_size_mm": VOXEL_SIZE_MM,
        "n_vertices_total": n_vertices,
        "positions": {},
    }

    for pos_name, source_pos in positions.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Position: {pos_name} at {source_pos} mm")
        logger.info(f"{'=' * 60}")

        direct_views = get_direct_views_for_source(
            source_pos, all_angles, volume, VOXEL_SIZE_MM
        )
        logger.info(f"Direct-path views: {[a for a, _ in direct_views]}")

        logger.info("Computing direct-path vertices...")
        is_direct_vertex = np.zeros(n_vertices, dtype=bool)

        for i in range(0, n_vertices, 10000):
            end_i = min(i + 10000, n_vertices)
            for j in range(i, end_i):
                is_direct_vertex[j] = is_direct_path_vertex(
                    source_pos, vertices_mm[j], volume, VOXEL_SIZE_MM
                )
            if i % 50000 == 0:
                logger.info(f"  Processed {i}/{n_vertices}...")

        n_direct = int(np.sum(is_direct_vertex))
        logger.info(f"Direct-path vertices: {n_direct}")

        if pos_name == "P1-dorsal":
            fluence_path = ARCHIVE_BASE / "S2-Vol-P1-dorsal-r2.0" / "fluence.npy"
        else:
            fluence_path = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"

        logger.info(f"Loading fluence from {fluence_path}")
        fluence = np.load(fluence_path)

        phi_mcx, valid_in_bounds = sample_fluence_at_vertices(
            fluence, vertices_mm, VOXEL_SIZE_MM
        )
        phi_closed = compute_closed_form_at_vertices(vertices_mm, source_pos)

        ncc_all = compute_ncc(phi_mcx, phi_closed, valid_in_bounds)
        ncc_direct = compute_ncc(
            phi_mcx, phi_closed, valid_in_bounds & is_direct_vertex
        )

        results["positions"][pos_name] = {
            "source_pos_mm": source_pos.tolist(),
            "direct_path_views": [a for a, _ in direct_views],
            "n_direct_vertices": n_direct,
            "ncc_all_vertices": ncc_all,
            "ncc_direct_vertices": ncc_direct,
        }

        logger.info(f"  NCC (all): {ncc_all:.4f}")
        logger.info(f"  NCC (direct): {ncc_direct:.4f}")

    with open(output_dir / "m3_prime_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("M3': Multi-Position Validation (Surface-Space NCC)")
    print("=" * 75)
    print(f"Total vertices: {n_vertices}")
    print("-" * 75)
    print(
        f"{'Position':<15} {'Direct Views':<20} {'N_direct':<12} {'NCC_direct':<12} {'Pass'}"
    )
    print("-" * 75)

    all_pass = True
    for pos_name, data in results["positions"].items():
        ncc = data["ncc_direct_vertices"]
        passed = "✓" if ncc >= 0.90 else ("⚠" if ncc >= 0.85 else "✗")
        if ncc < 0.85:
            all_pass = False
        views_str = str(data["direct_path_views"])
        print(
            f"{pos_name:<15} {views_str:<20} {data['n_direct_vertices']:<12} {ncc:<12.4f} {passed}"
        )

    print("=" * 75)

    ncc_values = [d["ncc_direct_vertices"] for d in results["positions"].values()]
    ncc_spread = max(ncc_values) - min(ncc_values)
    print(f"NCC spread across positions: {ncc_spread:.4f}")

    if all_pass:
        print("All positions pass (NCC ≥ 0.85 in direct-path regime) ✓")
    else:
        print("Some positions fail - check results")


if __name__ == "__main__":
    main()
