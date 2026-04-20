"""D2.1: Direct-path vertex subset NCC.

For each vertex in the skin mesh, check if the line from source to vertex
traverses only soft tissue. Compute NCC only on this subset.

This is the authoritative metric for paper §3.3 scope declaration.
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
SOURCE_POS_PHYSICAL = np.array([-0.6, 2.4, 5.8])

SOFT_TISSUE_LABEL = 1
AIR_LABEL = 0


def load_volume() -> np.ndarray:
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    return volume.reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(
    binary_mask: np.ndarray, voxel_size_mm: float
) -> np.ndarray:
    verts, faces, normals, values = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    verts_physical = verts - center
    return verts_physical


def is_direct_path_vertex(
    source_pos_mm: np.ndarray,
    vertex_pos_mm: np.ndarray,
    volume_labels: np.ndarray,
    voxel_size_mm: float,
    max_steps: int = 500,
) -> Tuple[bool, set]:
    """Check if line from source to vertex traverses only soft tissue.

    Returns (is_direct, labels_encountered).
    """
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return True, set()

    direction = direction / distance
    step_mm = 0.1

    labels = set()
    n_steps = int(distance / step_mm)

    for i in range(1, min(n_steps, max_steps) + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = mm_to_voxel(pos_mm)

        if not (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        ):
            break

        label = volume_labels[voxel[0], voxel[1], voxel[2]]
        labels.add(int(label))

        if label not in {AIR_LABEL, SOFT_TISSUE_LABEL}:
            return False, labels

    return True, labels


def sample_fluence_at_vertices(
    fluence: np.ndarray, vertices_mm: np.ndarray, voxel_size_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.array(fluence.shape) / 2
    verts_voxel = vertices_mm / voxel_size_mm + center
    verts_voxel = np.floor(verts_voxel).astype(int)

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
    vertices_mm: np.ndarray, source_pos_mm: np.ndarray, optical
) -> np.ndarray:
    dx = vertices_mm[:, 0] - source_pos_mm[0]
    dy = vertices_mm[:, 1] - source_pos_mm[1]
    dz = vertices_mm[:, 2] - source_pos_mm[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    phi_closed = G_inf(r, optical).astype(np.float32)
    return phi_closed


def compute_metrics(
    phi_mcx: np.ndarray, phi_closed: np.ndarray, mask: np.ndarray
) -> dict:
    valid = mask & (phi_mcx > 0) & (phi_closed > 0)
    if np.sum(valid) < 10:
        return {"ncc": 0.0, "k": 0.0, "n_valid": 0, "rmse": 0.0}

    mcx_vals = phi_mcx[valid]
    closed_vals = phi_closed[valid]

    log_mcx = np.log10(mcx_vals + 1e-20)
    log_closed = np.log10(closed_vals + 1e-20)
    ncc = np.corrcoef(log_mcx, log_closed)[0, 1]
    k = np.sum(mcx_vals) / np.sum(closed_vals)
    rmse = np.sqrt(np.mean((log_mcx - log_closed) ** 2))

    return {
        "ncc": float(ncc),
        "k": float(k),
        "n_valid": int(np.sum(valid)),
        "rmse": float(rmse),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="D2.1: Direct-path vertex subset NCC")
    parser.add_argument("--fluence", type=str, default=None)
    parser.add_argument("--source-pos", nargs=3, type=float, default=[-0.6, 2.4, 5.8])
    parser.add_argument(
        "--output", type=str, default="pilot/paper04b_forward/results/d2_1"
    )
    args = parser.parse_args()

    source_pos_mm = np.array(args.source_pos)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    binary_mask = volume > 0

    logger.info(f"Extracting surface vertices...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    logger.info(f"Total vertices: {n_vertices}")

    if args.fluence:
        fluence_path = Path(args.fluence)
    else:
        fluence_path = Path(
            "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/fluence.npy"
        )

    logger.info(f"Loading fluence from {fluence_path}")
    fluence = np.load(fluence_path)

    logger.info("Sampling fluence at vertices...")
    phi_mcx, valid_in_bounds = sample_fluence_at_vertices(
        fluence, vertices_mm, VOXEL_SIZE_MM
    )

    logger.info("Computing closed-form at vertices...")
    phi_closed = compute_closed_form_at_vertices(vertices_mm, source_pos_mm, OPTICAL)

    logger.info("Filtering direct-path vertices...")
    is_direct = np.zeros(n_vertices, dtype=bool)
    path_labels_all = []

    for i in range(n_vertices):
        if i % 50000 == 0:
            logger.info(f"  Processing vertex {i}/{n_vertices}...")
        ok, labels = is_direct_path_vertex(
            source_pos_mm, vertices_mm[i], volume, VOXEL_SIZE_MM
        )
        is_direct[i] = ok
        if not ok:
            path_labels_all.append(labels)

    n_direct = np.sum(is_direct)
    logger.info(
        f"Direct-path vertices: {n_direct} / {n_vertices} ({100 * n_direct / n_vertices:.1f}%)"
    )

    forbidden_labels = set()
    for labels in path_labels_all:
        forbidden_labels.update(labels - {AIR_LABEL, SOFT_TISSUE_LABEL})
    logger.info(f"Forbidden labels encountered: {forbidden_labels}")

    metrics_all = compute_metrics(phi_mcx, phi_closed, valid_in_bounds)
    metrics_direct = compute_metrics(phi_mcx, phi_closed, valid_in_bounds & is_direct)

    results = {
        "source_pos_mm": source_pos_mm.tolist(),
        "voxel_size_mm": float(VOXEL_SIZE_MM),
        "n_vertices_total": int(n_vertices),
        "n_vertices_valid_bounds": int(np.sum(valid_in_bounds)),
        "n_vertices_direct_path": int(n_direct),
        "fraction_direct": float(n_direct / n_vertices),
        "forbidden_labels": [int(x) for x in sorted(list(forbidden_labels))],
        "metrics_all_vertices": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics_all.items()
        },
        "metrics_direct_path_only": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics_direct.items()
        },
    }

    with open(output_dir / "d2_1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(output_dir / "is_direct_vertex.npy", is_direct)

    print("\n" + "=" * 70)
    print("D2.1: Direct-Path Vertex Subset NCC")
    print("=" * 70)
    print(f"Source position: {source_pos_mm} mm")
    print(f"Total vertices: {n_vertices}")
    print(f"Direct-path vertices: {n_direct} ({100 * n_direct / n_vertices:.1f}%)")
    print(f"Forbidden labels: {sorted(list(forbidden_labels))}")
    print("-" * 70)
    print(f"{'Metric':<25} {'All vertices':<20} {'Direct-path only':<20}")
    print("-" * 70)
    print(f"{'N':<25} {metrics_all['n_valid']:<20} {metrics_direct['n_valid']:<20}")
    print(f"{'NCC':<25} {metrics_all['ncc']:<20.4f} {metrics_direct['ncc']:<20.4f}")
    print(f"{'k':<25} {metrics_all['k']:<20.2e} {metrics_direct['k']:<20.2e}")
    print(f"{'RMSE':<25} {metrics_all['rmse']:<20.4f} {metrics_direct['rmse']:<20.4f}")
    print("=" * 70)

    if metrics_direct["ncc"] >= 0.90:
        print("PASS: NCC >= 0.90, proceed to M2'")
    elif metrics_direct["ncc"] >= 0.80:
        print(f"ACCEPTABLE: 0.80 <= NCC < 0.90, can proceed with paper target >= 0.85")
    else:
        print("FAIL: NCC < 0.80, STOP and investigate")

    return results


if __name__ == "__main__":
    main()
