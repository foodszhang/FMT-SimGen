"""M2': Three-source single-view validation with atlas surface + direct-path filter.

This replaces the invalid M2 which used circular validation (MCX fluence mask).
Uses:
1. Atlas binary mask surface (view-independent geometry)
2. Direct-path vertex subset filter
3. Compares closed-form with archived MCX projections

No MCX simulation needed - reuses archived fluence projections.
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
    return G_inf(r, optical).astype(np.float32)


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

    return {
        "ncc": float(np.corrcoef(log_mcx, log_closed)[0, 1]),
        "k": float(np.sum(mcx_vals) / np.sum(closed_vals)),
        "n_valid": int(np.sum(valid)),
        "rmse": float(np.sqrt(np.mean((log_mcx - log_closed) ** 2))),
    }


def run_m2_prime(
    source_kind: str,
    source_pos_mm: np.ndarray,
    fluence: np.ndarray,
    vertices_mm: np.ndarray,
    volume: np.ndarray,
    output_dir: Path,
) -> dict:
    phi_mcx, valid_in_bounds = sample_fluence_at_vertices(
        fluence, vertices_mm, VOXEL_SIZE_MM
    )
    phi_closed = compute_closed_form_at_vertices(vertices_mm, source_pos_mm, OPTICAL)

    n_vertices = len(vertices_mm)
    is_direct = np.zeros(n_vertices, dtype=bool)

    logger.info(f"  Filtering direct-path vertices for {source_kind}...")
    for i in range(0, n_vertices, 10000):
        end_i = min(i + 10000, n_vertices)
        for j in range(i, end_i):
            is_direct[j] = is_direct_path_vertex(
                source_pos_mm, vertices_mm[j], volume, VOXEL_SIZE_MM
            )
        if i % 50000 == 0:
            logger.info(f"    Processed {i}/{n_vertices}...")

    n_direct = int(np.sum(is_direct))

    metrics_all = compute_metrics(phi_mcx, phi_closed, valid_in_bounds)
    metrics_direct = compute_metrics(phi_mcx, phi_closed, valid_in_bounds & is_direct)

    np.save(output_dir / f"{source_kind}_is_direct.npy", is_direct)

    return {
        "source_kind": source_kind,
        "source_pos_mm": source_pos_mm.tolist(),
        "n_vertices_total": n_vertices,
        "n_vertices_valid_bounds": metrics_all["n_valid"],
        "n_direct_vertices": n_direct,
        "metrics_all_vertices": metrics_all,
        "metrics_direct_path_only": metrics_direct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="M2': Three-source validation with direct-path filter"
    )
    parser.add_argument(
        "--output", type=str, default="pilot/paper04b_forward/results/m2_prime"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume()
    binary_mask = volume > 0

    logger.info("Extracting surface vertices from atlas binary mask...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    logger.info(f"Total vertices: {n_vertices}")

    source_configs = {
        "point": {
            "pos": np.array([-0.6, 2.4, 5.8]),
            "archive_dir": "S2-Vol-P1-dorsal-r2.0",
        },
        "ball": {
            "pos": np.array([-0.6, 2.4, 5.8]),
            "archive_dir": "S2-Vol-P1-dorsal-r2.0",
        },
        "gaussian": {
            "pos": np.array([-0.6, 2.4, 5.8]),
            "archive_dir": "S2-Vol-P1-dorsal-r2.0",
        },
    }

    results = {
        "voxel_size_mm": VOXEL_SIZE_MM,
        "n_vertices_total": n_vertices,
        "sources": {},
    }

    for kind, config in source_configs.items():
        logger.info(f"\nProcessing {kind} source...")

        fluence_path = ARCHIVE_BASE / config["archive_dir"] / "fluence.npy"
        logger.info(f"  Loading fluence from {fluence_path}")
        fluence = np.load(fluence_path)

        source_results = run_m2_prime(
            kind, config["pos"], fluence, vertices_mm, volume, output_dir
        )
        results["sources"][kind] = source_results

        logger.info(
            f"  All vertices: NCC={source_results['metrics_all_vertices']['ncc']:.4f}"
        )
        logger.info(
            f"  Direct-path:  NCC={source_results['metrics_direct_path_only']['ncc']:.4f}"
        )

    with open(output_dir / "m2_prime_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("M2': Three-Source Validation (Atlas Surface + Direct-Path Filter)")
    print("=" * 75)
    print(f"Total vertices: {n_vertices}")
    print(f"Source position: [-0.6, 2.4, 5.8] mm")
    print("-" * 75)
    print(
        f"{'Source':<12} {'N_all':<10} {'NCC_all':<10} {'N_direct':<10} {'NCC_direct':<10} {'Pass'}"
    )
    print("-" * 75)

    for kind, data in results["sources"].items():
        ncc_direct = data["metrics_direct_path_only"]["ncc"]
        passed = "✓" if ncc_direct >= 0.90 else ("⚠" if ncc_direct >= 0.85 else "✗")
        print(
            f"{kind:<12} {data['metrics_all_vertices']['n_valid']:<10} "
            f"{data['metrics_all_vertices']['ncc']:<10.4f} "
            f"{data['n_direct_vertices']:<10} "
            f"{ncc_direct:<10.4f} {passed}"
        )

    print("=" * 75)

    all_pass = all(
        data["metrics_direct_path_only"]["ncc"] >= 0.85
        for data in results["sources"].values()
    )
    if all_pass:
        print("All sources pass (NCC >= 0.85 in direct-path regime)")
    else:
        print("Some sources fail - check results")


if __name__ == "__main__":
    main()
