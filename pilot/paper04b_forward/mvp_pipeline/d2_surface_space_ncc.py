"""D2: Independent surface-space NCC comparison.

This is the decisive experiment for validating closed-form forward.

Key insight: MCX projection is camera-space (line integral along view direction),
while closed-form computes surface fluence (view-independent). The only fair
comparison is on MESH VERTICES, not camera pixels.

Method:
1. Extract skin surface mesh vertices from atlas binary mask (marching cubes)
2. Sample MCX fluence volume at each vertex (nearest neighbor or 1-voxel shell)
3. Compute closed-form Green's function at each vertex
4. NCC / k / RMSE on vertex values

This is view-independent: same vertices for all angles.
If physics is correct, NCC should be similar across all 4 angles.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.metrics import ncc, ncc_log

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
SOURCE_POS_PHYSICAL = np.array([-0.6, 2.4, 5.8])


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


def sample_fluence_at_vertices(
    fluence: np.ndarray, vertices_mm: np.ndarray, voxel_size_mm: float
) -> np.ndarray:
    center = np.array(fluence.shape) / 2
    verts_voxel = vertices_mm / voxel_size_mm + center
    verts_voxel = np.round(verts_voxel).astype(int)
    valid = (
        (verts_voxel[:, 0] >= 0)
        & (verts_voxel[:, 0] < fluence.shape[0])
        & (verts_voxel[:, 1] >= 0)
        & (verts_voxel[:, 1] < fluence.shape[1])
        & (verts_voxel[:, 2] >= 0)
        & (verts_voxel[:, 2] < fluence.shape[2])
    )
    phi_mcx = np.zeros(len(vertices_mm), dtype=np.float32)
    phi_mcx[valid] = fluence[
        verts_voxel[valid, 0], verts_voxel[valid, 1], verts_voxel[valid, 2]
    ]
    return phi_mcx, valid


def sample_fluence_shell_average(
    fluence: np.ndarray,
    vertices_mm: np.ndarray,
    voxel_size_mm: float,
    shell_thickness: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.array(fluence.shape) / 2
    verts_voxel = vertices_mm / voxel_size_mm + center
    verts_voxel = np.round(verts_voxel).astype(int)

    phi_mcx = np.zeros(len(vertices_mm), dtype=np.float32)
    valid = np.ones(len(vertices_mm), dtype=bool)

    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if not (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            valid[i] = False
            continue

        x_start = max(0, vx - shell_thickness)
        x_end = min(fluence.shape[0], vx + shell_thickness + 1)
        y_start = max(0, vy - shell_thickness)
        y_end = min(fluence.shape[1], vy + shell_thickness + 1)
        z_start = max(0, vz - shell_thickness)
        z_end = min(fluence.shape[2], vz + shell_thickness + 1)

        shell = fluence[x_start:x_end, y_start:y_end, z_start:z_end]
        if shell.size > 0 and np.any(shell > 0):
            phi_mcx[i] = np.mean(shell[shell > 0])
        else:
            phi_mcx[i] = fluence[vx, vy, vz] if valid[i] else 0.0

    return phi_mcx, valid


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
    phi_mcx: np.ndarray, phi_closed: np.ndarray, valid: np.ndarray
) -> dict:
    mask = valid & (phi_mcx > 0) & (phi_closed > 0)
    if np.sum(mask) < 10:
        return {"ncc": 0.0, "ncc_log": 0.0, "k": 0.0, "n_valid": 0, "rmse": 0.0}

    mcx_vals = phi_mcx[mask]
    closed_vals = phi_closed[mask]

    ncc_linear = ncc(mcx_vals, closed_vals)
    ncc_log_val = ncc_log(mcx_vals, closed_vals)

    k = np.sum(mcx_vals) / np.sum(closed_vals)

    log_mcx = np.log10(mcx_vals + 1e-20)
    log_closed = np.log10(closed_vals + 1e-20)
    rmse = np.sqrt(np.mean((log_mcx - log_closed) ** 2))

    return {
        "ncc": float(ncc_linear),
        "ncc_log": float(ncc_log_val),
        "k": float(k),
        "n_valid": int(np.sum(mask)),
        "rmse": float(rmse),
        "mcx_sum": float(np.sum(mcx_vals)),
        "closed_sum": float(np.sum(closed_vals)),
    }


def run_d2_validation(
    source_pos_mm: np.ndarray,
    fluence_path: Path,
    output_dir: Path,
) -> dict:
    volume = load_volume()
    binary_mask = volume > 0

    logger.info(f"Extracting surface vertices from atlas binary mask...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    logger.info(f"Extracted {len(vertices_mm)} surface vertices")

    logger.info(f"Loading fluence from {fluence_path}")
    fluence = np.load(fluence_path)
    logger.info(f"Fluence shape: {fluence.shape}, max: {fluence.max():.2e}")

    logger.info("Sampling fluence at surface vertices...")
    phi_mcx, valid = sample_fluence_shell_average(fluence, vertices_mm, VOXEL_SIZE_MM)
    logger.info(f"Valid vertices: {np.sum(valid)} / {len(valid)}")

    logger.info("Computing closed-form Green at vertices...")
    phi_closed = compute_closed_form_at_vertices(vertices_mm, source_pos_mm, OPTICAL)

    metrics = compute_metrics(phi_mcx, phi_closed, valid)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "vertices_mm.npy", vertices_mm)
    np.save(output_dir / "phi_mcx.npy", phi_mcx)
    np.save(output_dir / "phi_closed.npy", phi_closed)
    np.save(output_dir / "valid.npy", valid)

    return {
        "source_pos_mm": source_pos_mm.tolist(),
        "voxel_size_mm": VOXEL_SIZE_MM,
        "n_vertices": len(vertices_mm),
        "n_valid": int(np.sum(valid)),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="D2: Surface-space NCC validation")
    parser.add_argument(
        "--fluence",
        type=str,
        default="pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P1-dorsal-r2.0/fluence.npy",
        help="Path to MCX fluence volume",
    )
    parser.add_argument(
        "--source-pos",
        nargs=3,
        type=float,
        default=[-0.6, 2.4, 5.8],
        help="Source position in mm [x, y, z]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pilot/paper04b_forward/results/d2",
        help="Output directory",
    )
    args = parser.parse_args()

    source_pos_mm = np.array(args.source_pos)
    fluence_path = Path(args.fluence)
    output_dir = Path(args.output)

    logger.info(f"Source position: {source_pos_mm} mm")
    logger.info(
        f"Optical params: mu_a={OPTICAL.mu_a}, mus'={OPTICAL.mus_p}, delta={OPTICAL.delta:.3f}mm"
    )

    results = run_d2_validation(source_pos_mm, fluence_path, output_dir)

    with open(output_dir / "d2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("D2 Surface-Space NCC Results")
    print(f"{'=' * 60}")
    print(f"Source position: {source_pos_mm} mm")
    print(f"Surface vertices: {results['n_vertices']}")
    print(f"Valid vertices: {results['n_valid']}")
    print(f"NCC (log space): {results['ncc']:.4f}")
    print(f"k (MCX/closed): {results['k']:.2e}")
    print(f"RMSE (log): {results['rmse']:.4f}")
    print(f"{'=' * 60}")

    if results["ncc"] >= 0.90:
        print("PASS: NCC >= 0.90, physics layer OK")
    elif results["ncc"] >= 0.70:
        print("WARNING: 0.70 <= NCC < 0.90, boundary case")
    else:
        print("FAIL: NCC < 0.70, physics layer issue")

    return results


if __name__ == "__main__":
    main()
