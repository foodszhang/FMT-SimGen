#!/usr/bin/env python3
"""P5 Tissue Path Analysis for heterogeneity attribution.

Analyzes the photon paths from source to detector for P5-ventral position:
1. Loads Digimouse 10-label volume
2. For source center → ventral detector pixels, traces ray paths
3. Counts tissue composition along each path
4. Computes effective μ_a weighted by path length
5. Compares with homogeneous (soft_tissue) assumption
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"
ATLAS_PATH = Path("/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin")
MATERIAL_PATH = Path("/home/foods/pro/FMT-SimGen/output/shared/mcx_material.yaml")

VOXEL_SIZE_MM = 0.4
DOWNSAMPLE_FACTOR = 2


def load_atlas_volume() -> np.ndarray:
    """Load the original atlas volume (ZYX order)."""
    volume = np.fromfile(ATLAS_PATH, dtype=np.uint8)
    volume = volume.reshape((104, 200, 190))
    return volume


def load_material_params() -> Dict:
    """Load material optical parameters."""
    with open(MATERIAL_PATH) as f:
        materials = yaml.safe_load(f)

    params = {}
    for m in materials:
        params[m["tag"]] = {
            "name": m["name"],
            "mua": m["mua"],
            "mus": m["mus"],
            "g": m["g"],
        }
    return params


def trace_ray_through_volume(
    start_mm: np.ndarray,
    end_mm: np.ndarray,
    volume: np.ndarray,
    voxel_size_mm: float,
    n_samples: int = 100,
) -> Dict:
    """Trace a ray through the volume and count tissue labels.

    Parameters
    ----------
    start_mm : np.ndarray
        Start point in mm (centered coordinates)
    end_mm : np.ndarray
        End point in mm (centered coordinates)
    volume : np.ndarray
        Atlas volume (ZYX order)
    voxel_size_mm : float
        Voxel size in mm
    n_samples : int
        Number of samples along the ray

    Returns
    -------
    Dict with tissue path statistics
    """
    nz, ny, nx = volume.shape
    center = np.array([nx / 2, ny / 2, nz / 2])

    t_values = np.linspace(0, 1, n_samples)
    path_length = np.linalg.norm(end_mm - start_mm)

    tissue_counts = {}
    total_path = 0.0

    for t in t_values:
        pos_mm = start_mm + t * (end_mm - start_mm)

        pos_vox = pos_mm / voxel_size_mm + center
        pos_vox = pos_vox[::-1]

        ix = int(round(pos_vox[2]))
        iy = int(round(pos_vox[1]))
        iz = int(round(pos_vox[0]))

        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            label = volume[iz, iy, ix]
            tissue_counts[label] = tissue_counts.get(label, 0) + 1
            total_path += 1

    if total_path > 0:
        for label in tissue_counts:
            tissue_counts[label] = tissue_counts[label] / total_path * path_length

    return {
        "tissue_counts": tissue_counts,
        "total_path_mm": path_length,
        "n_samples": int(total_path),
    }


def compute_effective_mua(
    tissue_path: Dict,
    material_params: Dict,
) -> Tuple[float, float]:
    """Compute effective μ_a weighted by path length.

    Returns
    -------
    mua_eff : float
        Effective absorption coefficient (/mm)
    mua_soft_tissue : float
        Absorption coefficient for soft_tissue (/mm)
    """
    mua_eff = 0.0
    total_path = tissue_path["total_path_mm"]

    for label, path_mm in tissue_path["tissue_counts"].items():
        if label in material_params:
            mua = material_params[label]["mua"]
            mua_eff += mua * (path_mm / total_path)

    mua_soft_tissue = material_params[1]["mua"]

    return mua_eff, mua_soft_tissue


def get_p5_detector_pixels(
    projection: np.ndarray,
    n_pixels: int = 6,
) -> List[Tuple[int, int, str]]:
    """Select detector pixels for path analysis.

    Returns pixels from:
    - 2 brightest pixels (peak signal)
    - 2 edge pixels (medium signal)
    - 2 noise floor pixels (low signal)
    """
    h, w = projection.shape
    pixels = []

    flat = projection.flatten()
    sorted_idx = np.argsort(flat)[::-1]

    peak_indices = sorted_idx[:2]
    for idx in peak_indices:
        v, u = np.unravel_index(idx, projection.shape)
        pixels.append((int(v), int(u), "peak"))

    nonzero_mask = projection > 0
    nonzero_vals = projection[nonzero_mask]
    if len(nonzero_vals) > 10:
        median_val = np.median(nonzero_vals)
        edge_mask = (projection > median_val * 0.5) & (projection < median_val * 2)
        edge_coords = np.argwhere(edge_mask)
        if len(edge_coords) >= 2:
            for i in range(min(2, len(edge_coords))):
                v, u = edge_coords[i]
                pixels.append((int(v), int(u), "edge"))

    low_mask = (projection > 0) & (projection < np.percentile(nonzero_vals, 10))
    low_coords = np.argwhere(low_mask)
    if len(low_coords) >= 2:
        for i in range(min(2, len(low_coords))):
            v, u = low_coords[i]
            pixels.append((int(v), int(u), "floor"))

    return pixels


def analyze_p5_paths():
    """Main analysis for P5 tissue paths."""
    print("=" * 80)
    print("P5 TISSUE PATH ANALYSIS")
    print("=" * 80)

    print("\n[1] Loading data...")
    volume_zyx = load_atlas_volume()
    material_params = load_material_params()

    with open(RESULTS_BASE / "S2-Vol-P5-ventral-r2.0" / "results.json") as f:
        p5_results = json.load(f)

    mcx_proj = np.load(RESULTS_BASE / "S2-Vol-P5-ventral-r2.0" / "mcx_a60.npy")
    green_proj = np.load(RESULTS_BASE / "S2-Vol-P5-ventral-r2.0" / "green_a60.npy")

    source_pos_mm = np.array(p5_results["source_pos"])
    print(f"  Source position: {source_pos_mm} mm")
    print(f"  Volume shape: {volume_zyx.shape} (ZYX)")

    print("\n[2] Material properties:")
    print("-" * 80)
    print(f"{'Label':<8} {'Name':<15} {'μa (/mm)':<12} {'μs (/mm)':<12}")
    print("-" * 80)
    for label, params in sorted(material_params.items()):
        print(
            f"{label:<8} {params['name']:<15} {params['mua']:<12.4f} {params['mus']:<12.2f}"
        )

    soft_tissue_mua = material_params[1]["mua"]
    liver_mua = material_params[7]["mua"]
    print(f"\n  Key comparison:")
    print(f"    soft_tissue μa = {soft_tissue_mua:.4f} /mm")
    print(f"    liver μa = {liver_mua:.4f} /mm")
    print(f"    ratio = {liver_mua / soft_tissue_mua:.1f}×")

    print("\n[3] Selecting detector pixels...")
    pixels = get_p5_detector_pixels(mcx_proj, n_pixels=6)
    print(f"  Selected {len(pixels)} pixels:")
    for v, u, ptype in pixels:
        val = mcx_proj[v, u]
        print(f"    ({v}, {u}) [{ptype}]: MCX={val:.4e}")

    print("\n[4] Tracing rays from source to detector...")
    print("-" * 80)

    camera_distance = 200.0
    fov = 50.0
    h, w = mcx_proj.shape
    px_size = fov / w

    all_results = []

    for v, u, ptype in pixels:
        px_x = (u - w / 2 + 0.5) * px_size
        px_y = (v - h / 2 + 0.5) * px_size

        angle_rad = np.deg2rad(60)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        px_x_rot = px_x * cos_a
        px_y_rot = px_y
        px_z_rot = -px_x * sin_a

        detector_mm = np.array([px_x_rot, px_y_rot, camera_distance + px_z_rot])

        ray_result = trace_ray_through_volume(
            source_pos_mm, detector_mm, volume_zyx, VOXEL_SIZE_MM * DOWNSAMPLE_FACTOR
        )

        mua_eff, mua_soft = compute_effective_mua(ray_result, material_params)

        print(f"\n  Pixel ({v}, {u}) [{ptype}]:")
        print(f"    Detector position: {detector_mm} mm")
        print(f"    Path length: {ray_result['total_path_mm']:.2f} mm")
        print(f"    Effective μa: {mua_eff:.4f} /mm (vs soft_tissue {mua_soft:.4f})")
        print(f"    μa ratio: {mua_eff / mua_soft:.2f}×")
        print(f"    Tissue composition:")

        for label, path_mm in sorted(
            ray_result["tissue_counts"].items(), key=lambda x: -x[1]
        ):
            if label in material_params:
                name = material_params[label]["name"]
                pct = path_mm / ray_result["total_path_mm"] * 100
                print(f"      {name}: {path_mm:.2f} mm ({pct:.1f}%)")

        all_results.append(
            {
                "pixel": (v, u),
                "type": ptype,
                "detector_mm": detector_mm.tolist(),
                "path_length_mm": float(ray_result["total_path_mm"]),
                "mua_eff": float(mua_eff),
                "mua_ratio": float(mua_eff / mua_soft),
                "tissue_composition": {
                    material_params[l]["name"]: p / ray_result["total_path_mm"] * 100
                    for l, p in ray_result["tissue_counts"].items()
                    if l in material_params
                },
            }
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mua_ratios = [r["mua_ratio"] for r in all_results]
    print(f"\n  Average μa ratio: {np.mean(mua_ratios):.2f}×")
    print(f"  Max μa ratio: {np.max(mua_ratios):.2f}×")
    print(f"  Min μa ratio: {np.min(mua_ratios):.2f}×")

    liver_percentages = []
    for r in all_results:
        liver_pct = r["tissue_composition"].get("liver", 0)
        liver_percentages.append(liver_pct)

    print(f"\n  Liver path percentage:")
    print(f"    Average: {np.mean(liver_percentages):.1f}%")
    print(f"    Max: {np.max(liver_percentages):.1f}%")

    print("\n[5] CONCLUSION")
    print("-" * 80)

    if np.mean(liver_percentages) >= 30:
        print("  ✓ Liver occupies ≥30% of P5 paths → HETEROGENEITY CONFIRMED")
    else:
        print(f"  ✗ Liver only {np.mean(liver_percentages):.1f}% of paths")
        print("    → Need to investigate other high-μa tissues")

    if np.mean(mua_ratios) >= 1.5:
        print(f"  ✓ Effective μa is {np.mean(mua_ratios):.2f}× soft_tissue")
        print("    → Significant absorption increase from heterogeneity")
    else:
        print(f"  ~ Effective μa is only {np.mean(mua_ratios):.2f}× soft_tissue")
        print("    → Heterogeneity effect is moderate")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "p5_tissue_path_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved: {output_dir / 'p5_tissue_path_analysis.json'}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    analyze_p5_paths()
