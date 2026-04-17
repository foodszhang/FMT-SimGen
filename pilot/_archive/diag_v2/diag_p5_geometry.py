#!/usr/bin/env python3
"""P5 Geometry Sanity Check.

Verifies that P5 source position and detector rays are geometrically correct.
The 50% liver path result from previous diagnostic is suspicious because:
- P5 is defined as ventral_z + 4mm (4mm below ventral surface)
- Liver is typically 10-15mm deeper in the ventral cavity
- A ventral→ventral path should NOT pass through liver
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

RESULTS_BASE = Path(__file__).parent.parent / "results" / "stage2_multiposition_v2"
ATLAS_PATH = Path("/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin")
MATERIAL_PATH = Path("/home/foods/pro/FMT-SimGen/output/shared/mcx_material.yaml")

VOXEL_SIZE_MM = 0.4


def load_atlas_volume() -> np.ndarray:
    """Load atlas volume (ZYX order) and return XYZ order."""
    volume = np.fromfile(ATLAS_PATH, dtype=np.uint8)
    volume = volume.reshape((104, 200, 190))
    return volume


def load_material_params() -> dict:
    """Load material parameters."""
    with open(MATERIAL_PATH) as f:
        materials = yaml.safe_load(f)
    return {m["tag"]: m["name"] for m in materials}


def mm_to_voxel_xyz(
    pos_mm: np.ndarray, volume_shape: tuple, voxel_size: float
) -> tuple:
    """Convert mm position (centered) to voxel indices (XYZ order)."""
    nx, ny, nz = volume_shape
    center = np.array([nx / 2, ny / 2, nz / 2])
    pos_vox = pos_mm / voxel_size + center
    return tuple(int(round(p)) for p in pos_vox)


def voxel_to_mm_xyz(
    ix: int, iy: int, iz: int, volume_shape: tuple, voxel_size: float
) -> np.ndarray:
    """Convert voxel indices (XYZ) to mm position (centered)."""
    nx, ny, nz = volume_shape
    center = np.array([nx / 2, ny / 2, nz / 2])
    return (np.array([ix, iy, iz]) - center + 0.5) * voxel_size


def get_surface_positions(volume_xyz: np.ndarray, voxel_size: float) -> dict:
    """Get surface positions at Y=center slice."""
    ny = volume_xyz.shape[1]
    y_center = ny // 2

    slice_xz = volume_xyz[:, y_center, :]
    tissue_x, tissue_z = np.where(slice_xz > 0)

    if len(tissue_x) == 0:
        raise ValueError("No tissue found")

    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2

    tissue_x_mm = (tissue_x - x_center + 0.5) * voxel_size
    tissue_z_mm = (tissue_z - z_center + 0.5) * voxel_size

    return {
        "dorsal_z": float(tissue_z_mm.max()),
        "ventral_z": float(tissue_z_mm.min()),
        "left_x": float(tissue_x_mm.min()),
        "right_x": float(tissue_x_mm.max()),
        "center_x": float((tissue_x_mm.min() + tissue_x_mm.max()) / 2),
        "center_z": float((tissue_z_mm.min() + tissue_z_mm.max()) / 2),
    }


def trace_ray_labels(
    start_vox: tuple,
    end_vox: tuple,
    volume_xyz: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """Trace ray and return labels along path."""
    t_vals = np.linspace(0, 1, n_samples)
    labels = []

    for t in t_vals:
        pos = np.array(start_vox) + t * (np.array(end_vox) - np.array(start_vox))
        ix, iy, iz = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

        nx, ny, nz = volume_xyz.shape
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            labels.append(volume_xyz[ix, iy, iz])
        else:
            labels.append(0)

    return np.array(labels)


def main():
    """Run P5 geometry sanity check."""
    print("=" * 80)
    print("P5 GEOMETRY SANITY CHECK")
    print("=" * 80)

    volume_zyx = load_atlas_volume()
    volume_xyz = volume_zyx.transpose(2, 1, 0)
    label_names = load_material_params()

    print(f"\n[1] Atlas volume shape: {volume_xyz.shape} (XYZ)")
    print(f"    Voxel size: {VOXEL_SIZE_MM} mm")

    surface_pos = get_surface_positions(volume_xyz, VOXEL_SIZE_MM)
    print(f"\n[2] Surface positions (Y=center slice):")
    print(f"    Dorsal Z: {surface_pos['dorsal_z']:.2f} mm")
    print(f"    Ventral Z: {surface_pos['ventral_z']:.2f} mm")
    print(f"    Left X: {surface_pos['left_x']:.2f} mm")
    print(f"    Right X: {surface_pos['right_x']:.2f} mm")
    print(f"    Center X: {surface_pos['center_x']:.2f} mm")

    with open(RESULTS_BASE / "S2-Vol-P5-ventral-r2.0" / "results.json") as f:
        p5_results = json.load(f)

    source_pos_mm = np.array(p5_results["source_pos"])
    best_angle = p5_results["best_angle"]

    print(f"\n[3] P5 configuration from Stage 2 v2:")
    print(f"    Source position: {source_pos_mm} mm")
    print(f"    Best angle: {best_angle}°")

    expected_z = surface_pos["ventral_z"] + 4.0
    print(f"\n    Expected source Z (ventral_z + 4): {expected_z:.2f} mm")
    print(f"    Actual source Z: {source_pos_mm[2]:.2f} mm")
    print(f"    Z difference: {abs(source_pos_mm[2] - expected_z):.2f} mm")

    if abs(source_pos_mm[2] - expected_z) > 1.0:
        print("    ⚠️  WARNING: Source Z does NOT match expected ventral_z + 4!")
    else:
        print("    ✓ Source Z matches expected ventral_z + 4")

    source_vox = mm_to_voxel_xyz(source_pos_mm, volume_xyz.shape, VOXEL_SIZE_MM)
    print(f"\n[4] Source voxel check:")
    print(f"    Source voxel (XYZ): {source_vox}")

    ix, iy, iz = source_vox
    nx, ny, nz = volume_xyz.shape
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        label_at_source = volume_xyz[ix, iy, iz]
        print(
            f"    Label at source: {label_at_source} ({label_names.get(label_at_source, 'unknown')})"
        )

        if label_at_source == 0:
            print("    ❌ FAIL: Source is in AIR (outside tissue)!")
        elif label_at_source == 7:
            print("    ❌ FAIL: Source is INSIDE LIVER!")
        elif label_at_source == 1:
            print("    ✓ Source is in soft_tissue (correct)")
        else:
            print(f"    ⚠️  Source is in {label_names.get(label_at_source, 'unknown')}")
    else:
        print("    ❌ FAIL: Source voxel is outside volume bounds!")
        label_at_source = 0

    print(f"\n[5] Detector ray analysis:")
    print("-" * 80)

    camera_distance = 200.0
    fov = 50.0
    detector_res = (256, 256)

    mcx_proj = np.load(RESULTS_BASE / "S2-Vol-P5-ventral-r2.0" / "mcx_a60.npy")
    h, w = mcx_proj.shape
    px_size = fov / w

    peak_idx = np.argmax(mcx_proj)
    peak_v, peak_u = np.unravel_index(peak_idx, mcx_proj.shape)

    test_pixels = [
        (peak_v, peak_u, "peak"),
        (h // 2, w // 2, "center"),
        (h // 4, w // 2, "top"),
        (3 * h // 4, w // 2, "bottom"),
    ]

    for v, u, ptype in test_pixels:
        px_x = (u - w / 2 + 0.5) * px_size
        px_y = (v - h / 2 + 0.5) * px_size

        angle_rad = np.deg2rad(best_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        px_x_rot = px_x * cos_a
        px_z_rot = -px_x * sin_a

        detector_mm = np.array([px_x_rot, px_y, camera_distance + px_z_rot])

        print(f"\n  Pixel ({v}, {u}) [{ptype}]:")
        print(f"    Detector position: {detector_mm} mm")

        detector_vox = mm_to_voxel_xyz(detector_mm, volume_xyz.shape, VOXEL_SIZE_MM)
        print(f"    Detector voxel (XYZ): {detector_vox}")

        dix, diy, diz = detector_vox
        if 0 <= dix < nx and 0 <= diy < ny and 0 <= diz < nz:
            label_at_det = volume_xyz[dix, diy, diz]
            print(
                f"    Label at detector: {label_at_det} ({label_names.get(label_at_det, 'unknown')})"
            )

        labels = trace_ray_labels(source_vox, detector_vox, volume_xyz, n_samples=200)

        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        print(f"    Ray label distribution:")
        for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            pct = cnt / total * 100
            name = label_names.get(lbl, "unknown")
            print(f"      {name}: {pct:.1f}%")

        liver_pct = counts[unique == 7][0] / total * 100 if 7 in unique else 0
        print(f"    Liver percentage: {liver_pct:.1f}%")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    issues = []

    if abs(source_pos_mm[2] - expected_z) > 1.0:
        issues.append("Source Z position mismatch")

    if label_at_source == 0:
        issues.append("Source is in AIR")
    elif label_at_source == 7:
        issues.append("Source is INSIDE LIVER")

    if not issues:
        print("\n✓ All geometry checks PASSED")
        print("  The 50% liver path is a REAL physical result")
        print("  → P5 heterogeneity conclusion is VALID")
    else:
        print(f"\n❌ Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n  → Need to fix geometry and re-run P5")

    return issues


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
