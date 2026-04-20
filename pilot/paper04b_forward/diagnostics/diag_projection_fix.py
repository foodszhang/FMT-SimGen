"""Projection NCC Fix Diagnosis: F-1 through F-5.

Goal: Identify why projection linear NCC = 0.65 instead of expected ≥ 0.9.

Only uses P5-ventral @ +150° (best angle from previous run).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.metrics import ncc
from shared.green_surface_projection import (
    render_green_surface_projection,
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)
from fmt_simgen.mcx_projection import project_volume_reference

VOXEL_SIZE_MM = 0.4
ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/projection_fix")

GT_POS = np.array([-0.6, 2.4, -3.8])
BEST_ANGLE = 150


def load_data():
    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))
    fluence = np.load(ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")
    return atlas, fluence


def get_projections(atlas, fluence, angle):
    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    proj_mcx, depth_mcx = project_volume_reference(
        fluence,
        angle,
        voxel_size_mm=VOXEL_SIZE_MM,
        camera_distance=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
    )

    proj_green = render_green_surface_projection(
        GT_POS,
        atlas > 0,
        angle,
        camera_distance_mm=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
        tissue_params=tissue_params,
        voxel_size_mm=VOXEL_SIZE_MM,
    )

    return proj_mcx, proj_green, tissue_params


def f1_mask_alignment(proj_mcx, proj_green):
    """F-1: Check valid_mask alignment between MCX and Green projections."""
    print("\n" + "=" * 70)
    print("F-1: Valid Mask Alignment")
    print("=" * 70)

    mask_mcx = proj_mcx > 0
    mask_green = proj_green > 0

    n_mcx = np.sum(mask_mcx)
    n_green = np.sum(mask_green)

    intersection = np.sum(mask_mcx & mask_green)
    union = np.sum(mask_mcx | mask_green)
    iou = intersection / union if union > 0 else 0

    print(f"  MCX mask pixels: {n_mcx}")
    print(f"  Green mask pixels: {n_green}")
    print(f"  Intersection: {intersection}")
    print(f"  Union: {union}")
    print(f"  IoU: {iou:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ax = axes[0, 0]
    ax.imshow(mask_mcx, cmap="gray", origin="lower")
    ax.set_title(f"MCX mask (n={n_mcx})")

    ax = axes[0, 1]
    ax.imshow(mask_green, cmap="gray", origin="lower")
    ax.set_title(f"Green mask (n={n_green})")

    ax = axes[1, 0]
    ax.imshow(mask_mcx & mask_green, cmap="Greens", origin="lower")
    ax.set_title(f"Intersection (n={intersection})")

    ax = axes[1, 1]
    xor_mask = mask_mcx ^ mask_green
    ax.imshow(xor_mask, cmap="Reds", origin="lower")
    ax.set_title(f"XOR / non-overlap (n={np.sum(xor_mask)})")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "F1_mask_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    verdict = "PASS" if iou >= 0.9 else "FAIL"
    print(f"  Verdict: {verdict} (IoU {'≥' if iou >= 0.9 else '<'} 0.9)")

    return {
        "n_mcx": int(n_mcx),
        "n_green": int(n_green),
        "intersection": int(intersection),
        "union": int(union),
        "iou": float(iou),
        "verdict": verdict,
    }


def f2_ncc_decomposition(proj_mcx, proj_green):
    """F-2: Decompose NCC into shape vs amplitude components."""
    print("\n" + "=" * 70)
    print("F-2: NCC Decomposition")
    print("=" * 70)

    valid = (proj_mcx > 0) & (proj_green > 0)
    mcx_valid = proj_mcx[valid]
    green_valid = proj_green[valid]

    linear_ncc = ncc(mcx_valid, green_valid)

    rank_corr, _ = spearmanr(mcx_valid, green_valid)

    eps = 1e-20
    log_ncc = ncc(np.log10(mcx_valid + eps), np.log10(green_valid + eps))

    total_mcx = np.sum(mcx_valid)
    total_green = np.sum(green_valid)
    mcx_centroid = np.array(
        [
            np.sum(np.where(valid)[1] * proj_mcx[valid]) / total_mcx,
            np.sum(np.where(valid)[0] * proj_mcx[valid]) / total_mcx,
        ]
    )
    green_centroid = np.array(
        [
            np.sum(np.where(valid)[1] * proj_green[valid]) / total_green,
            np.sum(np.where(valid)[0] * proj_green[valid]) / total_green,
        ]
    )
    centroid_offset = np.linalg.norm(mcx_centroid - green_centroid)

    peak_mcx = np.unravel_index(np.argmax(proj_mcx), proj_mcx.shape)
    peak_green = np.unravel_index(np.argmax(proj_green), proj_green.shape)
    peak_offset = np.linalg.norm(np.array(peak_mcx) - np.array(peak_green))

    print(f"  Linear NCC: {linear_ncc:.4f}")
    print(f"  Rank NCC (Spearman): {rank_corr:.4f}")
    print(f"  Log-linear NCC: {log_ncc:.4f}")
    print(f"  Centroid offset: {centroid_offset:.2f} px")
    print(f"  Peak offset: {peak_offset:.2f} px")
    print(f"    MCX peak: row={peak_mcx[0]}, col={peak_mcx[1]}")
    print(f"    Green peak: row={peak_green[0]}, col={peak_green[1]}")

    verdict_parts = []
    if rank_corr > linear_ncc + 0.1:
        verdict_parts.append("rank>>linear: monotonic OK, amplitude mismatch")
    if log_ncc > linear_ncc + 0.1:
        verdict_parts.append("log>>linear: tail weight difference")
    if peak_offset >= 3:
        verdict_parts.append(
            f"peak offset {peak_offset:.1f}px ≥ 3: geometry misalignment"
        )

    verdict = "; ".join(verdict_parts) if verdict_parts else "PASS"
    print(f"  Verdict: {verdict}")

    return {
        "linear_ncc": float(linear_ncc),
        "rank_ncc": float(rank_corr),
        "log_ncc": float(log_ncc),
        "centroid_offset_px": float(centroid_offset),
        "peak_offset_px": float(peak_offset),
        "mcx_peak": peak_mcx,
        "green_peak": peak_green,
        "verdict": verdict,
    }


def f3_shape_voxel_consistency(atlas, fluence):
    """F-3: Check shape and voxel_size consistency."""
    print("\n" + "=" * 70)
    print("F-3: Shape/Voxel Consistency")
    print("=" * 70)

    print(f"  fluence.shape: {fluence.shape}")
    print(f"  atlas.shape: {atlas.shape}")
    print(f"  Expected shape: (95, 100, 52)")
    print(f"  voxel_size: {VOXEL_SIZE_MM} mm")

    shape_diff = np.abs(np.array(fluence.shape) - np.array(atlas.shape)).max()
    print(f"  Shape difference: {shape_diff}")

    fluence_argmax = np.unravel_index(np.argmax(fluence), fluence.shape)
    print(f"  fluence argmax (voxel): {fluence_argmax}")

    center = np.array(fluence.shape) / 2
    expected_gt_voxel = np.floor(GT_POS / VOXEL_SIZE_MM + center).astype(int)
    print(f"  Expected GT voxel: {tuple(expected_gt_voxel)}")

    voxel_offset = np.linalg.norm(np.array(fluence_argmax) - expected_gt_voxel)
    print(f"  Voxel offset: {voxel_offset:.2f}")

    shape_match = fluence.shape == atlas.shape == (95, 100, 52)
    verdict = "PASS" if shape_match else "FAIL"
    print(f"  Verdict: {verdict}")

    if not shape_match:
        print("\n  *** STOPPING: Shape mismatch detected! ***")
        return {"verdict": "FAIL", "stop": True}

    return {
        "fluence_shape": list(fluence.shape),
        "atlas_shape": list(atlas.shape),
        "shape_diff": int(shape_diff),
        "voxel_size": VOXEL_SIZE_MM,
        "fluence_argmax": fluence_argmax,
        "expected_gt_voxel": tuple(expected_gt_voxel),
        "voxel_offset": float(voxel_offset),
        "verdict": verdict,
        "stop": False,
    }


def f4_rotation_consistency(atlas):
    """F-4: Check rotation axis/direction consistency."""
    print("\n" + "=" * 70)
    print("F-4: Rotation Axis Consistency")
    print("=" * 70)

    test_angles = [-90, 0, 90, 150]

    test_source = np.array([0.0, 0.0, 10.0])

    center = np.array(atlas.shape) / 2
    test_voxel = np.floor(test_source / VOXEL_SIZE_MM + center).astype(int)
    test_voxel = np.clip(test_voxel, 0, np.array(atlas.shape) - 1)

    test_fluence = np.zeros(atlas.shape, dtype=np.float32)
    test_fluence[tuple(test_voxel)] = 1.0

    camera_params = {
        "camera_distance": 200.0,
        "fov_mm": 50.0,
        "detector_resolution": (256, 256),
    }

    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    results = []

    for angle in test_angles:
        proj_mcx, _ = project_volume_reference(
            test_fluence,
            angle,
            voxel_size_mm=VOXEL_SIZE_MM,
            camera_distance=200.0,
            fov_mm=50.0,
            detector_resolution=(256, 256),
        )

        proj_green = render_green_surface_projection(
            test_source,
            atlas > 0,
            angle,
            camera_distance_mm=200.0,
            fov_mm=50.0,
            detector_resolution=(256, 256),
            tissue_params=tissue_params,
            voxel_size_mm=VOXEL_SIZE_MM,
        )

        peak_mcx = np.unravel_index(np.argmax(proj_mcx), proj_mcx.shape)
        peak_green = np.unravel_index(np.argmax(proj_green), proj_green.shape)
        offset = np.linalg.norm(np.array(peak_mcx) - np.array(peak_green))

        print(
            f"  angle {angle:+4d}°: MCX peak {peak_mcx}, Green peak {peak_green}, Δ={offset:.1f} px"
        )

        results.append(
            {
                "angle": angle,
                "mcx_peak": peak_mcx,
                "green_peak": peak_green,
                "offset": float(offset),
            }
        )

    max_offset = max(r["offset"] for r in results)
    verdict = "PASS" if max_offset <= 1 else "FAIL"
    print(
        f"  Verdict: {verdict} (max offset {max_offset:.1f} px {'≤' if max_offset <= 1 else '>'} 1 px)"
    )

    return {
        "angles": results,
        "max_offset": float(max_offset),
        "verdict": verdict,
    }


def f5_homogeneous_sanity():
    """F-5: Test with homogeneous medium."""
    print("\n" + "=" * 70)
    print("F-5: Homogeneous Medium Sanity Check")
    print("=" * 70)

    shape = (200, 200, 200)
    homogeneous_atlas = np.ones(shape, dtype=np.uint8)

    center = np.array(shape) / 2
    test_source = np.array([0.0, 0.0, 0.0])

    test_fluence = np.zeros(shape, dtype=np.float32)
    test_voxel = np.floor(test_source / VOXEL_SIZE_MM + center).astype(int)
    test_fluence[tuple(test_voxel)] = 1.0

    camera_params = {
        "camera_distance": 200.0,
        "fov_mm": 50.0,
        "detector_resolution": (256, 256),
    }

    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    proj_mcx, _ = project_volume_reference(
        test_fluence,
        BEST_ANGLE,
        voxel_size_mm=VOXEL_SIZE_MM,
        camera_distance=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
    )

    proj_green = render_green_surface_projection(
        test_source,
        homogeneous_atlas,
        BEST_ANGLE,
        camera_distance_mm=200.0,
        fov_mm=50.0,
        detector_resolution=(256, 256),
        tissue_params=tissue_params,
        voxel_size_mm=VOXEL_SIZE_MM,
    )

    valid = (proj_mcx > 0) & (proj_green > 0)

    if np.sum(valid) < 100:
        print("  WARNING: Too few valid pixels in homogeneous test")
        return {"verdict": "INCONCLUSIVE", "linear_ncc": None}

    linear_ncc = ncc(proj_mcx[valid], proj_green[valid])

    print(f"  Homogeneous atlas shape: {shape}")
    print(f"  Test source: {test_source}")
    print(f"  Valid pixels: {np.sum(valid)}")
    print(f"  Linear NCC (homogeneous): {linear_ncc:.4f}")

    if linear_ncc >= 0.98:
        verdict = "projection pipeline OK"
        print(f"  Verdict: {verdict}")
        print("  → 0.65 gap comes from finite medium / boundary / G_inf approximation")
    else:
        verdict = "projection pipeline broken"
        print(f"  Verdict: {verdict}")
        print("  → Fix projection pipeline first")

    return {
        "linear_ncc": float(linear_ncc),
        "verdict": verdict,
    }


def write_report(f1, f2, f3, f4, f5):
    """Write diagnostic report."""

    report = f"""# Projection NCC Fix Diagnosis

## Setup
- Source: P5-ventral, gt_pos={GT_POS.tolist()}
- Angle: +{BEST_ANGLE}° (baseline NCC=0.6461)

## F-1: Valid Mask Alignment
- MCX mask pixels: {f1["n_mcx"]}
- Green mask pixels: {f1["n_green"]}
- Intersection: {f1["intersection"]}
- Union: {f1["union"]}
- **IoU: {f1["iou"]:.4f}**
- Verdict: **{f1["verdict"]}**
- Figure: F1_mask_overlay.png

## F-2: NCC Decomposition
- Linear NCC: {f2["linear_ncc"]:.4f}
- Rank NCC (Spearman): {f2["rank_ncc"]:.4f}
- Log-linear NCC: {f2["log_ncc"]:.4f}
- Centroid offset: {f2["centroid_offset_px"]:.2f} px
- **Peak offset: {f2["peak_offset_px"]:.2f} px**
  - MCX peak: row={f2["mcx_peak"][0]}, col={f2["mcx_peak"][1]}
  - Green peak: row={f2["green_peak"][0]}, col={f2["green_peak"][1]}
- Verdict: {f2["verdict"]}

## F-3: Shape/Voxel Consistency
- fluence.shape: {f3["fluence_shape"]}
- atlas.shape: {f3["atlas_shape"]}
- voxel_size: {f3["voxel_size"]} mm
- fluence argmax: {f3["fluence_argmax"]}
- Expected GT voxel: {f3["expected_gt_voxel"]}
- Voxel offset: {f3["voxel_offset"]:.2f}
- Verdict: **{f3["verdict"]}**

## F-4: Rotation Axis Consistency
"""

    for r in f4["angles"]:
        report += f"- angle {r['angle']:+d}°: MCX peak {r['mcx_peak']}, Green peak {r['green_peak']}, Δ={r['offset']:.1f} px\n"

    report += f"- Max offset: {f4['max_offset']:.1f} px\n"
    report += f"- Verdict: **{f4['verdict']}**\n\n"
    report += "## F-5: Homogeneous Medium Sanity\n"
    ncc_str = f"{f5['linear_ncc']:.4f}" if f5["linear_ncc"] is not None else "N/A"
    report += f"- Linear NCC (homogeneous): {ncc_str}\n"
    report += f"- Verdict: **{f5['verdict']}**\n\n"
    report += "## Bottom Line\n\n"

    error_sources = []

    if f1["iou"] < 0.9:
        error_sources.append(
            f"Mask IoU = {f1['iou']:.2f} < 0.9: body outlines don't align"
        )

    if f2["peak_offset_px"] >= 3:
        error_sources.append(
            f"Peak offset = {f2['peak_offset_px']:.1f} px ≥ 3: geometry misalignment"
        )

    if f4["max_offset"] > 1:
        error_sources.append(
            f"Rotation inconsistency: max offset {f4['max_offset']:.1f} px > 1 px"
        )

    if f5["linear_ncc"] is not None and f5["linear_ncc"] < 0.98:
        error_sources.append(
            f"Homogeneous NCC = {f5['linear_ncc']:.2f} < 0.98: projection pipeline issue"
        )

    if error_sources:
        report += "**Dominant error sources:**\n"
        for i, src in enumerate(error_sources, 1):
            report += f"{i}. {src}\n"
        report += f"\n**Fix direction:** Address the first error source above.\n"
    else:
        report += "**No major error sources found.** Gap may be from finite medium boundary effects.\n"

    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)

    print(f"\nSaved: {OUTPUT_DIR / 'REPORT.md'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Projection NCC Fix Diagnosis: F-1 through F-5")
    print("=" * 70)

    atlas, fluence = load_data()

    f3 = f3_shape_voxel_consistency(atlas, fluence)
    if f3.get("stop"):
        print("\n*** STOPPING due to F-3 FAIL ***")
        return

    proj_mcx, proj_green, tissue_params = get_projections(atlas, fluence, BEST_ANGLE)

    f1 = f1_mask_alignment(proj_mcx, proj_green)
    f2 = f2_ncc_decomposition(proj_mcx, proj_green)
    f4 = f4_rotation_consistency(atlas)
    f5 = f5_homogeneous_sanity()

    write_report(f1, f2, f3, f4, f5)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print(f"F-1 IoU: {f1['iou']:.4f}")
    print(f"F-2 Peak offset: {f2['peak_offset_px']:.1f} px")
    print(f"F-3 Shape match: {f3['verdict']}")
    print(f"F-4 Rotation: {f4['verdict']}")
    ncc_str = f"{f5['linear_ncc']:.4f}" if f5["linear_ncc"] is not None else "N/A"
    print(f"F-5 Homogeneous NCC: {ncc_str}")


if __name__ == "__main__":
    main()
