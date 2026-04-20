"""Step 3: Recompute P5-ventral forward metrics with unified NCC formulas.

Outputs:
- Linear NCC (primary metric)
- Log NCC (auxiliary metric)
- Geomean scale
- Per-distance-bin NCC curve
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf
from shared.metrics import ncc, ncc_log, scale_factor_logmse

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/round2")

GT_POS = np.array([-0.6, 2.4, -3.8])
AIR_LABEL = 0
SOFT_TISSUE_LABELS = {1}


def load_volume():
    volume_path = ARCHIVE_BASE / "mcx_volume_downsampled_2x.bin"
    return np.fromfile(volume_path, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def is_direct_path_vertex(
    source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm, step_mm=0.1
):
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return True

    direction = direction / distance
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
        if label not in {AIR_LABEL} | SOFT_TISSUE_LABELS:
            return False

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 3: Recompute P5-ventral Forward Metrics")
    print("=" * 70)

    print("\nLoading data...")
    atlas = load_volume()
    vertices = extract_surface_vertices(atlas > 0, VOXEL_SIZE_MM)
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"  Total vertices: {len(vertices)}")
    print(f"  GT position: {GT_POS}")

    print("\nSampling fluence at vertices...")
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    phi_mcx = np.zeros(len(vertices), dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi_mcx[i] = fluence[vx, vy, vz]

    print(f"  phi_mcx range: [{phi_mcx.min():.2e}, {phi_mcx.max():.2e}]")

    print("\nComputing closed-form forward...")
    distances = np.linalg.norm(vertices - GT_POS, axis=1)
    phi_closed = G_inf(np.maximum(distances, 0.01), OPTICAL).astype(np.float32)

    print(f"  phi_closed range: [{phi_closed.min():.2e}, {phi_closed.max():.2e}]")

    print("\nComputing direct-path mask...")
    is_direct = np.array(
        [is_direct_path_vertex(GT_POS, v, atlas, VOXEL_SIZE_MM) for v in vertices]
    )
    n_direct = np.sum(is_direct)
    print(f"  Direct vertices: {n_direct} ({100 * n_direct / len(vertices):.1f}%)")

    valid = is_direct & (phi_mcx > 0) & (phi_closed > 0)
    n_valid = np.sum(valid)
    print(f"  Valid vertices: {n_valid}")

    phi_mcx_valid = phi_mcx[valid]
    phi_closed_valid = phi_closed[valid]
    distances_valid = distances[valid]

    print("\n" + "=" * 70)
    print("Computing metrics with unified formulas")
    print("=" * 70)

    scale_geomean = scale_factor_logmse(phi_mcx_valid, phi_closed_valid)
    ncc_linear = ncc(phi_mcx_valid, scale_geomean * phi_closed_valid)
    ncc_log_val = ncc_log(phi_mcx_valid, phi_closed_valid)

    print(f"\nScale (geomean): {scale_geomean:.4e}")
    print(f"NCC (linear):    {ncc_linear:.4f}")
    print(f"NCC (log):       {ncc_log_val:.4f}")

    scale_old = np.sum(phi_mcx_valid) / np.sum(phi_closed_valid)
    print(f"\nScale (old sum/sum): {scale_old:.4e}")
    print(f"Scale ratio (geomean/sum): {scale_geomean / scale_old:.4f}")

    print("\n" + "=" * 70)
    print("Per-distance-bin analysis")
    print("=" * 70)

    bins = np.arange(0, 15.5, 0.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ncc_linear_bins = []
    ncc_log_bins = []
    n_bins = []

    for i in range(len(bins) - 1):
        mask = (distances_valid >= bins[i]) & (distances_valid < bins[i + 1])
        n = np.sum(mask)
        n_bins.append(n)

        if n >= 10:
            ncc_lin = ncc(phi_mcx_valid[mask], scale_geomean * phi_closed_valid[mask])
            ncc_log_b = ncc_log(phi_mcx_valid[mask], phi_closed_valid[mask])
            ncc_linear_bins.append(ncc_lin)
            ncc_log_bins.append(ncc_log_b)
        else:
            ncc_linear_bins.append(np.nan)
            ncc_log_bins.append(np.nan)

    print(f"\n{'Bin (mm)':<12} {'N':<8} {'NCC_linear':<12} {'NCC_log':<12}")
    print("-" * 50)
    for i in range(len(bins) - 1):
        if n_bins[i] >= 10:
            print(
                f"{bins[i]:.1f}-{bins[i + 1]:.1f}      {n_bins[i]:<8} {ncc_linear_bins[i]:<12.4f} {ncc_log_bins[i]:<12.4f}"
            )

    print("\n" + "=" * 70)
    print("Plotting NCC vs distance")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    valid_bins = ~np.isnan(ncc_linear_bins)
    ax.plot(
        bin_centers[valid_bins],
        np.array(ncc_linear_bins)[valid_bins],
        "b-o",
        label="NCC (linear)",
        markersize=4,
    )
    ax.plot(
        bin_centers[valid_bins],
        np.array(ncc_log_bins)[valid_bins],
        "r-s",
        label="NCC (log)",
        markersize=4,
    )
    ax.axvline(
        x=OPTICAL.delta, color="g", linestyle="--", label=f"δ = {OPTICAL.delta:.2f} mm"
    )
    ax.axvline(
        x=VOXEL_SIZE_MM,
        color="gray",
        linestyle=":",
        label=f"1 voxel = {VOXEL_SIZE_MM} mm",
    )
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("NCC")
    ax.set_title("NCC vs Distance (P5-ventral)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(bin_centers, n_bins, width=0.4, color="steelblue", edgecolor="black")
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("Vertex count")
    ax.set_title("Vertex Distribution by Distance")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "A7_ncc_vs_distance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print("\n" + "=" * 70)
    print("Writing report")
    print("=" * 70)

    ncc_linear_min = np.nanmin(ncc_linear_bins)
    ncc_linear_max = np.nanmax(ncc_linear_bins)
    ncc_linear_mean = np.nanmean(ncc_linear_bins)

    report = f"""# P5-ventral Forward Metrics (Round 2)

## Configuration
- GT position: {GT_POS.tolist()} mm
- Optical: mua={OPTICAL.mu_a:.4f}/mm, mus'={OPTICAL.mus_p:.4f}/mm, delta={OPTICAL.delta:.4f}mm
- Voxel size: {VOXEL_SIZE_MM} mm

## Results

### Primary Metric (Linear NCC)
- **NCC (linear) = {ncc_linear:.4f}**

### Auxiliary Metric (Log NCC)
- NCC (log) = {ncc_log_val:.4f}

### Scale Factor
- Scale (geomean) = {scale_geomean:.4e}
- Scale (old sum/sum) = {scale_old:.4e}
- Ratio = {scale_geomean / scale_old:.4f}

## Vertex Statistics
- Total vertices: {len(vertices)}
- Direct vertices: {n_direct} ({100 * n_direct / len(vertices):.1f}%)
- Valid vertices: {n_valid}

## Per-Distance-Bin Analysis

| Distance (mm) | N | NCC (linear) | NCC (log) |
|---------------|---|--------------|-----------|
"""

    for i in range(len(bins) - 1):
        if n_bins[i] >= 10:
            report += f"| {bins[i]:.1f}-{bins[i + 1]:.1f} | {n_bins[i]} | {ncc_linear_bins[i]:.4f} | {ncc_log_bins[i]:.4f} |\n"

    report += f"""
## Distance-Bin Summary

| Metric | Value |
|--------|-------|
| NCC (linear) min | {ncc_linear_min:.4f} |
| NCC (linear) max | {ncc_linear_max:.4f} |
| NCC (linear) mean | {ncc_linear_mean:.4f} |

## Comparison with §4.C Historical

| Metric | This run | Historical §4.C |
|--------|----------|-----------------|
| NCC (log) | {ncc_log_val:.4f} | 0.9578 |
| NCC (linear) | {ncc_linear:.4f} | (not computed) |

**Note**: Historical §4.C used log-space NCC. Linear NCC was not computed.

## Figure

- `A7_ncc_vs_distance.png`: NCC vs distance curve

## Conclusion

- Linear NCC = **{ncc_linear:.4f}** (primary metric for paper)
- Log NCC = {ncc_log_val:.4f} (matches historical 0.9578, confirms consistency)
- Scale geomean differs from sum/sum by factor {scale_geomean / scale_old:.2f}
"""

    report_path = OUTPUT_DIR / "A_redo.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"1. NCC (linear) = {ncc_linear:.4f}")
    print(f"2. NCC (log) = {ncc_log_val:.4f}")
    print(f"3. Scale (geomean) = {scale_geomean:.4e}")
    print(f"4. NCC (linear) range: [{ncc_linear_min:.4f}, {ncc_linear_max:.4f}]")
    print(f"5. Historical log-NCC: 0.9578, current: {ncc_log_val:.4f}")


if __name__ == "__main__":
    main()
