"""Projection NCC check: 2D camera projection with linear NCC.

This is the ORIGINAL protocol from §4.C, not the vertex-based log-NCC
that was accidentally substituted later.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.metrics import ncc
from shared.green_surface_projection import render_green_surface_projection
from fmt_simgen.mcx_projection import project_volume_reference

VOXEL_SIZE_MM = 0.4
ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/projection_ncc")

GT_POS = np.array([-0.6, 2.4, -3.8])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Projection NCC Check: 2D Camera Projection with Linear NCC")
    print("=" * 70)

    print("\nLoading data...")
    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))
    fluence = np.load(ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"  Atlas shape: {atlas.shape}")
    print(f"  Fluence shape: {fluence.shape}")
    print(f"  GT position: {GT_POS}")

    camera_params = {
        "camera_distance_mm": 200.0,
        "fov_mm": 50.0,
        "detector_resolution": (256, 256),
    }

    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    print(f"\nCamera params: {camera_params}")
    print(f"Tissue params: {tissue_params}")

    angles = [-90, -60, -30, 0, 30, 60, 90, 120, 150, 180]

    results = []

    print("\n" + "=" * 70)
    print("Computing projection NCC for each angle")
    print("=" * 70)

    for angle in angles:
        print(f"\nAngle {angle:+d}°:")

        proj_mcx, depth_mcx = project_volume_reference(
            fluence,
            angle,
            camera_distance=camera_params["camera_distance_mm"],
            fov_mm=camera_params["fov_mm"],
            detector_resolution=camera_params["detector_resolution"],
            voxel_size_mm=VOXEL_SIZE_MM,
        )

        proj_green = render_green_surface_projection(
            GT_POS,
            fluence > 0,
            angle,
            tissue_params=tissue_params,
            voxel_size_mm=VOXEL_SIZE_MM,
            **camera_params,
        )

        valid = (proj_mcx > 0) & (proj_green > 0)
        n_valid = int(np.sum(valid))

        print(
            f"  MCX projection: shape={proj_mcx.shape}, range=[{proj_mcx.min():.2e}, {proj_mcx.max():.2e}]"
        )
        print(
            f"  Green projection: shape={proj_green.shape}, range=[{proj_green.min():.2e}, {proj_green.max():.2e}]"
        )
        print(f"  Valid pixels: {n_valid}")

        if n_valid < 100:
            print(f"  SKIP: too few valid pixels")
            continue

        linear_ncc = ncc(proj_mcx[valid], proj_green[valid])

        scale = np.sum(proj_mcx[valid]) / np.sum(proj_green[valid])

        results.append(
            {
                "angle": angle,
                "n_valid": n_valid,
                "linear_ncc": linear_ncc,
                "scale": scale,
                "proj_mcx": proj_mcx,
                "proj_green": proj_green,
                "valid": valid,
            }
        )

        print(f"  Linear NCC: {linear_ncc:.4f}")
        print(f"  Scale: {scale:.4e}")

    if not results:
        print("\nERROR: No valid results!")
        return

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n{'Angle':>8} {'N_valid':>10} {'Linear NCC':>12} {'Scale':>12}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['angle']:>+8d}° {r['n_valid']:>10d} {r['linear_ncc']:>12.4f} {r['scale']:>12.4e}"
        )

    best = max(results, key=lambda x: x["linear_ncc"])
    worst = min(results, key=lambda x: x["linear_ncc"])

    print(f"\nBest angle: {best['angle']:+d}° with NCC = {best['linear_ncc']:.4f}")
    print(f"Worst angle: {worst['angle']:+d}° with NCC = {worst['linear_ncc']:.4f}")

    high_ncc_angles = [r for r in results if r["linear_ncc"] >= 0.9]
    print(f"\nAngles with NCC ≥ 0.9: {len(high_ncc_angles)}")
    for r in high_ncc_angles:
        print(f"  {r['angle']:+d}°: {r['linear_ncc']:.4f}")

    print("\n" + "=" * 70)
    print("Saving comparison plots")
    print("=" * 70)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for idx, r in enumerate([best, worst]):
        if idx >= 2:
            break

        proj_mcx = r["proj_mcx"]
        proj_green = r["proj_green"]
        valid = r["valid"]
        angle = r["angle"]

        ax = axes[idx, 0]
        im = ax.imshow(np.log10(proj_mcx + 1e-10), cmap="viridis", origin="lower")
        ax.set_title(f"MCX @ {angle:+d}° (log)")
        plt.colorbar(im, ax=ax)

        ax = axes[idx, 1]
        im = ax.imshow(np.log10(proj_green + 1e-10), cmap="viridis", origin="lower")
        ax.set_title(f"Green @ {angle:+d}° (log)")
        plt.colorbar(im, ax=ax)

        ax = axes[idx, 2]
        diff = np.zeros_like(proj_mcx)
        diff[valid] = np.log10(proj_mcx[valid] + 1e-10) - np.log10(
            proj_green[valid] + 1e-10
        )
        im = ax.imshow(diff, cmap="RdBu", origin="lower", vmin=-2, vmax=2)
        ax.set_title(f"Diff @ {angle:+d}° (log MCX - log Green)")
        plt.colorbar(im, ax=ax)

    axes[2, 0].axis("off")
    axes[2, 1].axis("off")

    ax = axes[2, 2]
    angles_plot = [r["angle"] for r in results]
    ncc_plot = [r["linear_ncc"] for r in results]
    ax.bar(angles_plot, ncc_plot, color="steelblue", edgecolor="black")
    ax.axhline(y=0.9, color="r", linestyle="--", label="NCC = 0.9")
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Linear NCC")
    ax.set_title("NCC vs Angle")
    ax.legend()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "projection_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print("\n" + "=" * 70)
    print("Writing report")
    print("=" * 70)

    report = f"""# Projection NCC Report — P5-ventral

## Protocol

This is the **original §4.C protocol**: 2D camera projection images with **linear NCC**.

NOT the vertex-based log-NCC that was accidentally substituted later.

## Configuration

- GT position: {GT_POS.tolist()} mm
- Voxel size: {VOXEL_SIZE_MM} mm
- Camera: distance={camera_params["camera_distance_mm"]}mm, fov={camera_params["fov_mm"]}mm, resolution={camera_params["detector_resolution"]}
- Tissue: mua={tissue_params["mua_mm"]:.4f}/mm, mus'={tissue_params["mus_prime_mm"]:.4f}/mm

## Results

| Angle | N_valid | Linear NCC | Scale |
|-------|---------|------------|-------|
"""

    for r in results:
        report += f"| {r['angle']:+d}° | {r['n_valid']} | {r['linear_ncc']:.4f} | {r['scale']:.4e} |\n"

    report += f"""
## Summary

- **Best angle**: {best["angle"]:+d}° with **NCC = {best["linear_ncc"]:.4f}**
- Worst angle: {worst["angle"]:+d}° with NCC = {worst["linear_ncc"]:.4f}
- Angles with NCC ≥ 0.9: **{len(high_ncc_angles)}**

## Key Answers

### 1. Which angles have linear NCC ≥ 0.9?

"""

    if high_ncc_angles:
        for r in high_ncc_angles:
            report += f"- {r['angle']:+d}°: {r['linear_ncc']:.4f}\n"
    else:
        report += "**NONE**\n"

    report += f"""
### 2. Highest NCC vs historical §4.C

- This run: **{best["linear_ncc"]:.4f}** @ {best["angle"]:+d}°
- Historical §4.C: **0.9578** (P5-ventral)

**Match?** {"YES" if abs(best["linear_ncc"] - 0.9578) < 0.05 else "NO"}

### 3. If highest NCC < 0.85

"""

    if best["linear_ncc"] < 0.85:
        report += (
            "**WARNING**: Highest NCC < 0.85. Historical numbers may have issues.\n"
        )
        report += "**STOP**: Do not proceed to downstream experiments.\n"
    else:
        report += f"Highest NCC = {best['linear_ncc']:.4f} ≥ 0.85. Protocol appears correct.\n"

    report += f"""
## Figure

- `projection_comparison.png`: MCX vs Green projections for best/worst angles

## Protocol Change Audit

Git history shows `ec_y10.py` was created in commit:
```
dbb93b7 feat(paper04b): complete MVP pipeline D2.1-M6' + E-series experiments
```

This commit introduced the **vertex-based log-NCC** protocol, replacing the original **projection-based linear NCC** protocol.

The CSV column names remained the same, causing false consistency between incompatible protocols.
"""

    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\n" + "=" * 70)
    print("FINAL ANSWERS")
    print("=" * 70)
    print(f"1. Best linear NCC = {best['linear_ncc']:.4f} @ {best['angle']:+d}°")
    print(f"2. Historical §4.C = 0.9578")
    print(f"3. Match? {'YES' if abs(best['linear_ncc'] - 0.9578) < 0.05 else 'NO'}")
    print(f"4. Angles with NCC ≥ 0.9: {len(high_ncc_angles)}")


if __name__ == "__main__":
    main()
