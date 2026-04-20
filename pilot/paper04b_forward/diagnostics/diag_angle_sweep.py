"""R-3: Angle sweep + physics-valid NCC.

P5-ventral 单源 × 10° 步长扫一圈角度，用 MCX 信号强度筛出有效角度子集。
只在源的光子实际出射的角度上评价 NCC。
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.metrics import ncc
from shared.green_surface_projection import render_green_surface_projection
from fmt_simgen.mcx_projection import project_volume_reference

VOXEL_SIZE_MM = 0.4
ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUT = Path("pilot/paper04b_forward/results/projection_fix")
GT_POS = np.array([-0.6, 2.4, -3.8])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R-3: Angle sweep + physics-valid NCC")
    print("=" * 70)

    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))
    fluence = np.load(ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"\nAtlas shape: {atlas.shape}")
    print(f"Fluence shape: {fluence.shape}")
    print(f"GT position: {GT_POS}")

    cam = dict(camera_distance_mm=200.0, fov_mm=50.0, detector_resolution=(256, 256))
    tiss = dict(mua_mm=OPTICAL.mu_a, mus_prime_mm=OPTICAL.mus_p)

    angles = list(range(-180, 180, 10))  # 36 angles
    rows = []

    print(f"\nSweeping {len(angles)} angles...")
    print(f"{'Angle':>7} {'MCX sum':>12} {'MCX peak':>12} {'n_valid':>8} {'NCC':>8}")
    print("-" * 50)

    for ang in angles:
        proj_mcx, _ = project_volume_reference(
            fluence,
            ang,
            voxel_size_mm=VOXEL_SIZE_MM,
            camera_distance=cam["camera_distance_mm"],
            fov_mm=cam["fov_mm"],
            detector_resolution=cam["detector_resolution"],
        )
        proj_green = render_green_surface_projection(
            GT_POS,
            atlas > 0,
            ang,
            tissue_params=tiss,
            voxel_size_mm=VOXEL_SIZE_MM,
            **cam,
        )
        mcx_sum = float(proj_mcx.sum())
        mcx_peak = float(proj_mcx.max())
        thr = mcx_peak * 1e-5 if mcx_peak > 0 else 0.0
        valid = (proj_mcx > thr) & (proj_green > 0)
        n_valid = int(valid.sum())
        if n_valid < 50:
            ncc_val = float("nan")
        else:
            ncc_val = float(ncc(proj_mcx[valid], proj_green[valid]))
        rows.append((ang, mcx_sum, mcx_peak, n_valid, ncc_val))

        ncc_str = f"{ncc_val:.4f}" if not np.isnan(ncc_val) else "N/A"
        print(f"{ang:+4d}° {mcx_sum:12.2e} {mcx_peak:12.2e} {n_valid:8d} {ncc_str:>8}")

    # 写 CSV
    csv_path = OUT / "angle_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle", "mcx_sum", "mcx_peak", "n_valid", "linear_ncc"])
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # 全局 peak
    global_peak = max(r[2] for r in rows)
    validity_threshold = global_peak * 1e-2

    # 有效角度筛选
    valid_angles = [r for r in rows if r[2] >= validity_threshold]
    invalid_angles = [r for r in rows if r[2] < validity_threshold]
    valid_nccs = [r[4] for r in valid_angles if not np.isnan(r[4])]

    print("\n" + "=" * 70)
    print("Valid angle subset (mcx_peak >= global_peak * 1e-2)")
    print("=" * 70)
    print(f"\nGlobal MCX peak: {global_peak:.4e}")
    print(f"Validity threshold: {validity_threshold:.4e}")
    print(f"\nn_valid_angles: {len(valid_angles)} / {len(rows)}")
    print(f"Valid angle list: {[r[0] for r in valid_angles]}")

    if valid_nccs:
        print(f"\nlinear NCC mean: {np.mean(valid_nccs):.4f}")
        print(f"linear NCC min:  {np.min(valid_nccs):.4f}")
        print(f"linear NCC max:  {np.max(valid_nccs):.4f}")
    else:
        print("\nNo valid NCC values!")

    print("\n" + "=" * 70)
    print("Invalid angle subset (liver-blocked direction)")
    print("=" * 70)
    print(f"n_invalid: {len(invalid_angles)} / {len(rows)}")
    print(f"Invalid angle list: {[r[0] for r in invalid_angles]}")

    # 判定
    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)

    if valid_nccs:
        mean_ncc = np.mean(valid_nccs)
        if mean_ncc >= 0.9:
            verdict = "PASS"
            print(f"\n✓ Valid-subset NCC mean = {mean_ncc:.4f} ≥ 0.9")
            print("Forward model OK, protocol fix = filter by physics-valid angles")
        elif mean_ncc >= 0.8:
            verdict = "PARTIAL"
            print(f"\n⚠ Valid-subset NCC mean = {mean_ncc:.4f} (0.8 ≤ NCC < 0.9)")
            print("Forward model partially correct, may need minor adjustments")
        else:
            verdict = "FAIL"
            print(f"\n✗ Valid-subset NCC mean = {mean_ncc:.4f} < 0.8")
            print("Forward model has residual issue beyond heterogeneity")
    else:
        verdict = "NO_DATA"
        mean_ncc = float("nan")
        print("\nNo valid NCC values to evaluate")

    # 图
    fig, ax1 = plt.subplots(figsize=(12, 5))
    angs = [r[0] for r in rows]
    nccs = [r[4] for r in rows]
    peaks = [r[2] for r in rows]

    ax1.axhline(y=0.9, color="green", ls="--", alpha=0.4, label="NCC=0.9")
    ax1.axhline(y=0, color="gray", ls="-", alpha=0.3)
    ax1.plot(angs, nccs, "o-", color="steelblue", label="linear NCC")
    ax1.set_xlabel("Camera angle (°)")
    ax1.set_ylabel("linear NCC", color="steelblue")
    ax1.set_ylim(-1, 1.05)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        angs,
        np.log10(np.array(peaks) + 1e-20),
        "s--",
        color="orange",
        alpha=0.7,
        label="log10 MCX peak",
    )
    ax2.axhline(
        y=np.log10(validity_threshold),
        color="red",
        ls=":",
        alpha=0.5,
        label="validity threshold",
    )
    ax2.set_ylabel("log10(MCX peak)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    for r in invalid_angles:
        ax1.axvspan(r[0] - 5, r[0] + 5, color="red", alpha=0.08)

    ax1.set_title(
        "P5-ventral: NCC and MCX signal vs camera angle\n"
        "(red shaded = physics-invalid angles)"
    )
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    fig_path = OUT / "angle_sweep.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path}")

    # 追加到 REPORT
    valid_angle_list = [r[0] for r in valid_angles]
    invalid_angle_list = [r[0] for r in invalid_angles]

    report = f"""

---

## R-3: Angle sweep + physics-valid NCC

### Full sweep (angle = -180 to +170, step 10°)

- See angle_sweep.csv
- Figure: angle_sweep.png
- Global MCX peak: {global_peak:.4e}
- Validity threshold: global_peak * 1e-2 = {validity_threshold:.4e}

### Valid angle subset

- n_valid_angles: {len(valid_angles)} / {len(rows)}
- Valid angle list: {valid_angle_list}
- linear NCC mean: {np.mean(valid_nccs):.4f}
- linear NCC min:  {np.min(valid_nccs):.4f}
- linear NCC max:  {np.max(valid_nccs):.4f}

### Invalid angle subset (liver-blocked direction)

- n_invalid: {len(invalid_angles)} / {len(rows)}
- Invalid angle list: {invalid_angle_list}
- (NCC on these is physically meaningless; should NOT be included in aggregate)

### Verdict

- **{verdict}**
- Valid-subset NCC mean = {np.mean(valid_nccs):.4f}
"""

    with open(OUT / "REPORT.md", "a") as f:
        f.write(report)

    print(f"\nReport appended to: {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
