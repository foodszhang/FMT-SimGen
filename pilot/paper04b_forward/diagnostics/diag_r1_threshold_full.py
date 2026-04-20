"""R-1 补充：完整的阈值扫描 + 0° 对照。

关键问题：NCC 是被光子噪声拖低，还是 forward model 本身就错？
- 如果 NCC 随阈值升高 → 光子噪声问题，高 SNR 区域 forward model 对
- 如果 NCC 平台 ≈ 0.64 → forward model 本身需要升级

对照：
- +150° (跨体侧成像)：源到成像面距离远，光子信号弱
- +0° (ventral 面成像)：源离成像面最近，光子信号最强
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.metrics import ncc
from shared.green_surface_projection import render_green_surface_projection
from fmt_simgen.mcx_projection import project_volume_reference

VOXEL_SIZE_MM = 0.4
ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/projection_fix")

GT_POS = np.array([-0.6, 2.4, -3.8])


def run_threshold_scan(proj_mcx, proj_green, angle_deg):
    """Run threshold scan for a given projection pair."""
    mcx_max = proj_mcx.max()
    results = []

    for thr_factor in [1e-3, 1e-5, 1e-7]:
        thr = mcx_max * thr_factor
        valid = (proj_mcx > thr) & (proj_green > 0)
        n_valid = int(np.sum(valid))

        if n_valid < 100:
            ncc_val = np.nan
        else:
            ncc_val = ncc(proj_mcx[valid], proj_green[valid])

        results.append(
            {
                "thr_factor": thr_factor,
                "thr": thr,
                "n_valid": n_valid,
                "ncc": ncc_val,
            }
        )

    # 无阈值对照
    valid_all = (proj_mcx > 0) & (proj_green > 0)
    n_valid_all = int(np.sum(valid_all))
    ncc_all = (
        ncc(proj_mcx[valid_all], proj_green[valid_all])
        if n_valid_all >= 100
        else np.nan
    )

    return results, mcx_max, n_valid_all, ncc_all


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R-1 补充：阈值扫描 NCC + 0° 对照")
    print("=" * 70)

    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))
    fluence = np.load(ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    print(f"\nAtlas shape: {atlas.shape}")
    print(f"Fluence shape: {fluence.shape}")
    print(f"GT position: {GT_POS}")

    tissue_params = {
        "mua_mm": OPTICAL.mu_a,
        "mus_prime_mm": OPTICAL.mus_p,
    }

    # 测试两个角度
    angles = [0, 150]
    all_results = {}

    for angle in angles:
        print(f"\n{'=' * 70}")
        print(f"Angle: {angle:+d}°")
        print("=" * 70)

        proj_mcx, _ = project_volume_reference(
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

        results, mcx_max, n_valid_all, ncc_all = run_threshold_scan(
            proj_mcx, proj_green, angle
        )

        all_results[angle] = {
            "results": results,
            "mcx_max": mcx_max,
            "n_valid_all": n_valid_all,
            "ncc_all": ncc_all,
        }

        print(f"\nMCX max: {mcx_max:.4e}")
        print(f"\n{'thr':<15} {'n_valid':>10} {'linear_NCC':>12}")
        print("-" * 40)
        for r in results:
            ncc_str = f"{r['ncc']:.4f}" if not np.isnan(r["ncc"]) else "N/A"
            print(f"max * {r['thr_factor']:<8.0e} {r['n_valid']:>10} {ncc_str:>12}")
        print("-" * 40)
        print(f"{'(no thr)':<15} {n_valid_all:>10} {ncc_all:>12.4f}")

    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)

    print(
        f"\n{'角度':<8} {'max*NCC@1e-3':>15} {'max*NCC@1e-5':>15} {'max*NCC@1e-7':>15} {'NCC(no thr)':>12}"
    )
    print("-" * 70)
    for angle in angles:
        r = all_results[angle]["results"]
        ncc_3 = f"{r[0]['ncc']:.4f}" if not np.isnan(r[0]["ncc"]) else "N/A"
        ncc_5 = f"{r[1]['ncc']:.4f}" if not np.isnan(r[1]["ncc"]) else "N/A"
        ncc_7 = f"{r[2]['ncc']:.4f}" if not np.isnan(r[2]["ncc"]) else "N/A"
        print(
            f"{angle:+d}°     {ncc_3:>15} {ncc_5:>15} {ncc_7:>15} {all_results[angle]['ncc_all']:>12.4f}"
        )

    print("\n" + "=" * 70)
    print("判定")
    print("=" * 70)

    ncc_150_1e3 = all_results[150]["results"][0]["ncc"]
    ncc_150_1e5 = all_results[150]["results"][1]["ncc"]
    ncc_0_1e5 = all_results[0]["results"][1]["ncc"]

    print(f"\n+150° (跨体侧): NCC@1e-5 = {ncc_150_1e5:.4f}")
    print(f"+0° (ventral面): NCC@1e-5 = {ncc_0_1e5:.4f}")

    if ncc_0_1e5 >= 0.9:
        print("\n✓ 0° NCC ≥ 0.9：forward model 正确，+150° 被光子噪声拖低")
        print("  → 建议：重跑 MCX with 1e10 photons")
        verdict = "PHOTON_NOISE"
    elif ncc_0_1e5 >= 0.8:
        print("\n⚠ 0° NCC ≈ 0.8：forward model 部分正确，光子噪声仍有影响")
        print("  → 建议：高光子数 MCX + 检查 forward model 边界条件")
        verdict = "PARTIAL"
    else:
        print("\n✗ 0° NCC < 0.8：forward model 本身问题")
        print("  → 建议：升级 forward model (半无限 Green's / DA-FEM / 边界修正)")
        verdict = "FORWARD_MODEL"

    # 检查 NCC 是否随阈值变化
    ncc_diff = ncc_150_1e3 - ncc_150_1e5 if not np.isnan(ncc_150_1e3) else 0
    if abs(ncc_diff) > 0.1:
        print(f"\n  NCC 随阈值变化 {ncc_diff:+.2f}：高 SNR 区域表现更好")
    else:
        print(f"\n  NCC 平台化 (阈值不敏感)：forward model 问题为主")

    # 追加到 REPORT
    report = f"""

---

## R-1 补充：完整阈值扫描 + 0° 对照

### +150° (跨体侧成像)

| thr | n_valid | linear_NCC |
|-----|---------|------------|
| max × 1e-3 | {all_results[150]["results"][0]["n_valid"]} | {all_results[150]["results"][0]["ncc"]:.4f} |
| max × 1e-5 | {all_results[150]["results"][1]["n_valid"]} | {all_results[150]["results"][1]["ncc"]:.4f} |
| max × 1e-7 | {all_results[150]["results"][2]["n_valid"]} | {all_results[150]["results"][2]["ncc"]:.4f} |
| (no thr) | {all_results[150]["n_valid_all"]} | {all_results[150]["ncc_all"]:.4f} |

### +0° (ventral 面成像，对照)

| thr | n_valid | linear_NCC |
|-----|---------|------------|
| max × 1e-3 | {all_results[0]["results"][0]["n_valid"]} | {all_results[0]["results"][0]["ncc"]:.4f} |
| max × 1e-5 | {all_results[0]["results"][1]["n_valid"]} | {all_results[0]["results"][1]["ncc"]:.4f} |
| max × 1e-7 | {all_results[0]["results"][2]["n_valid"]} | {all_results[0]["results"][2]["ncc"]:.4f} |
| (no thr) | {all_results[0]["n_valid_all"]} | {all_results[0]["ncc_all"]:.4f} |

### 判定

- **Verdict**: {verdict}
- +0° NCC@1e-5 = {ncc_0_1e5:.4f}
- +150° NCC@1e-5 = {ncc_150_1e5:.4f}
"""

    with open(OUTPUT_DIR / "REPORT.md", "a") as f:
        f.write(report)

    print(f"\nReport appended to: {OUTPUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
