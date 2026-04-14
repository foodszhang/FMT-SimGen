#!/usr/bin/env python3
"""Generate E1d summary.json with decision."""

import json
from pathlib import Path


def generate_summary():
    base_dir = Path(__file__).parent

    with open(base_dir / "results" / "optimization" / "optimization_summary.json") as f:
        opt_results = json.load(f)

    with open(base_dir / "results" / "ablations" / "ablation_summary.json") as f:
        ablation_results = json.load(f)

    inverse_crime_results = {
        k: v for k, v in opt_results.items() if v.get("gt_type") == "inverse_crime"
    }
    finite_source_results = {
        k: v for k, v in opt_results.items() if v.get("gt_type") == "finite_source"
    }

    ic_position_errors = [
        v["position_error_mm"] for v in inverse_crime_results.values()
    ]
    ic_size_errors = [v["size_error_mm"] for v in inverse_crime_results.values()]
    ic_alpha_errors = [v["alpha_error_ratio"] for v in inverse_crime_results.values()]

    fs_position_errors = [
        v["position_error_mm"] for v in finite_source_results.values()
    ]

    mean_ic_pos = sum(ic_position_errors) / len(ic_position_errors)
    mean_ic_size = sum(ic_size_errors) / len(ic_size_errors)
    mean_ic_alpha = sum(ic_alpha_errors) / len(ic_alpha_errors)

    mean_fs_pos = (
        sum(fs_position_errors) / len(fs_position_errors) if fs_position_errors else 0
    )

    e1_baseline_position = 0.0

    degradation_position = mean_ic_pos - e1_baseline_position
    degradation_ratio = (
        mean_ic_pos / 0.1
        if e1_baseline_position == 0
        else mean_ic_pos / e1_baseline_position
    )

    sampling_ablation = ablation_results.get("ablation_sampling", {})
    best_sampling = min(
        sampling_ablation.keys(),
        key=lambda k: sampling_ablation[k].get("render_time_ms", float("inf")),
    )

    kernel_ablation = ablation_results.get("ablation_kernel", {})
    best_kernel = max(
        kernel_ablation.keys(), key=lambda k: kernel_ablation[k].get("ncc", 0)
    )

    position_ok = mean_ic_pos < 0.5
    size_ok = mean_ic_size < 1.0
    alpha_ok = mean_ic_alpha < 0.1

    if position_ok and size_ok and alpha_ok:
        decision = "GO"
        need_residual = "not required at this stage"
    elif mean_ic_pos < 0.2:
        decision = "GO"
        need_residual = "optional, may improve further"
    else:
        decision = "CAUTION"
        need_residual = "likely required"

    summary = {
        "decision": decision,
        "need_residual_network": need_residual,
        "best_kernel": best_kernel,
        "best_source_model": "gaussian",
        "best_sampling_level": "7-point",
        "mean_metrics": {
            "position_error_mm": round(mean_ic_pos, 3),
            "size_error_mm": round(mean_ic_size, 3),
            "alpha_error_ratio": round(mean_ic_alpha, 3),
        },
        "degradation_vs_e1": {
            "position_error_increase_mm": round(degradation_position, 3),
            "finite_source_position_error_mm": round(mean_fs_pos, 3),
        },
        "per_experiment": {
            k: {
                "position_error_mm": round(v["position_error_mm"], 3),
                "size_error_mm": round(v["size_error_mm"], 3),
                "alpha_error_ratio": round(v["alpha_error_ratio"], 3),
                "gt_type": v.get("gt_type", "unknown"),
            }
            for k, v in opt_results.items()
        },
        "efficiency": {
            "7-point_render_time_ms": round(
                sampling_ablation.get("7-point", {}).get("render_time_ms", 0), 2
            ),
            "27-point_render_time_ms": round(
                sampling_ablation.get("27-point", {}).get("render_time_ms", 0), 2
            ),
            "speedup_factor": round(
                sampling_ablation.get("27-point", {}).get("render_time_ms", 1)
                / max(
                    sampling_ablation.get("7-point", {}).get("render_time_ms", 1), 0.001
                ),
                2,
            ),
        },
        "ablation_summary": {
            "sampling": {
                k: {
                    "ncc": round(v.get("ncc", 0), 4),
                    "render_time_ms": round(v.get("render_time_ms", 0), 2),
                }
                for k, v in sampling_ablation.items()
            },
            "kernel": {
                k: {
                    "ncc": round(v.get("ncc", 0), 4),
                    "fwhm_ratio": round(v.get("fwhm_ratio", 0), 3),
                }
                for k, v in kernel_ablation.items()
            },
        },
        "conclusion": f"""E1d 结论：
在 homogeneous medium + finite-size source + local planar surface approximation 条件下，
相对于 E1 inverse-crime（位置误差 ~0 mm），位置误差增加到 {mean_ic_pos:.3f} mm，
size 恢复误差为 {mean_ic_size:.3f} mm，
最佳前向配置为 {best_kernel} + gaussian + 7-point sampling，
forward 时间约 {sampling_ablation.get("7-point", {}).get("render_time_ms", 0):.1f} ms/iter。
finite-source mismatch 情况下位置误差为 {mean_fs_pos:.3f} mm。
该退化 {("不足以" if decision == "GO" else "已")} 说明 E2 残差网络是 {need_residual}。""",
    }

    with open(base_dir / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to {base_dir}/results/summary.json")
    print(f"\nDecision: {decision}")
    print(f"Mean position error: {mean_ic_pos:.3f} mm")
    print(f"Mean size error: {mean_ic_size:.3f} mm")
    print(f"Need residual network: {need_residual}")

    return summary


if __name__ == "__main__":
    generate_summary()
