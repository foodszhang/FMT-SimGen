"""三路对比 + 可视化 + go/no-go 自动判定

对比 MCX (ground truth) vs 解析 Green's function vs Gaussian fit
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """计算归一化互相关系数 (NCC)"""
    a_n = a - a.mean()
    b_n = b - b.mean()
    denom = np.linalg.norm(a_n) * np.linalg.norm(b_n)
    return float(np.dot(a_n, b_n) / denom) if denom > 0 else 0.0


def compute_fwhm(rho: np.ndarray, intensity: np.ndarray) -> float:
    """计算半高全宽 (FWHM)"""
    half = intensity[0] / 2
    idx = np.searchsorted(-intensity, -half)
    if idx >= len(rho):
        return 2 * rho[-1]
    return 2 * rho[idx]


def compute_rmse_normalized(pred: np.ndarray, target: np.ndarray) -> float:
    """计算归一化 RMSE (相对于目标峰值)"""
    peak = target.max() if target.max() > 0 else 1.0
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return float(rmse / peak)


def compute_metrics(
    rho: np.ndarray,
    I_mcx: np.ndarray,
    I_semi: np.ndarray,
    I_gauss: np.ndarray,
) -> Dict:
    """计算对比指标

    Args:
        rho: 径向距离数组 (mm)
        I_mcx: MCX ground truth
        I_semi: 解析 Green's function
        I_gauss: Gaussian fit

    Returns:
        dict with all comparison metrics
    """
    # 归一化到峰值=1
    I_mcx_norm = I_mcx / I_mcx.max() if I_mcx.max() > 0 else I_mcx
    I_semi_norm = I_semi / I_semi.max() if I_semi.max() > 0 else I_semi
    I_gauss_norm = I_gauss / I_gauss.max() if I_gauss.max() > 0 else I_gauss

    # NCC
    ncc_mcx_semi = compute_ncc(I_mcx_norm, I_semi_norm)
    ncc_mcx_gauss = compute_ncc(I_mcx_norm, I_gauss_norm)
    ncc_semi_gauss = compute_ncc(I_semi_norm, I_gauss_norm)

    # FWHM
    fwhm_mcx = compute_fwhm(rho, I_mcx_norm)
    fwhm_semi = compute_fwhm(rho, I_semi_norm)
    fwhm_gauss = compute_fwhm(rho, I_gauss_norm)

    # FWHM ratio
    fwhm_ratio_mcx_semi = fwhm_semi / fwhm_mcx if fwhm_mcx > 0 else 0.0
    fwhm_ratio_mcx_gauss = fwhm_gauss / fwhm_mcx if fwhm_mcx > 0 else 0.0

    # Peak ratio
    peak_mcx = I_mcx.max()
    peak_semi = I_semi.max()
    peak_gauss = I_gauss.max()
    peak_ratio_mcx_semi = peak_semi / peak_mcx if peak_mcx > 0 else 0.0
    peak_ratio_mcx_gauss = peak_gauss / peak_mcx if peak_mcx > 0 else 0.0

    # RMSE
    rmse_semi = compute_rmse_normalized(I_semi_norm, I_mcx_norm)
    rmse_gauss = compute_rmse_normalized(I_gauss_norm, I_mcx_norm)

    return {
        "ncc_mcx_semi": float(ncc_mcx_semi),
        "ncc_mcx_gauss": float(ncc_mcx_gauss),
        "ncc_semi_gauss": float(ncc_semi_gauss),
        "fwhm_mcx": float(fwhm_mcx),
        "fwhm_semi": float(fwhm_semi),
        "fwhm_gauss": float(fwhm_gauss),
        "fwhm_ratio_mcx_semi": float(fwhm_ratio_mcx_semi),
        "fwhm_ratio_mcx_gauss": float(fwhm_ratio_mcx_gauss),
        "peak_ratio_mcx_semi": float(peak_ratio_mcx_semi),
        "peak_ratio_mcx_gauss": float(peak_ratio_mcx_gauss),
        "rmse_semi": float(rmse_semi),
        "rmse_gauss": float(rmse_gauss),
    }


def load_all_results(profiles_dir: str) -> Dict:
    """加载所有配置的解析 + MCX 结果

    Args:
        profiles_dir: 包含 *_analytic.npz 和 *_mcx.npz 的目录

    Returns:
        dict: config_id -> {rho, I_mcx, I_semi, I_gauss, ...}
    """
    profiles_dir = Path(profiles_dir)
    results = {}

    # 找所有配置 ID
    analytic_files = sorted(profiles_dir.glob("*_analytic.npz"))
    for analytic_path in analytic_files:
        config_id = analytic_path.stem.replace("_analytic", "")
        mcx_path = profiles_dir / f"{config_id}_mcx.npz"

        if not mcx_path.exists():
            logger.warning(f"MCX result not found for {config_id}")
            continue

        # 加载数据
        analytic = np.load(analytic_path)
        mcx = np.load(mcx_path)

        results[config_id] = {
            "rho_analytic": analytic["rho"],
            "I_semi": analytic["I_semi"],
            "I_gauss": analytic["I_gauss"],
            "rho_mcx": mcx["rho"],
            "I_mcx": mcx["intensity"],
            "depth_mm": float(analytic.get("depth_mm", mcx.get("depth_mm", 0))),
            "tissue_type": str(
                analytic.get("tissue_type", mcx.get("tissue_type", "unknown"))
            ),
        }

    return results


def plot_comparison(
    config_id: str,
    rho: np.ndarray,
    I_mcx: np.ndarray,
    I_semi: np.ndarray,
    I_gauss: np.ndarray,
    metrics: Dict,
    save_path: str,
) -> None:
    """绘制单配置对比图

    上：三条曲线叠加（MCX=黑, Green=蓝, Gauss=红虚线）
    下：相对误差 (I_analytic - I_mcx) / I_mcx
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 归一化
    I_mcx_norm = I_mcx / I_mcx.max() if I_mcx.max() > 0 else I_mcx
    I_semi_norm = I_semi / I_semi.max() if I_semi.max() > 0 else I_semi
    I_gauss_norm = I_gauss / I_gauss.max() if I_gauss.max() > 0 else I_gauss

    # 上图：曲线对比
    ax = axes[0]
    ax.plot(rho, I_mcx_norm, "k-", linewidth=2, label="MCX (Ground Truth)")
    ax.plot(rho, I_semi_norm, "b-", linewidth=1.5, label="Green's function")
    ax.plot(rho, I_gauss_norm, "r--", linewidth=1.5, label="Gaussian fit")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(
        f"{config_id}: {metrics.get('tissue_type', '')} @ {metrics.get('depth_mm', 0)}mm | "
        f"NCC(MCX, Gauss)={metrics['ncc_mcx_gauss']:.3f}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)

    # 下图：相对误差
    ax = axes[1]
    # 避免除以零
    I_mcx_safe = np.where(I_mcx_norm > 1e-10, I_mcx_norm, np.nan)
    err_semi = (I_semi_norm - I_mcx_norm) / I_mcx_safe
    err_gauss = (I_gauss_norm - I_mcx_norm) / I_mcx_safe

    ax.plot(rho, err_semi, "b-", linewidth=1.5, label="Green's - MCX")
    ax.plot(rho, err_gauss, "r--", linewidth=1.5, label="Gaussian - MCX")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Radial Distance ρ (mm)")
    ax.set_ylabel("Relative Error")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {save_path}")


def plot_summary_table(all_metrics: Dict, save_path: str) -> None:
    """绘制汇总表格热力图

    行=配置, 列=指标, 颜色=好/中/差
    """
    config_ids = sorted(all_metrics.keys())
    metrics_names = [
        "ncc_mcx_semi",
        "ncc_mcx_gauss",
        "fwhm_ratio_mcx_semi",
        "fwhm_ratio_mcx_gauss",
        "rmse_semi",
        "rmse_gauss",
    ]

    # 构建矩阵
    data = np.zeros((len(config_ids), len(metrics_names)))
    for i, cid in enumerate(config_ids):
        for j, mname in enumerate(metrics_names):
            data[i, j] = all_metrics[cid].get(mname, 0)

    # NCC: 越高越好 (0.7-1.0)
    # FWHM ratio: 越接近 1 越好 (0.7-1.3)
    # RMSE: 越低越好 (0-0.3)

    # 归一化到 0-1 (1=好, 0=差)
    norm_data = np.zeros_like(data)
    for i, cid in enumerate(config_ids):
        for j, mname in enumerate(metrics_names):
            val = data[i, j]
            if "ncc" in mname:
                norm_data[i, j] = (val - 0.7) / 0.3  # 0.7->0, 1.0->1
            elif "fwhm_ratio" in mname:
                norm_data[i, j] = 1 - abs(val - 1) / 0.3  # 1->1, 0.7/1.3->0
            elif "rmse" in mname:
                norm_data[i, j] = 1 - val / 0.3  # 0->1, 0.3->0

    norm_data = np.clip(norm_data, 0, 1)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_xticklabels(metrics_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(config_ids)))
    ax.set_yticklabels(config_ids)

    # 添加数值标注
    for i in range(len(config_ids)):
        for j in range(len(metrics_names)):
            text_color = "white" if norm_data[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.set_title("PSF Validation Metrics Summary (Green=Good, Red=Bad)")
    plt.colorbar(im, ax=ax, label="Quality Score (0=Bad, 1=Good)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary figure: {save_path}")


def go_nogo_decision(all_metrics: Dict) -> Dict:
    """自动 go/no-go 判定

    规则（仅对匀质配置 C01-C06 判定）：
    - GO:      所有配置 NCC(MCX, Gauss) > 0.90 且 FWHM ratio 在 [0.8, 1.2]
    - CAUTION: 任一配置 NCC 在 [0.70, 0.90] 或 FWHM ratio 在 [0.7, 0.8]∪[1.2, 1.3]
    - NOGO:    任一配置 NCC < 0.70 或 FWHM ratio 在 [0, 0.7]∪[1.3, +∞]

    对双层配置 C07-C09：仅记录，不参与 go/no-go

    Returns:
        {'decision': 'GO'|'CAUTION'|'NOGO', 'details': {...}, 'recommendation': str}
    """
    homogeneous_ids = ["C01", "C02", "C03", "C04", "C05", "C06"]
    bilayer_ids = ["C07", "C08", "C09"]

    details = {}
    verdicts = []

    for config_id, metrics in all_metrics.items():
        ncc = metrics.get("ncc_mcx_gauss", 0)
        fwhm_ratio = metrics.get("fwhm_ratio_mcx_gauss", 0)

        if config_id in homogeneous_ids:
            # 匀质配置：参与判定
            if ncc > 0.90 and 0.8 <= fwhm_ratio <= 1.2:
                verdict = "GO"
            elif ncc >= 0.70 and 0.7 <= fwhm_ratio <= 1.3:
                verdict = "CAUTION"
            else:
                verdict = "NOGO"
            verdicts.append(verdict)
        else:
            # 双层配置：仅记录
            verdict = "INFO"

        details[config_id] = {
            "ncc_mcx_gauss": float(ncc),
            "fwhm_ratio": float(fwhm_ratio),
            "verdict": verdict,
        }

    # 综合判定
    if "NOGO" in verdicts:
        decision = "NOGO"
        recommendation = (
            "主线方案不可行。Gaussian PSF 与 MCX 差异过大。"
            "建议：切换到 MCX 预计算 PSF 查找表方案。"
        )
    elif "CAUTION" in verdicts:
        decision = "CAUTION"
        recommendation = (
            "Gaussian PSF 近似在部分配置下误差较大。"
            "建议：启用 PSF + 轻量残差网络补偿方案。"
        )
    else:
        decision = "GO"
        recommendation = (
            "主线方案可行。Gaussian PSF 近似在 1.5-5mm 深度下与 MCX 高度一致。"
            "可直接进入 E1（单源 3DGS 优化验证）。"
        )

    return {
        "decision": decision,
        "details": details,
        "recommendation": recommendation,
    }


def run_comparison(profiles_dir: str, output_dir: str) -> Tuple[Dict, Dict]:
    """运行完整对比分析

    Args:
        profiles_dir: 包含 *_analytic.npz 和 *_mcx.npz 的目录
        output_dir: 输出目录

    Returns:
        (all_metrics, decision_result)
    """
    profiles_dir = Path(profiles_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有结果
    results = load_all_results(profiles_dir)

    if not results:
        logger.error("No valid results found for comparison")
        return {}, {}

    all_metrics = {}

    # 对每个配置进行对比
    for config_id, data in results.items():
        logger.info(f"Processing {config_id}")

        # 插值 MCX 到解析的 rho 网格（如果需要）
        rho_analytic = data["rho_analytic"]
        rho_mcx = data["rho_mcx"]
        I_mcx = data["I_mcx"]

        if not np.allclose(rho_analytic, rho_mcx):
            I_mcx_interp = np.interp(rho_analytic, rho_mcx, I_mcx)
        else:
            I_mcx_interp = I_mcx

        rho = rho_analytic
        I_semi = data["I_semi"]
        I_gauss = data["I_gauss"]

        # 计算指标
        metrics = compute_metrics(rho, I_mcx_interp, I_semi, I_gauss)
        metrics["depth_mm"] = data["depth_mm"]
        metrics["tissue_type"] = data["tissue_type"]
        all_metrics[config_id] = metrics

        # 绘制对比图
        plot_comparison(
            config_id=config_id,
            rho=rho,
            I_mcx=I_mcx_interp,
            I_semi=I_semi,
            I_gauss=I_gauss,
            metrics=metrics,
            save_path=figures_dir / f"{config_id}_comparison.png",
        )

    # 绘制汇总表格
    plot_summary_table(all_metrics, figures_dir / "summary_table.png")

    # Go/No-Go 判定
    decision_result = go_nogo_decision(all_metrics)

    # 保存汇总
    summary = {
        "decision": decision_result["decision"],
        "recommendation": decision_result["recommendation"],
        "configs": decision_result["details"],
        "metrics": all_metrics,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # 打印决策
    logger.info("=" * 60)
    logger.info(f"DECISION: {decision_result['decision']}")
    logger.info(f"RECOMMENDATION: {decision_result['recommendation']}")
    logger.info("=" * 60)

    return all_metrics, decision_result


def main():
    parser = argparse.ArgumentParser(description="Compare MCX vs Analytic PSF")
    parser.add_argument(
        "--profiles_dir",
        default="pilot/e0_psf_validation/results/profiles/",
        help="Directory with analytic and MCX profiles",
    )
    parser.add_argument(
        "--output",
        default="pilot/e0_psf_validation/results/",
        help="Output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_comparison(args.profiles_dir, args.output)
    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()
