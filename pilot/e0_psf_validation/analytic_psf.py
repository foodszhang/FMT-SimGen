"""解析 CW 扩散 Green's function + Gaussian fit

基于 García et al. JBO 2026, Eq.(7) + Eq.(14)
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class TissueParams:
    """组织光学参数"""

    mu_a: float  # 吸收系数 (mm^-1)
    mu_sp: float  # 约化散射系数 (mm^-1)
    n: float = 1.37  # 折射率

    @property
    def D(self) -> float:
        """扩散系数 (mm)"""
        return 1.0 / (3.0 * (self.mu_a + self.mu_sp))

    @property
    def mu_eff(self) -> float:
        """有效衰减系数 (mm^-1)"""
        return np.sqrt(self.mu_a / self.D)

    @property
    def z_b(self) -> float:
        """外推边界距离 (mm)"""
        # R_eff for n=1.37 (tissue/air interface)
        R_eff = 0.493
        A = (1 + R_eff) / (1 - R_eff)  # ≈ 2.94
        return 2 * A * self.D

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "mu_a": self.mu_a,
            "mu_sp": self.mu_sp,
            "n": self.n,
            "D": self.D,
            "mu_eff": self.mu_eff,
            "z_b": self.z_b,
        }


def green_infinite(rho: np.ndarray, d: float, tissue: TissueParams) -> np.ndarray:
    """无限介质 CW Green's function (García Eq.7)

    Args:
        rho: 横向距离数组 (mm), shape (N,)
        d: 源深度 (mm)
        tissue: 组织光学参数

    Returns:
        surface intensity array, shape (N,)
    """
    r = np.sqrt(rho**2 + d**2)
    G = np.exp(-tissue.mu_eff * r) / (4 * np.pi * tissue.D * r)
    return G


def green_semi_infinite(rho: np.ndarray, d: float, tissue: TissueParams) -> np.ndarray:
    """半无限介质 CW Green's function (García Eq.14)
    含镜像源，外推边界条件 (EBC)

    Args:
        rho: 横向距离数组 (mm), shape (N,)
        d: 源深度 (mm), 从表面到源的距离
        tissue: 组织光学参数

    Returns:
        surface intensity at z=0, shape (N,)
    """
    z_b = tissue.z_b
    r1 = np.sqrt(rho**2 + d**2)
    r2 = np.sqrt(rho**2 + (d + 2 * z_b) ** 2)

    G = (np.exp(-tissue.mu_eff * r1) / r1 - np.exp(-tissue.mu_eff * r2) / r2) / (
        4 * np.pi * tissue.D
    )
    return np.maximum(G, 0)  # 物理约束：非负


def fit_gaussian(rho: np.ndarray, intensity: np.ndarray) -> Tuple[float, float]:
    """对径向强度分布做 Gaussian fit: I(ρ) = T * exp(-ρ²/(2σ²))

    Args:
        rho: 横向距离数组 (mm)
        intensity: 强度数组

    Returns:
        (sigma_psf, T_peak): PSF 宽度 (mm) 和峰值衰减
    """

    def gaussian(r, T, sigma):
        return T * np.exp(-(r**2) / (2 * sigma**2))

    T0 = intensity[0]  # ρ=0 处的值
    # 从 FWHM 估计初始 sigma
    half_max = T0 / 2
    # 找到降到半高的位置（从高到低搜索）
    idx = np.searchsorted(-intensity, -half_max)
    sigma0 = rho[min(idx, len(rho) - 1)] / np.sqrt(2 * np.log(2))
    sigma0 = max(sigma0, 0.1)  # 保底

    try:
        popt, _ = curve_fit(
            gaussian, rho, intensity, p0=[T0, sigma0], bounds=(0, np.inf)
        )
        return popt[1], popt[0]  # sigma, T
    except RuntimeError:
        logger.warning("Gaussian fit failed, returning initial estimate")
        return sigma0, T0


def compute_fwhm(rho: np.ndarray, intensity: np.ndarray) -> float:
    """计算半高全宽 (FWHM)"""
    half = intensity[0] / 2
    idx = np.searchsorted(-intensity, -half)
    if idx >= len(rho):
        return 2 * rho[-1]
    return 2 * rho[idx]


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """计算归一化互相关系数 (NCC)"""
    a_n = a - a.mean()
    b_n = b - b.mean()
    denom = np.linalg.norm(a_n) * np.linalg.norm(b_n)
    return float(np.dot(a_n, b_n) / denom) if denom > 0 else 0.0


def compute_psf_profile(
    d: float,
    tissue: TissueParams,
    rho_max: float = 15.0,
    n_points: int = 500,
) -> dict:
    """计算单个 (depth, tissue) 配置的完整 PSF 分析

    Args:
        d: 源深度 (mm)
        tissue: 组织光学参数
        rho_max: 最大横向距离 (mm)
        n_points: 采样点数

    Returns:
        dict with keys:
            rho, I_inf, I_semi, I_gauss,
            sigma_psf, T_peak, fwhm_semi, fwhm_gauss,
            ncc_inf_semi, ncc_semi_gauss
    """
    rho = np.linspace(0, rho_max, n_points)

    I_inf = green_infinite(rho, d, tissue)
    I_semi = green_semi_infinite(rho, d, tissue)

    sigma_psf, T_peak = fit_gaussian(rho, I_semi)
    I_gauss = T_peak * np.exp(-(rho**2) / (2 * sigma_psf**2))

    # FWHM
    fwhm_semi = compute_fwhm(rho, I_semi)
    fwhm_gauss = compute_fwhm(rho, I_gauss)

    # NCC
    ncc_inf_semi = compute_ncc(I_inf, I_semi)
    ncc_semi_gauss = compute_ncc(I_semi, I_gauss)

    return {
        "rho": rho,
        "I_inf": I_inf,
        "I_semi": I_semi,
        "I_gauss": I_gauss,
        "sigma_psf": float(sigma_psf),
        "T_peak": float(T_peak),
        "fwhm_semi": float(fwhm_semi),
        "fwhm_gauss": float(fwhm_gauss),
        "ncc_inf_semi": float(ncc_inf_semi),
        "ncc_semi_gauss": float(ncc_semi_gauss),
    }


def load_experiment_config(config_path: str) -> dict:
    """加载实验配置"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_all_analytic(config_path: str, output_dir: str) -> dict:
    """运行所有配置的解析 PSF 计算

    Args:
        config_path: 实验矩阵配置文件路径
        output_dir: 输出目录

    Returns:
        所有配置的结果字典
    """
    config = load_experiment_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for cfg in config["configs"]:
        config_id = cfg["id"]
        depth = cfg["depth_mm"]
        tissue_type = cfg["tissue_type"]

        logger.info(
            f"Computing analytic PSF for {config_id}: {tissue_type} @ {depth}mm"
        )

        # 创建组织参数
        if tissue_type in ["muscle", "soft_tissue"]:
            tissue = TissueParams(
                mu_a=config["tissue_params"]["muscle"]["mu_a"],
                mu_sp=config["tissue_params"]["muscle"]["mu_sp"],
                n=config["tissue_params"]["muscle"]["n"],
            )
        elif tissue_type == "liver":
            tissue = TissueParams(
                mu_a=config["tissue_params"]["liver"]["mu_a"],
                mu_sp=config["tissue_params"]["liver"]["mu_sp"],
                n=config["tissue_params"]["liver"]["n"],
            )
        elif tissue_type == "bilayer":
            # 对于双层，使用等效参数（平均）作为近似
            # 真正的异质介质需要 MCX 仿真
            muscle = config["tissue_params"]["muscle"]
            liver = config["tissue_params"]["liver"]
            # 简单加权平均（skin 1mm + liver 主体）
            tissue = TissueParams(
                mu_a=(muscle["mu_a"] * 1.0 + liver["mu_a"] * 9.0) / 10.0,
                mu_sp=(muscle["mu_sp"] * 1.0 + liver["mu_sp"] * 9.0) / 10.0,
                n=1.37,
            )
        else:
            raise ValueError(f"Unknown tissue type: {tissue_type}")

        # 计算 PSF
        profile = compute_psf_profile(
            d=depth,
            tissue=tissue,
            rho_max=config["simulation_params"]["rho_max_mm"],
            n_points=config["simulation_params"]["n_points"],
        )

        # 保存结果
        result = {
            "config_id": config_id,
            "depth_mm": depth,
            "tissue_type": tissue_type,
            "tissue_params": tissue.to_dict(),
            "sigma_psf": profile["sigma_psf"],
            "T_peak": profile["T_peak"],
            "fwhm_semi": profile["fwhm_semi"],
            "fwhm_gauss": profile["fwhm_gauss"],
            "ncc_inf_semi": profile["ncc_inf_semi"],
            "ncc_semi_gauss": profile["ncc_semi_gauss"],
        }
        results[config_id] = result

        # 保存 npz（包含完整曲线）
        npz_path = output_dir / f"{config_id}_analytic.npz"
        np.savez(
            npz_path,
            rho=profile["rho"],
            I_inf=profile["I_inf"],
            I_semi=profile["I_semi"],
            I_gauss=profile["I_gauss"],
            **{k: v for k, v in result.items() if k not in ["config_id"]},
        )
        logger.info(f"  Saved: {npz_path}")
        logger.info(
            f"  sigma_psf={profile['sigma_psf']:.3f}mm, "
            f"fwhm_semi={profile['fwhm_semi']:.3f}mm, "
            f"ncc_semi_gauss={profile['ncc_semi_gauss']:.4f}"
        )

    # 保存汇总 JSON
    summary_path = output_dir / "analytic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute analytic PSF profiles")
    parser.add_argument(
        "--config",
        default="pilot/e0_psf_validation/config.yaml",
        help="Experiment matrix config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e0_psf_validation/results/profiles/",
        help="Output directory for profiles",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_all_analytic(args.config, args.output)
    logger.info("Analytic PSF computation complete!")


if __name__ == "__main__":
    main()
