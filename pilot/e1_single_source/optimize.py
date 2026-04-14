"""
Per-sample 优化：从多视角 2D 图像恢复单个 3D Gaussian 源参数

优化目标：
    L = Σ_v ||Î_v - I_v^gt||² / N_pixels / N_views

优化参数：
    center (3,), log_sigma (1,), log_alpha (1,) — 共 5 个标量
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

from psf_splatting import (
    GaussianSource,
    PSFSplattingRenderer,
    TissueParams,
    build_turntable_views,
    load_config,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def initialize_with_offset(
    gt_center: np.ndarray,
    gt_sigma: float,
    gt_alpha: float,
    offset_scale: float = 0.5,
    seed: int = None,
) -> Tuple[np.ndarray, float, float]:
    """
    初始化参数：在 GT 附近随机偏移

    Args:
        offset_scale: 偏移比例（相对于 GT 值）

    Returns:
        (init_center, init_sigma, init_alpha)
    """
    if seed is not None:
        np.random.seed(seed)

    center_offset = (np.random.rand(3) - 0.5) * 2 * offset_scale * np.abs(gt_center)
    center_offset[2] = np.random.uniform(-0.5, 1.0)

    init_center = gt_center + center_offset
    init_sigma = gt_sigma * (1 + (np.random.rand() - 0.5) * 2 * offset_scale)
    init_sigma = max(init_sigma, 0.1)
    init_alpha = gt_alpha * (1 + (np.random.rand() - 0.5) * 2 * offset_scale)
    init_alpha = max(init_alpha, 0.1)

    return init_center, init_sigma, init_alpha


def optimize_single_source(
    gt_images: torch.Tensor,
    view_matrices: list,
    surface_normals: list,
    tissue_params: dict,
    psf_calibration: dict,
    init_center: np.ndarray,
    init_sigma: float,
    init_alpha: float,
    n_steps: int = 500,
    lr_center: float = 0.05,
    lr_sigma: float = 0.02,
    lr_alpha: float = 0.02,
    image_size: int = 256,
    pixel_size_mm: float = 0.15,
    vol_size_mm: tuple = (30.0, 30.0, 20.0),
    device: torch.device = None,
    verbose: bool = False,
) -> dict:
    """
    Per-sample 优化单个 Gaussian

    Returns:
        {
            'center_final': (3,),
            'sigma_final': float,
            'alpha_final': float,
            'loss_history': list,
            'center_history': list of (3,),
            'sigma_history': list,
            'converged': bool,
            'n_steps_to_converge': int,
        }
    """
    if device is None:
        device = torch.device("cpu")

    gt_images = gt_images.to(device)

    source = GaussianSource(
        center_init=init_center.astype(np.float32),
        sigma_init=init_sigma,
        alpha_init=init_alpha,
    ).to(device)

    tissue = TissueParams(
        mu_a=tissue_params["mu_a"],
        mu_sp=tissue_params["mu_sp"],
        n=tissue_params["n"],
    )

    renderer = PSFSplattingRenderer(
        tissue=tissue,
        psf_calibration=psf_calibration,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        vol_size_mm=vol_size_mm,
    ).to(device)

    view_matrices = [vm.to(device) for vm in view_matrices]
    surface_normals = [sn.to(device) for sn in surface_normals]

    optimizer = optim.Adam(
        [
            {"params": [source.center], "lr": lr_center},
            {"params": [source.log_sigma], "lr": lr_sigma},
            {"params": [source.log_alpha], "lr": lr_alpha},
        ]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    loss_history = []
    center_history = []
    sigma_history = []
    alpha_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        pred_images = renderer.render_all_views(source, view_matrices, surface_normals)

        loss = torch.mean((pred_images - gt_images) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        center_history.append(source.center.detach().cpu().numpy().copy())
        sigma_history.append(source.sigma.item())
        alpha_history.append(source.alpha.item())

        if verbose and step % 50 == 0:
            logger.info(
                f"Step {step}: loss={loss.item():.6e}, "
                f"center={source.center.detach().cpu().numpy()}, "
                f"sigma={source.sigma.item():.4f}, "
                f"alpha={source.alpha.item():.4f}"
            )

    recent_loss = loss_history[-50:]
    loss_range = max(recent_loss) - min(recent_loss)
    converged = loss_range / (max(recent_loss) + 1e-20) < 0.01

    return {
        "center_final": source.center.detach().cpu().numpy(),
        "sigma_final": source.sigma.item(),
        "alpha_final": source.alpha.item(),
        "loss_history": loss_history,
        "center_history": center_history,
        "sigma_history": sigma_history,
        "alpha_history": alpha_history,
        "converged": converged,
        "n_steps_to_converge": len(loss_history),
        "final_loss": loss_history[-1],
    }


def compute_metrics(result: dict, gt_params: dict) -> dict:
    """计算优化结果与 GT 的误差指标"""
    center_pred = np.array(result["center_final"])
    center_gt = np.array(gt_params["center"])

    position_error = np.linalg.norm(center_pred - center_gt)
    sigma_error_ratio = (
        abs(result["sigma_final"] - gt_params["sigma"]) / gt_params["sigma"]
    )
    alpha_error_ratio = (
        abs(result["alpha_final"] - gt_params["alpha"]) / gt_params["alpha"]
    )

    return {
        "position_error_mm": float(position_error),
        "sigma_error_ratio": float(sigma_error_ratio),
        "alpha_error_ratio": float(alpha_error_ratio),
        "center_pred": center_pred.tolist(),
        "center_gt": center_gt.tolist(),
        "sigma_pred": float(result["sigma_final"]),
        "sigma_gt": float(gt_params["sigma"]),
        "alpha_pred": float(result["alpha_final"]),
        "alpha_gt": float(gt_params["alpha"]),
    }


def run_single_config(
    config: dict,
    config_id: str,
    gt_data_path: str,
    output_dir: str,
    device: torch.device = None,
    seed: int = None,
) -> dict:
    """运行单个配置的优化"""
    if device is None:
        device = torch.device("cpu")

    gt_data = np.load(gt_data_path)
    with open(Path(gt_data_path).with_suffix(".json")) as f:
        gt_meta = json.load(f)

    gt_images = torch.from_numpy(gt_data["images"]).float()

    tissue_type = str(gt_data["tissue_type"])
    psf_calibration = config["psf_calibration"][tissue_type]

    _, angles_deg = build_turntable_views(
        n_views=config["camera"]["n_views"],
        angles_deg=config["camera"]["angles_deg"],
        device=device,
    )
    view_matrices, surface_normals = build_turntable_views(
        n_views=config["camera"]["n_views"],
        angles_deg=config["camera"]["angles_deg"],
        device=device,
    )

    gt_params = gt_meta["source_gt"]
    gt_center = np.array(gt_params["center"])
    gt_sigma = gt_params["sigma"]
    gt_alpha = gt_params["alpha"]

    init_center, init_sigma, init_alpha = initialize_with_offset(
        gt_center,
        gt_sigma,
        gt_alpha,
        offset_scale=0.5,
        seed=seed,
    )

    logger.info(f"GT: center={gt_center}, sigma={gt_sigma:.3f}, alpha={gt_alpha:.3f}")
    logger.info(
        f"Init: center={init_center}, sigma={init_sigma:.3f}, alpha={init_alpha:.3f}"
    )

    result = optimize_single_source(
        gt_images=gt_images,
        view_matrices=view_matrices,
        surface_normals=surface_normals,
        tissue_params=gt_meta["tissue"],
        psf_calibration=psf_calibration,
        init_center=init_center,
        init_sigma=init_sigma,
        init_alpha=init_alpha,
        n_steps=config["optimization"]["n_steps"],
        lr_center=config["optimization"]["lr_center"],
        lr_sigma=config["optimization"]["lr_sigma"],
        lr_alpha=config["optimization"]["lr_alpha"],
        image_size=config["camera"]["image_size"],
        pixel_size_mm=config["camera"]["pixel_size_mm"],
        vol_size_mm=tuple(config["camera"]["vol_size_mm"]),
        device=device,
        verbose=True,
    )

    metrics = compute_metrics(result, gt_params)

    logger.info(
        f"Final: center={result['center_final']}, sigma={result['sigma_final']:.3f}, alpha={result['alpha_final']:.3f}"
    )
    logger.info(f"Position error: {metrics['position_error_mm']:.3f} mm")
    logger.info(f"Sigma error: {metrics['sigma_error_ratio'] * 100:.1f}%")
    logger.info(f"Alpha error: {metrics['alpha_error_ratio'] * 100:.1f}%")
    logger.info(f"Converged: {result['converged']}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed_suffix = f"_seed{seed}" if seed is not None else ""
    result_path = output_path / f"{config_id}_optim{seed_suffix}.npz"
    np.savez(
        result_path,
        loss_history=result["loss_history"],
        center_history=np.array(result["center_history"]),
        sigma_history=np.array(result["sigma_history"]),
        alpha_history=np.array(result["alpha_history"]),
        center_final=result["center_final"],
        sigma_final=result["sigma_final"],
        alpha_final=result["alpha_final"],
        converged=result["converged"],
        **metrics,
    )

    return {**result, **metrics}


def make_decision(metrics: dict, thresholds: dict) -> str:
    """根据指标判定 GO/CAUTION/NOGO"""
    pos_ok = metrics["position_error_mm"] < thresholds["position_error_mm"]
    sigma_ok = metrics["sigma_error_ratio"] < thresholds["sigma_error_ratio"]
    alpha_ok = metrics["alpha_error_ratio"] < thresholds["alpha_error_ratio"]

    if pos_ok and sigma_ok and alpha_ok:
        return "GO"
    else:
        return "CAUTION"


def run_all_configs(
    config_path: str,
    gt_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> dict:
    """运行所有配置的优化"""
    config = load_config(config_path)
    torch_device = torch.device(device)

    results = {}

    for cfg in config["configs"]:
        config_id = cfg["id"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running {config_id}: {cfg['description']}")
        logger.info(f"{'=' * 60}")

        gt_path = Path(gt_dir) / f"{config_id}_gt.npz"

        if "n_random_seeds" in cfg:
            seed_results = []
            for seed in range(cfg["n_random_seeds"]):
                logger.info(f"\n--- Seed {seed} ---")
                result = run_single_config(
                    config, config_id, str(gt_path), output_dir, torch_device, seed=seed
                )
                result["seed"] = seed
                result["verdict"] = make_decision(result, config["decision_thresholds"])
                seed_results.append(result)
            results[config_id] = {
                "seeds": seed_results,
                "verdict": "GO"
                if all(r["verdict"] == "GO" for r in seed_results)
                else "CAUTION",
            }
        else:
            result = run_single_config(
                config, config_id, str(gt_path), output_dir, torch_device, seed=None
            )
            result["verdict"] = make_decision(result, config["decision_thresholds"])
            results[config_id] = result

    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    logger.info(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize single source for E1")
    parser.add_argument(
        "--config",
        default="pilot/e1_single_source/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--gt-dir",
        default="pilot/e1_single_source/results/gt_data",
        help="GT data directory",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1_single_source/results/optimization",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_all_configs(args.config, args.gt_dir, args.output, args.device)


if __name__ == "__main__":
    main()
