"""
E1b 主脚本：MCX GT 生成 + 优化 + 与 E1 对比

流程：
1. 加载/生成 MCX GT
2. 用 E1 相同的优化器 + 解析前向优化
3. 与 E1 结果对比
4. 生成汇总报告
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilot.e1_single_source.psf_splatting import (
    GaussianSource,
    PSFSplattingRenderer,
    TissueParams,
    build_turntable_views,
)
from pilot.e1b_model_mismatch.generate_mcx_gt import generate_mcx_multiview_gt

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_images(images: np.ndarray) -> np.ndarray:
    """归一化图像到 [0, 1]"""
    img_max = images.max()
    if img_max > 0:
        return images / img_max
    return images


def optimize_with_analytical_forward(
    gt_images: torch.Tensor,
    view_matrices: list,
    surface_normals: list,
    tissue_params: dict,
    psf_calibration: dict,
    source_center_gt: np.ndarray,
    n_steps: int = 500,
    lr_center: float = 0.05,
    lr_sigma: float = 0.02,
    lr_alpha: float = 0.02,
    init_noise_mm: float = 1.0,
    image_size: int = 256,
    pixel_size_mm: float = 0.15,
    vol_size_mm: tuple = (30.0, 30.0, 20.0),
    device: torch.device = None,
    verbose: bool = False,
) -> dict:
    """
    用解析 PSF 前向模型优化，GT 是 MCX

    Returns:
        optimization result dict
    """
    if device is None:
        device = torch.device("cpu")

    gt_images = gt_images.to(device)

    np.random.seed(42)
    init_center = source_center_gt + np.random.randn(3) * init_noise_mm
    init_center[2] = max(init_center[2], 0.5)  # 确保深度为正

    init_sigma = 0.5
    init_alpha = 1.0

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

        if verbose and step % 50 == 0:
            logger.info(
                f"Step {step}: loss={loss.item():.6e}, "
                f"center={source.center.detach().cpu().numpy()}, "
                f"sigma={source.sigma.item():.4f}"
            )

    center_final = source.center.detach().cpu().numpy()
    position_error = np.linalg.norm(center_final - source_center_gt)

    recent_loss = loss_history[-50:]
    converged = (max(recent_loss) - min(recent_loss)) / (
        max(recent_loss) + 1e-20
    ) < 0.01

    return {
        "center_final": center_final.tolist(),
        "sigma_final": float(source.sigma.item()),
        "alpha_final": float(source.alpha.item()),
        "position_error_mm": float(position_error),
        "loss_history": loss_history,
        "center_history": [c.tolist() for c in center_history],
        "sigma_history": sigma_history,
        "converged": converged,
        "final_loss": loss_history[-1],
    }


def run_e1b_single(
    config: dict,
    config_id: str,
    output_dir: str,
    device: torch.device = None,
    force_regenerate: bool = False,
) -> dict:
    """运行单个配置的 E1b 实验"""
    if device is None:
        device = torch.device("cpu")

    cfg = None
    for c in config["configs"]:
        if c["id"] == config_id:
            cfg = c
            break

    if cfg is None:
        raise ValueError(f"Config {config_id} not found")

    tissue_type = cfg["tissue_type"]
    source_center_gt = np.array(cfg["source_center"])

    tissue_params = config["tissue_params"][tissue_type]
    psf_calibration = config["psf_calibration"][tissue_type]
    camera_params = config["camera"]
    opt_params = config["optimization"]

    gt_dir = Path(output_dir) / "gt_data" / config_id
    gt_path = gt_dir / f"{config_id}_mcx_gt.npz"

    if not gt_path.exists() or force_regenerate:
        logger.info(f"Generating MCX GT for {config_id}...")
        gt_data = generate_mcx_multiview_gt(
            config, config_id, str(Path(output_dir) / "gt_data")
        )
    else:
        logger.info(f"Loading existing MCX GT: {gt_path}")
        data = np.load(gt_path)
        with open(gt_dir / f"{config_id}_mcx_gt_meta.json") as f:
            meta = json.load(f)
        gt_data = {
            "images": data["images"],
            "view_angles": data["view_angles"].tolist(),
            "source_center_gt": np.array(meta["source_center_gt"]),
            "tissue_type": meta["tissue_type"],
            "tissue_params": tissue_params,
        }

    gt_images = normalize_images(gt_data["images"])
    gt_images = torch.from_numpy(gt_images).float()

    view_matrices, surface_normals = build_turntable_views(
        n_views=camera_params["n_views"],
        angles_deg=camera_params["angles_deg"],
        device=device,
    )

    logger.info(f"Running optimization for {config_id}...")
    logger.info(f"  GT center: {source_center_gt}")

    result = optimize_with_analytical_forward(
        gt_images=gt_images,
        view_matrices=view_matrices,
        surface_normals=surface_normals,
        tissue_params=tissue_params,
        psf_calibration=psf_calibration,
        source_center_gt=source_center_gt,
        n_steps=opt_params["n_steps"],
        lr_center=opt_params["lr_center"],
        lr_sigma=opt_params["lr_sigma"],
        lr_alpha=opt_params["lr_alpha"],
        init_noise_mm=opt_params["init_center_noise_mm"],
        image_size=camera_params["image_size"],
        pixel_size_mm=camera_params["pixel_size_mm"],
        vol_size_mm=tuple(camera_params["vol_size_mm"]),
        device=device,
        verbose=True,
    )

    result["config_id"] = config_id
    result["tissue_type"] = tissue_type
    result["source_center_gt"] = source_center_gt.tolist()

    logger.info(f"Final center: {result['center_final']}")
    logger.info(f"Position error: {result['position_error_mm']:.4f} mm")
    logger.info(f"Sigma final: {result['sigma_final']:.4f} mm")
    logger.info(f"Final loss: {result['final_loss']:.6e}")

    return result


def make_decision(result: dict, thresholds: dict) -> str:
    """根据结果判定 GO/CAUTION/NOGO"""
    pos_err = result["position_error_mm"]
    sigma = result["sigma_final"]

    if (
        pos_err < thresholds["position_error_mm"]
        and sigma < thresholds["sigma_final_max_mm"]
    ):
        return "GO"
    elif pos_err < thresholds["position_error_caution"]:
        return "CAUTION"
    else:
        return "NOGO"


def run_all_e1b(
    config_path: str,
    output_dir: str,
    device: str = "cpu",
    force_regenerate: bool = False,
) -> dict:
    """运行所有配置的 E1b 实验"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch_device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for cfg in config["configs"]:
        config_id = cfg["id"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running E1b for {config_id}: {cfg['description']}")
        logger.info(f"{'=' * 60}")

        result = run_e1b_single(
            config, config_id, output_dir, torch_device, force_regenerate
        )
        result["verdict"] = make_decision(result, config["decision_thresholds"])
        results[config_id] = result

        logger.info(f"Verdict: {result['verdict']}")

    all_go = all(r["verdict"] == "GO" for r in results.values())
    all_nogo = any(r["verdict"] == "NOGO" for r in results.values())

    if all_go:
        overall = "GO"
    elif all_nogo:
        overall = "NOGO"
    else:
        overall = "CAUTION"

    summary = {
        "decision": overall,
        "configs": {
            k: {
                "position_error_mm": v["position_error_mm"],
                "sigma_final": v["sigma_final"],
                "final_loss": v["final_loss"],
                "verdict": v["verdict"],
            }
            for k, v in results.items()
        },
        "full_results": results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({k: convert(v) for k, v in summary.items()}, f, indent=2)

    logger.info(f"\nSummary saved to {summary_path}")
    logger.info(f"\nOverall decision: {overall}")

    if overall == "GO":
        logger.info("Next step: Proceed to E2 (multi-source) without residual network")
    elif overall == "CAUTION":
        logger.info(
            "Next step: Proceed to E2 but consider residual network in parallel"
        )
    else:
        logger.info("Next step: Must add residual network before E2")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run E1b experiments")
    parser.add_argument(
        "--config",
        default="pilot/e1b_model_mismatch/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1b_model_mismatch/results",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regenerate MCX GT even if exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_all_e1b(args.config, args.output, args.device, args.force_regenerate)


if __name__ == "__main__":
    main()
