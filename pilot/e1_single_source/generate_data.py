"""
用解析 PSF 前向模型生成单源多视角 GT 荧光图像

E1 阶段故意使用相同模型做 GT 和前向（inverse crime 条件）
以纯粹验证优化可行性，排除前向模型误差干扰
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

from psf_splatting import (
    GaussianSource,
    PSFSplattingRenderer,
    TissueParams,
    build_turntable_views,
    create_renderer_from_config,
    load_config,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def generate_gt_data(
    config: dict,
    config_id: str,
    device: torch.device = None,
) -> dict:
    """
    生成单配置的多视角 GT 数据

    Returns:
        {
            'images': np.ndarray (N_views, H, W),
            'view_matrices': list of (3,3),
            'surface_normals': list of (3,),
            'source_gt': {center, sigma, alpha},
            'tissue': {mu_a, mu_sp, ...},
            'config_id': str,
        }
    """
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
    source_params = cfg["source"]

    tissue_cfg = config["tissue_params"][tissue_type]
    psf_calib = config["psf_calibration"][tissue_type]

    tissue = TissueParams(
        mu_a=tissue_cfg["mu_a"],
        mu_sp=tissue_cfg["mu_sp"],
        n=tissue_cfg["n"],
    )

    renderer = PSFSplattingRenderer(
        tissue=tissue,
        psf_calibration=psf_calib,
        image_size=config["camera"]["image_size"],
        pixel_size_mm=config["camera"]["pixel_size_mm"],
        vol_size_mm=tuple(config["camera"]["vol_size_mm"]),
    ).to(device)

    view_matrices, surface_normals = build_turntable_views(
        n_views=config["camera"]["n_views"],
        angles_deg=config["camera"]["angles_deg"],
        device=device,
    )

    source = GaussianSource(
        center_init=np.array(source_params["center"], dtype=np.float32),
        sigma_init=source_params["sigma"],
        alpha_init=source_params["alpha"],
    ).to(device)

    with torch.no_grad():
        gt_images = renderer.render_all_views(source, view_matrices, surface_normals)

    gt_images_np = gt_images.cpu().numpy()
    view_matrices_np = [vm.cpu().numpy() for vm in view_matrices]
    surface_normals_np = [sn.cpu().numpy() for sn in surface_normals]

    result = {
        "images": gt_images_np,
        "view_matrices": view_matrices_np,
        "surface_normals": surface_normals_np,
        "source_gt": {
            "center": np.array(source_params["center"], dtype=np.float32).tolist(),
            "sigma": float(source_params["sigma"]),
            "alpha": float(source_params["alpha"]),
        },
        "tissue": {
            "mu_a": tissue.mu_a,
            "mu_sp": tissue.mu_sp,
            "n": tissue.n,
            "D": tissue.D,
            "mu_eff": tissue.mu_eff,
            "z_b": tissue.z_b,
        },
        "psf_calibration": psf_calib,
        "config_id": config_id,
        "tissue_type": tissue_type,
        "camera": config["camera"],
    }

    return result


def save_gt_data(data: dict, output_path: Path) -> None:
    """保存 GT 数据"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        images=data["images"],
        source_center=data["source_gt"]["center"],
        source_sigma=data["source_gt"]["sigma"],
        source_alpha=data["source_gt"]["alpha"],
        tissue_mu_a=data["tissue"]["mu_a"],
        tissue_mu_sp=data["tissue"]["mu_sp"],
        tissue_n=data["tissue"]["n"],
        tissue_D=data["tissue"]["D"],
        tissue_mu_eff=data["tissue"]["mu_eff"],
        tissue_z_b=data["tissue"]["z_b"],
        config_id=data["config_id"],
        tissue_type=data["tissue_type"],
    )

    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source_gt": data["source_gt"],
                "tissue": data["tissue"],
                "config_id": data["config_id"],
                "tissue_type": data["tissue_type"],
                "camera": data["camera"],
                "psf_calibration": data["psf_calibration"],
            },
            f,
            indent=2,
        )

    logger.info(f"Saved GT data: {output_path}")


def run_all_configs(config_path: str, output_dir: str, device: str = "cpu") -> None:
    """运行所有配置的 GT 数据生成"""
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)

    for cfg in config["configs"]:
        config_id = cfg["id"]
        logger.info(f"Generating GT for {config_id}: {cfg['description']}")

        data = generate_gt_data(config, config_id, torch_device)
        save_gt_data(data, output_dir / f"{config_id}_gt.npz")

    logger.info(f"All GT data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate GT multiview data for E1")
    parser.add_argument(
        "--config",
        default="pilot/e1_single_source/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1_single_source/results/gt_data",
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

    run_all_configs(args.config, args.output, args.device)


if __name__ == "__main__":
    main()
