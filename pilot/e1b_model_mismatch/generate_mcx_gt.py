"""
用 MCX 生成多视角 GT 荧光图像（使用 project_volume_reference）

流程：
1. MCX 跑一次：点源在固定位置
2. 加载 fluence 体积 [X×Y×Z]
3. 使用 FMT-SimGen 的 project_volume_reference 生成多视角投影
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilot.e0_psf_validation.mcx_point_source import (
    create_homogeneous_volume,
    generate_mcx_config_json,
    run_mcx_simulation,
    load_mcx_fluence,
)
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def generate_mcx_multiview_gt(
    config: dict,
    config_id: str,
    output_dir: str,
) -> dict:
    """
    生成单个配置的 MCX 多视角 GT 数据（使用 project_volume_reference）

    Returns:
        dict with images, view_angles, source_center_gt, etc.
    """
    cfg = None
    for c in config["configs"]:
        if c["id"] == config_id:
            cfg = c
            break

    if cfg is None:
        raise ValueError(f"Config {config_id} not found")

    tissue_type = cfg["tissue_type"]
    source_center = np.array(cfg["source_center"])

    tissue_params = config["tissue_params"][tissue_type]
    mcx_params = config["mcx"]
    camera_params = config["camera"]

    view_angles = camera_params["angles_deg"]
    image_size = camera_params["image_size"]
    pixel_size_mm = camera_params["pixel_size_mm"]
    vol_size_mm = tuple(mcx_params["vol_size_mm"])
    voxel_size_mm = mcx_params["voxel_size_mm"]

    config_output_dir = Path(output_dir) / config_id
    config_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating MCX GT for {config_id}: {cfg['description']}")
    logger.info(f"  Source center: {source_center}")
    logger.info(f"  Tissue: {tissue_type}")

    # Step 1: 创建 MCX 体积
    volume_dict = create_homogeneous_volume(
        tissue_mu_a=tissue_params["mu_a"],
        tissue_mu_sp=tissue_params["mu_sp"],
        tissue_g=tissue_params["g"],
        tissue_n=tissue_params["n"],
        vol_size_mm=vol_size_mm,
        voxel_size_mm=voxel_size_mm,
    )

    # Step 2: 准备 MCX 点源配置（固定位置）
    # MCX 源位置：体积坐标系（左上角为原点）
    source_pos_xy = (
        source_center[0] + vol_size_mm[0] / 2,
        source_center[1] + vol_size_mm[1] / 2,
    )
    source_depth_mm = source_center[2]

    logger.info(f"  MCX source position: xy={source_pos_xy}, depth={source_depth_mm}")

    # Step 3: 运行一次 MCX
    json_path, _ = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=source_depth_mm,
        source_pos_xy=source_pos_xy,
        n_photons=mcx_params["n_photons"],
        session_id="fluence_volume",
        output_dir=config_output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec=mcx_params["mcx_exec"])
    fluence = load_mcx_fluence(jnii_path)

    logger.info(f"  Fluence volume shape: {fluence.shape}")
    logger.info(f"  Fluence max: {fluence.max():.6e}")

    # Step 4: 使用 project_volume_reference 生成多视角投影
    # 构建 TurntableCamera 配置
    view_cfg = {
        "angles": view_angles,
        "pose": "prone",
        "camera_distance_mm": 200.0,
        "detector_resolution": [image_size, image_size],
        "projection_type": "orthographic",
        "platform_occlusion": False,  # 匀质体积不需要平台遮挡
        "fov_mm": pixel_size_mm * image_size,  # 计算 FOV
    }
    camera = TurntableCamera(view_cfg)

    images = []
    for angle in view_angles:
        logger.info(f"  Processing view {angle}°...")

        # 使用 project_volume_reference 生成投影
        proj, depth = project_volume_reference(
            fluence,
            angle,
            camera_distance_mm=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=voxel_size_mm,
        )

        images.append(proj)
        logger.info(f"    Projection max: {proj.max():.6e}")

    images = np.stack(images)

    # 保存 GT 数据
    gt_path = config_output_dir / f"{config_id}_mcx_gt.npz"
    np.savez(
        gt_path,
        images=images,
        view_angles=view_angles,
        source_center_gt=source_center,
        tissue_type=tissue_type,
        tissue_params=tissue_params,
    )

    meta_path = config_output_dir / f"{config_id}_mcx_gt_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "config_id": config_id,
                "tissue_type": tissue_type,
                "source_center_gt": source_center.tolist(),
                "view_angles": view_angles,
                "image_size": image_size,
                "pixel_size_mm": pixel_size_mm,
            },
            f,
            indent=2,
        )

    logger.info(f"  Saved: {gt_path}")

    return {
        "images": images,
        "view_angles": view_angles,
        "source_center_gt": source_center,
        "tissue_type": tissue_type,
        "tissue_params": tissue_params,
    }


def run_all_configs(config_path: str, output_dir: str) -> None:
    """运行所有配置的 MCX GT 生成"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in config["configs"]:
        config_id = cfg["id"]
        generate_mcx_multiview_gt(config, config_id, output_dir)

    logger.info(f"All MCX GT data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate MCX multiview GT for E1b")
    parser.add_argument(
        "--config",
        default="pilot/e1b_model_mismatch/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1b_model_mismatch/results/gt_data",
        help="Output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    run_all_configs(args.config, args.output)


if __name__ == "__main__":
    main()
