"""
用 MCX 生成多视角 GT 荧光图像

对每个视角 θ：
1. 将源位置绕 y 轴旋转 -θ（等效于体积旋转 +θ）
2. 构建匀质体积 + 设置旋转后的点源
3. MCX 仿真 → 3D fluence → 提取 z=0 表面 → 2D 图像
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilot.e0_psf_validation.mcx_point_source import (
    create_homogeneous_volume,
    generate_mcx_config_json,
    run_mcx_simulation,
    load_mcx_fluence,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def rotate_point_around_y(point: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    绕 y 轴旋转点（模拟 turntable）

    Args:
        point: (3,) [x, y, z]
        angle_deg: 旋转角度（度）

    Returns:
        rotated point (3,)
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            c * point[0] + s * point[2],
            point[1],
            -s * point[0] + c * point[2],
        ]
    )


def generate_single_view_mcx(
    tissue_params: dict,
    source_center: np.ndarray,
    view_angle_deg: float,
    vol_size_mm: Tuple[float, float, float],
    voxel_size_mm: float,
    n_photons: int,
    image_size: int,
    output_dir: Path,
    mcx_exec: str,
) -> np.ndarray:
    """
    生成单个视角的 MCX GT 图像

    Args:
        source_center: (3,) 源位置 [x, y, depth] 在 θ=0 时的坐标

    Returns:
        2D surface image (H, W)
    """
    rotated_center = rotate_point_around_y(source_center, -view_angle_deg)

    depth = rotated_center[2]
    if depth < 0.1:
        logger.info(
            f"  View {view_angle_deg}°: source outside volume (z={depth:.2f}), returning zeros"
        )
        return np.zeros((image_size, image_size), dtype=np.float32)

    volume_dict = create_homogeneous_volume(
        tissue_mu_a=tissue_params["mu_a"],
        tissue_mu_sp=tissue_params["mu_sp"],
        tissue_g=tissue_params["g"],
        tissue_n=tissue_params["n"],
        vol_size_mm=vol_size_mm,
        voxel_size_mm=voxel_size_mm,
    )

    source_pos_xy = (
        rotated_center[0] + vol_size_mm[0] / 2,
        rotated_center[1] + vol_size_mm[1] / 2,
    )

    json_path, _ = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=depth,
        source_pos_xy=source_pos_xy,
        n_photons=n_photons,
        session_id=f"view_{int(view_angle_deg):03d}",
        output_dir=output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec=mcx_exec)
    fluence = load_mcx_fluence(jnii_path)

    surface = fluence[:, :, 0]

    nx, ny = surface.shape
    cx, cy = nx // 2, ny // 2
    half = image_size // 2

    if cx - half < 0 or cy - half < 0 or cx + half > nx or cy + half > ny:
        padded = np.zeros((image_size, image_size), dtype=np.float32)
        min_x = max(0, half - cx)
        max_x = min(image_size, half + (nx - cx))
        min_y = max(0, half - cy)
        max_y = min(image_size, half + (ny - cy))
        src_x_start = max(0, cx - half)
        src_x_end = min(nx, cx + half)
        src_y_start = max(0, cy - half)
        src_y_end = min(ny, cy + half)
        padded[min_x:max_x, min_y:max_y] = surface[
            src_x_start:src_x_end, src_y_start:src_y_end
        ]
        return padded

    return surface[cx - half : cx + half, cy - half : cy + half].astype(np.float32)


def generate_mcx_multiview_gt(
    config: dict,
    config_id: str,
    output_dir: str,
) -> dict:
    """
    生成单个配置的 MCX 多视角 GT 数据

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
    vol_size_mm = tuple(mcx_params["vol_size_mm"])

    config_output_dir = Path(output_dir) / config_id
    config_output_dir.mkdir(parents=True, exist_ok=True)

    images = []

    logger.info(f"Generating MCX GT for {config_id}: {cfg['description']}")
    logger.info(f"  Source center (θ=0): {source_center}")

    for angle in view_angles:
        logger.info(f"  Processing view {angle}°...")

        view_output_dir = config_output_dir / f"view_{int(angle):03d}"

        surface = generate_single_view_mcx(
            tissue_params=tissue_params,
            source_center=source_center,
            view_angle_deg=angle,
            vol_size_mm=vol_size_mm,
            voxel_size_mm=mcx_params["voxel_size_mm"],
            n_photons=mcx_params["n_photons"],
            image_size=image_size,
            output_dir=view_output_dir,
            mcx_exec=mcx_params["mcx_exec"],
        )

        images.append(surface)

    images = np.stack(images)

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
