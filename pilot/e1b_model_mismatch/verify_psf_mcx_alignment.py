"""
验证 PSF 渲染器和 MCX GT 的峰值位置是否对齐
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import yaml

from pilot.e1_single_source.psf_splatting import (
    GaussianSource,
    PSFSplattingRenderer,
    TissueParams,
    build_turntable_views,
)


def find_peak_position(image: np.ndarray) -> tuple:
    """找到图像峰值位置（质心）"""
    img_norm = image / image.max() if image.max() > 0 else image
    y_coords, x_coords = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]

    total = img_norm.sum()
    if total > 0:
        cx = (x_coords * img_norm).sum() / total
        cy = (y_coords * img_norm).sum() / total
        return cx, cy
    return image.shape[1] / 2, image.shape[0] / 2


def verify_alignment(config_id: str = "M03"):
    """验证 PSF 和 MCX 的峰值位置对齐"""

    # 加载配置
    config_path = Path("pilot/e1b_model_mismatch/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 找到对应配置
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
    psf_calibration = config["psf_calibration"][tissue_type]
    camera_params = config["camera"]

    # 加载 MCX GT
    gt_path = Path(
        f"pilot/e1b_model_mismatch/results/gt_data/{config_id}/{config_id}_mcx_gt.npz"
    )
    meta_path = Path(
        f"pilot/e1b_model_mismatch/results/gt_data/{config_id}/{config_id}_mcx_gt_meta.json"
    )

    if not gt_path.exists():
        print(f"GT not found at {gt_path}, please run generate_mcx_gt.py first")
        return

    gt_data = np.load(gt_path)
    gt_images = gt_data["images"]
    view_angles = gt_data["view_angles"].tolist()

    with open(meta_path) as f:
        meta = json.load(f)

    pixel_size_mm = meta["pixel_size_mm"]
    image_size = meta["image_size"]

    print("=" * 80)
    print(f"验证 PSF vs MCX 峰值对齐: {config_id}")
    print(f"Source: {source_center}, Tissue: {tissue_type}")
    print(f"Image size: {image_size}x{image_size}, Pixel size: {pixel_size_mm}mm")
    print("=" * 80)

    # 构建 PSF 渲染器
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
        vol_size_mm=tuple(camera_params["vol_size_mm"]),
    )

    # 创建 Gaussian 源（点源，sigma=0）
    source = GaussianSource(
        center_init=source_center.astype(np.float32),
        sigma_init=0.1,  # 小 sigma 近似点源
        alpha_init=1.0,
    )

    # 构建视角
    view_matrices, surface_normals = build_turntable_views(
        n_views=len(view_angles),
        angles_deg=view_angles,
    )

    # 渲染 PSF 图像
    psf_images = renderer.render_all_views(source, view_matrices, surface_normals)
    psf_images = psf_images.detach().numpy()

    # 比较每个视角的峰值位置
    print(
        f"\n{'Angle':>8} | {'MCX Peak (px)':>20} | {'PSF Peak (px)':>20} | {'Diff (px)':>15}"
    )
    print("-" * 80)

    all_aligned = True
    for i, angle in enumerate(view_angles):
        mcx_img = gt_images[i]
        psf_img = psf_images[i]

        mcx_peak = find_peak_position(mcx_img)
        psf_peak = find_peak_position(psf_img)

        diff_x = mcx_peak[0] - psf_peak[0]
        diff_y = mcx_peak[1] - psf_peak[1]
        diff_px = np.sqrt(diff_x**2 + diff_y**2)
        diff_mm = diff_px * pixel_size_mm

        mcx_str = f"({mcx_peak[0]:.1f}, {mcx_peak[1]:.1f})"
        psf_str = f"({psf_peak[0]:.1f}, {psf_peak[1]:.1f})"
        diff_str = f"{diff_px:.1f}px ({diff_mm:.2f}mm)"

        aligned = diff_px < 2.0  # 允许 2 像素误差
        status = "✓" if aligned else "✗"

        print(f"{angle:>8}° | {mcx_str:>20} | {psf_str:>20} | {diff_str:>15} {status}")

        if not aligned:
            all_aligned = False

    print("-" * 80)
    if all_aligned:
        print("✓ 所有视角峰值对齐良好（< 2px）")
    else:
        print("✗ 存在峰值不对齐，需要检查坐标映射")

    # 也打印物理坐标
    print("\n物理坐标对比（以图像中心为原点）:")
    print(f"{'Angle':>8} | {'MCX (mm)':>20} | {'PSF (mm)':>20}")
    print("-" * 60)
    for i, angle in enumerate(view_angles):
        mcx_img = gt_images[i]
        psf_img = psf_images[i]

        mcx_peak = find_peak_position(mcx_img)
        psf_peak = find_peak_position(psf_img)

        # 转换为物理坐标（以图像中心为原点）
        mcx_x = (mcx_peak[0] - image_size / 2) * pixel_size_mm
        mcx_y = (mcx_peak[1] - image_size / 2) * pixel_size_mm
        psf_x = (psf_peak[0] - image_size / 2) * pixel_size_mm
        psf_y = (psf_peak[1] - image_size / 2) * pixel_size_mm

        print(
            f"{angle:>8}° | ({mcx_x:6.2f}, {mcx_y:6.2f}) | ({psf_x:6.2f}, {psf_y:6.2f})"
        )


if __name__ == "__main__":
    verify_alignment("M03")  # 偏心源最能检验坐标对齐
