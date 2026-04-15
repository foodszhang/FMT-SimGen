"""
最终验证：对比 MCX project_volume_reference 和 PSF 渲染器的输出
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from pilot.e0_psf_validation.mcx_point_source import (
    create_homogeneous_volume,
    generate_mcx_config_json,
    run_mcx_simulation,
    load_mcx_fluence,
)
from pilot.e1_single_source.psf_splatting import (
    GaussianSource,
    PSFSplattingRenderer,
    TissueParams,
    build_turntable_views,
)
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera


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


def final_verification():
    """最终验证 MCX 和 PSF 投影对齐"""

    # 参数
    vol_size_mm = (30.0, 30.0, 20.0)
    voxel_size_mm = 0.1
    source_center = np.array([3.0, 3.0, 2.0])
    tissue_params = {"mu_a": 0.087, "mu_sp": 4.291, "g": 0.90, "n": 1.37}
    psf_calibration = {"k": 0.801469, "p": 0.576760}

    image_size = 256
    fov_mm = 50.0
    pixel_size_mm = fov_mm / image_size

    angles = [-90, -60, -30, 0, 30, 60, 90]

    print("=" * 80)
    print("最终验证：MCX vs PSF 投影对齐")
    print("=" * 80)
    print(f"Source: {source_center}")
    print(
        f"FOV: {fov_mm}mm, Image: {image_size}x{image_size}, Pixel: {pixel_size_mm:.4f}mm"
    )

    # 创建 MCX 体积并运行
    volume_dict = create_homogeneous_volume(
        tissue_mu_a=tissue_params["mu_a"],
        tissue_mu_sp=tissue_params["mu_sp"],
        tissue_g=tissue_params["g"],
        tissue_n=tissue_params["n"],
        vol_size_mm=vol_size_mm,
        voxel_size_mm=voxel_size_mm,
    )

    source_pos_xy = (
        source_center[0] + vol_size_mm[0] / 2,
        source_center[1] + vol_size_mm[1] / 2,
    )
    source_depth_mm = source_center[2]

    output_dir = Path("pilot/e1b_model_mismatch/results/gt_data/final_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path, _ = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=source_depth_mm,
        source_pos_xy=source_pos_xy,
        n_photons=10000000,
        session_id="debug",
        output_dir=output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec="/mnt/f/win-pro/bin/mcx.exe")
    fluence = load_mcx_fluence(jnii_path)

    # MCX 投影
    mcx_images = []
    for angle in angles:
        proj, _ = project_volume_reference(
            fluence,
            angle,
            camera_distance=200.0,
            fov_mm=fov_mm,
            detector_resolution=(image_size, image_size),
            voxel_size_mm=voxel_size_mm,
        )
        mcx_images.append(proj)
    mcx_images = np.stack(mcx_images)

    # PSF 渲染
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
    )

    source = GaussianSource(
        center_init=source_center.astype(np.float32),
        sigma_init=0.1,
        alpha_init=1.0,
    )

    view_matrices, surface_normals = build_turntable_views(
        n_views=len(angles),
        angles_deg=angles,
    )

    psf_images = renderer.render_all_views(source, view_matrices, surface_normals)
    psf_images = psf_images.detach().numpy()

    # 对比峰值位置
    print(
        f"\n{'Angle':>8} | {'MCX Peak (mm)':>20} | {'PSF Peak (mm)':>20} | {'Diff (mm)':>12}"
    )
    print("-" * 70)

    all_close = True
    for i, angle in enumerate(angles):
        mcx_peak = find_peak_position(mcx_images[i])
        psf_peak = find_peak_position(psf_images[i])

        # 转换为物理坐标
        mcx_x = (mcx_peak[0] - image_size / 2) * pixel_size_mm
        mcx_y = (mcx_peak[1] - image_size / 2) * pixel_size_mm
        psf_x = (psf_peak[0] - image_size / 2) * pixel_size_mm
        psf_y = (psf_peak[1] - image_size / 2) * pixel_size_mm

        diff = np.sqrt((mcx_x - psf_x) ** 2 + (mcx_y - psf_y) ** 2)

        mcx_str = f"({mcx_x:6.2f}, {mcx_y:6.2f})"
        psf_str = f"({psf_x:6.2f}, {psf_y:6.2f})"

        close = diff < 0.5  # 0.5mm 容差
        status = "✓" if close else "✗"

        print(f"{angle:>8}° | {mcx_str:>20} | {psf_str:>20} | {diff:>10.2f} {status}")

        if not close:
            all_close = False

    print("-" * 70)
    if all_close:
        print("✓ 全部对齐（< 0.5mm）！")
    else:
        print("✗ 存在未对齐的视角")

    # 清理
    import shutil

    shutil.rmtree(output_dir, ignore_errors=True)

    return all_close


if __name__ == "__main__":
    final_verification()
