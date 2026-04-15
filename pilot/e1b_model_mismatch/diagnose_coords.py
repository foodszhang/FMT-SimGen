"""
诊断 MCX GT 和 PSF 渲染器之间的坐标映射

对比：
1. MCX 生成 GT 时的坐标变换
2. PSF 渲染器预测的投影位置
"""

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pilot.e1_single_source.psf_splatting import build_turntable_views


def rotate_point_around_y(point: np.ndarray, angle_deg: float) -> np.ndarray:
    """MCX 使用的坐标旋转（绕 y 轴）"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            c * point[0] + s * point[2],
            point[1],
            -s * point[0] + c * point[2],
        ]
    )


def analyze_projection():
    """分析投影坐标映射"""

    vol_size_mm = (30.0, 30.0, 20.0)
    image_size = 256
    pixel_size_mm = 0.15

    view_angles = [0, 30, 60, 90]

    view_matrices, surface_normals = build_turntable_views(
        n_views=len(view_angles),
        angles_deg=view_angles,
        device=torch.device("cpu"),
    )

    test_sources = {
        "M01 (center)": np.array([0.0, 0.0, 2.0]),
        "M03 (eccentric)": np.array([3.0, 3.0, 2.0]),
    }

    print("=" * 80)
    print("MCX vs PSF 坐标映射分析")
    print("=" * 80)

    for src_name, src_center in test_sources.items():
        print(f"\n{'=' * 60}")
        print(f"Source: {src_name} at {src_center}")
        print("=" * 60)

        for i, angle in enumerate(view_angles):
            print(f"\n--- View {angle}° ---")

            # MCX 坐标变换
            rotated_mcx = rotate_point_around_y(src_center, -angle)
            depth_mcx = rotated_mcx[2]

            # MCX 源在体积中的位置
            src_pos_x_mcx = rotated_mcx[0] + vol_size_mm[0] / 2
            src_pos_y_mcx = rotated_mcx[1] + vol_size_mm[1] / 2

            # MCX 表面图像坐标 (z=0 切片的峰值位置)
            # 峰值位置 = 源的 x,y 位置（因为 PSF 中心扩散）
            # 图像中心 = 体积中心
            # pixel_x = src_pos_x / pixel_size
            # pixel_y = src_pos_y / pixel_size
            pixel_x_mcx = src_pos_x_mcx / pixel_size_mm
            pixel_y_mcx = src_pos_y_mcx / pixel_size_mm

            # 图像坐标（以 256x256 中心为原点）
            # MCX volume 中心对应图像中心 (128, 128)
            img_x_mcx = pixel_x_mcx  # 已是以体积左上角为原点
            img_y_mcx = pixel_y_mcx

            # PSF 坐标变换
            vm = view_matrices[i]
            sn = surface_normals[i]

            center_world = torch.tensor(src_center, dtype=torch.float32)
            center_cam = vm @ center_world  # 3x3 @ 3

            # PSF 投影坐标（当前代码）
            proj_x_psf = center_cam[1].item()
            proj_y_psf = -center_cam[0].item()

            # PSF 图像坐标
            img_x_psf = proj_x_psf / pixel_size_mm + image_size / 2
            img_y_psf = proj_y_psf / pixel_size_mm + image_size / 2

            # 深度
            d_dot = torch.dot(torch.tensor(src_center, dtype=torch.float32), sn).item()

            print(
                f"MCX rotated center: [{rotated_mcx[0]:.3f}, {rotated_mcx[1]:.3f}, {rotated_mcx[2]:.3f}]"
            )
            print(f"MCX depth: {depth_mcx:.3f} mm")
            print(f"MCX src pos in vol: ({src_pos_x_mcx:.3f}, {src_pos_y_mcx:.3f}) mm")
            print(f"MCX pixel pos: ({pixel_x_mcx:.1f}, {pixel_y_mcx:.1f})")
            print()
            print(
                f"PSF center_cam: [{center_cam[0]:.3f}, {center_cam[1]:.3f}, {center_cam[2]:.3f}]"
            )
            print(f"PSF depth (dot): {d_dot:.3f} mm")
            print(f"PSF proj: ({proj_x_psf:.3f}, {proj_y_psf:.3f}) mm")
            print(f"PSF pixel pos: ({img_x_psf:.1f}, {img_y_psf:.1f})")
            print()

            # 关键问题：MCX surface 图像是 fluence[:,:,0]
            # fluence shape = (Nx, Ny, Nz)
            # 所以 surface[x, y] 对应物理坐标
            # 需要搞清楚 MCX fluence 的坐标轴顺序

            print(f"Coordinate analysis:")
            print(
                f"  MCX: rotated x={rotated_mcx[0]:.3f} → pixel_x={src_pos_x_mcx / pixel_size_mm:.1f}"
            )
            print(
                f"  MCX: rotated y={rotated_mcx[1]:.3f} → pixel_y={src_pos_y_mcx / pixel_size_mm:.1f}"
            )
            print(f"  PSF: cam[0]={center_cam[0]:.3f}, cam[1]={center_cam[1]:.3f}")
            print(
                f"  PSF: proj_x=cam[1]={proj_x_psf:.3f}, proj_y=-cam[0]={proj_y_psf:.3f}"
            )

    print("\n" + "=" * 80)
    print("关键发现：")
    print("=" * 80)
    print("""
MCX 坐标系：
- 旋转后源的 x 坐标 → 图像的 x 轴（水平）
- 旋转后源的 y 坐标 → 图像的 y 轴（垂直）
- z=0 表面切片，峰值在 源的位置

PSF 相机坐标系：
- center_cam[0] = 世界 x 在相机系中的投影
- center_cam[1] = 世界 y 在相机系中的投影
- center_cam[2] = 世界 z 在相机系中的投影（深度）

问题：center_cam[0] 和 center_cam[1] 的映射关系
""")


if __name__ == "__main__":
    analyze_projection()
