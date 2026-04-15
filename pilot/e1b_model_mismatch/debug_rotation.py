"""
调试旋转方向问题
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from pilot.e1_single_source.psf_splatting import build_turntable_views
from fmt_simgen.mcx_projection import rotation_matrix_y


def debug_rotation():
    """调试 PSF 和 MCX 的旋转方向"""

    source = np.array([3.0, 3.0, 2.0])

    print("=" * 80)
    print("调试旋转方向")
    print(f"Source: {source}")
    print("=" * 80)

    # project_volume_reference 的做法
    print("\nproject_volume_reference 的做法:")
    print("R = rotation_matrix_y(angle)")
    print("rotated = R @ source")

    for angle in [-90, -60, -30, 0, 30, 60, 90]:
        R = rotation_matrix_y(angle)
        rotated = R @ source
        print(
            f"  θ={angle:4d}°: rotated = [{rotated[0]:7.3f}, {rotated[1]:7.3f}, {rotated[2]:7.3f}]"
        )

    # build_turntable_views 的做法
    print("\nbuild_turntable_views 的做法:")
    vms, norms = build_turntable_views(angles_deg=[-90, -60, -30, 0, 30, 60, 90])

    for i, angle in enumerate([-90, -60, -30, 0, 30, 60, 90]):
        R = vms[i].numpy()
        rotated = R @ source
        print(
            f"  θ={angle:4d}°: rotated = [{rotated[0]:7.3f}, {rotated[1]:7.3f}, {rotated[2]:7.3f}]"
        )

    # 检查旋转矩阵
    print("\n旋转矩阵对比 (30°):")
    R_mcx = rotation_matrix_y(30)
    vms, _ = build_turntable_views(angles_deg=[30])
    R_psf = vms[0].numpy()

    print(f"MCX:\n{R_mcx}")
    print(f"PSF:\n{R_psf}")
    print(f"相同: {np.allclose(R_mcx, R_psf)}")

    # 关键问题：project_volume_reference 中的旋转
    print("\n\n关键问题分析:")
    print("=" * 80)
    print("project_volume_reference 中:")
    print("  1. 体素坐标绕 Y 轴旋转 angle 度")
    print("  2. 然后投影到 XY 平面")
    print("  3. cam_x = rotated_x, cam_y = rotated_y")

    # 让我们模拟一个点源在体积中的情况
    print("\n模拟: source = [3, 3, 2] (世界坐标)")
    print("期望的投影位置 (mm，以图像中心为原点):")

    for angle in [-90, -60, -30, 0, 30, 60, 90]:
        R = rotation_matrix_y(angle)
        rotated = R @ source

        # project_volume_reference 的投影
        proj_x = rotated[0]
        proj_y = rotated[1]

        print(f"  θ={angle:4d}°: proj=({proj_x:7.3f}, {proj_y:7.3f})")

    # 现在检查 PSF 渲染器的做法
    print("\nPSF 渲染器的实际做法:")
    print("center_cam = view_matrix @ source.center")
    print("proj_x = center_cam[0]")
    print("proj_y = center_cam[1]")

    vms, _ = build_turntable_views(angles_deg=[-90, -60, -30, 0, 30, 60, 90])
    for i, angle in enumerate([-90, -60, -30, 0, 30, 60, 90]):
        center_cam = vms[i] @ torch.tensor(source, dtype=torch.float32)
        proj_x = center_cam[0].item()
        proj_y = center_cam[1].item()
        print(f"  θ={angle:4d}°: proj=({proj_x:7.3f}, {proj_y:7.3f})")

    # 问题：MCX GT 的峰值位置
    print("\n\n实际 MCX GT 峰值位置 (从 verify_psf_mcx_alignment.py):")
    print("  -90°: (5.70, 3.02) mm")
    print("    0°: (2.99, 2.88) mm")
    print("   90°: (-5.17, 2.99) mm")

    print("\n问题分析:")
    print("  在 0° 时，MCX 和 PSF 都显示 (3, 3) - 正确")
    print("  在 -90° 时，MCX 显示 (5.7, 3.0)，但 PSF 显示 (0, 0)")
    print("  这意味着 MCX 的旋转方向与 PSF 相反！")

    print("\n\n验证反向旋转:")
    for angle in [-90, 0, 90]:
        R_psf = rotation_matrix_y(angle)
        R_mcx = rotation_matrix_y(-angle)  # 反向

        rotated_psf = R_psf @ source
        rotated_mcx = R_mcx @ source

        print(f"\nθ={angle}°:")
        print(f"  PSF (R@{angle:3d}°):  {rotated_psf}")
        print(f"  MCX (R@{-angle:3d}°): {rotated_mcx}")


if __name__ == "__main__":
    debug_rotation()
