"""
验证 PSF 渲染器和 MCX 投影模块的坐标一致性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from pilot.e1_single_source.psf_splatting import build_turntable_views
from fmt_simgen.mcx_projection import rotation_matrix_y


def verify_projection_coordinates():
    """验证 PSF 和 MCX 的投影坐标计算一致"""

    print("=" * 80)
    print("验证 PSF 和 MCX 投影坐标一致性")
    print("=" * 80)

    # 测试源位置
    test_sources = {
        "中心源": np.array([0.0, 0.0, 2.0]),
        "偏心源": np.array([3.0, 3.0, 2.0]),
        "负X源": np.array([-3.0, 2.0, 2.0]),
    }

    # 测试角度（与 TurntableCamera 默认值一致）
    angles = [-90, -60, -30, 0, 30, 60, 90]

    all_match = True

    for src_name, source in test_sources.items():
        print(f"\n{'=' * 60}")
        print(f"Source: {src_name} at {source}")
        print("=" * 60)

        for angle in angles:
            # MCX 投影的做法 (mcx_projection.py)
            R_mcx = rotation_matrix_y(angle)
            rotated_mcx = R_mcx @ source  # 列向量左乘
            cam_x_mcx = rotated_mcx[0]
            cam_y_mcx = rotated_mcx[1]

            # PSF 渲染器的做法 (修复后)
            vms, norms = build_turntable_views(angles_deg=[angle])
            R_psf = vms[0].numpy()
            rotated_psf = R_psf @ source
            cam_x_psf = rotated_psf[0]
            cam_y_psf = rotated_psf[1]

            # 检查是否匹配
            match = np.allclose(
                [cam_x_mcx, cam_y_mcx], [cam_x_psf, cam_y_psf], atol=0.01
            )
            status = "✓" if match else "✗"

            print(
                f"  θ={angle:4d}°: {status} MCX=({cam_x_mcx:7.3f},{cam_y_mcx:7.3f})  PSF=({cam_x_psf:7.3f},{cam_y_psf:7.3f})"
            )

            if not match:
                all_match = False
                print(
                    f"    差值: ({abs(cam_x_mcx - cam_x_psf):.6f}, {abs(cam_y_mcx - cam_y_psf):.6f})"
                )

    print("\n" + "=" * 80)
    if all_match:
        print("✓ 全部匹配！PSF 和 MCX 投影坐标一致")
    else:
        print("✗ 存在不匹配，需要检查旋转矩阵定义")
    print("=" * 80)

    return all_match


def verify_rotation_matrix():
    """验证旋转矩阵定义"""

    print("\n" + "=" * 80)
    print("验证旋转矩阵定义")
    print("=" * 80)

    # MCX 的 rotation_matrix_y
    from fmt_simgen.mcx_projection import rotation_matrix_y

    angle = 30
    R_mcx = rotation_matrix_y(angle)

    print(f"\nMCX rotation_matrix_y({angle}°):")
    print(R_mcx)

    # PSF 的旋转矩阵
    theta = np.radians(angle)
    R_psf = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    print(f"\nPSF build_turntable_views rotation ({angle}°):")
    print(R_psf)

    # 检查是否相同
    match = np.allclose(R_mcx, R_psf)
    print(f"\n旋转矩阵相同: {'✓' if match else '✗'}")

    # 测试一个点的旋转
    point = np.array([3.0, 0.0, 2.0])
    rotated_mcx = R_mcx @ point
    rotated_psf = R_psf @ point

    print(f"\n测试点 {point}:")
    print(f"  MCX: {rotated_mcx}")
    print(f"  PSF: {rotated_psf}")

    return match


if __name__ == "__main__":
    rot_match = verify_rotation_matrix()
    coord_match = verify_projection_coordinates()

    print("\n" + "=" * 80)
    print("总结:")
    print(f"  旋转矩阵定义: {'✓ 匹配' if rot_match else '✗ 不匹配'}")
    print(f"  投影坐标计算: {'✓ 匹配' if coord_match else '✗ 不匹配'}")
    print("=" * 80)
