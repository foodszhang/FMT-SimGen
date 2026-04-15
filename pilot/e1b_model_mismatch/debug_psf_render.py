"""
调试 PSF 渲染器的各个步骤
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from pilot.e1_single_source.psf_splatting import (
    GaussianSource,
    TissueParams,
    build_turntable_views,
)


def debug_psf_render():
    """调试 PSF 渲染器的每个步骤"""

    source_center = np.array([3.0, 3.0, 2.0])
    angles = [-90, -60, -30, 0, 30, 60, 90]

    print("=" * 80)
    print("调试 PSF 渲染器")
    print("=" * 80)
    print(f"Source: {source_center}")

    tissue = TissueParams(mu_a=0.087, mu_sp=4.291, n=1.37)

    source = GaussianSource(
        center_init=source_center.astype(np.float32),
        sigma_init=0.1,
        alpha_init=1.0,
    )

    view_matrices, surface_normals = build_turntable_views(
        n_views=len(angles),
        angles_deg=angles,
    )

    print(f"\n{'Angle':>8} | {'center_cam':>30} | {'depth (dot)':>12} | {'normal':>20}")
    print("-" * 90)

    for i, angle in enumerate(angles):
        vm = view_matrices[i]
        sn = surface_normals[i]

        # 计算 center_cam
        center_cam = vm @ source.center

        # 计算深度
        d = torch.dot(source.center, sn)

        # 检查是否会被过滤掉
        visible = d >= 0.05

        normal_str = f"[{sn[0]:.3f}, {sn[1]:.3f}, {sn[2]:.3f}]"
        cam_str = f"[{center_cam[0]:.2f}, {center_cam[1]:.2f}, {center_cam[2]:.2f}]"

        status = "✓" if visible else "✗ FILTERED"

        print(
            f"{angle:>8}° | {cam_str:>30} | {d.item():>12.3f} | {normal_str:>20} {status}"
        )


if __name__ == "__main__":
    debug_psf_render()
