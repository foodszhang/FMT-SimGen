"""
调试 project_volume_reference 的实际输出
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from pilot.e0_psf_validation.mcx_point_source import (
    create_homogeneous_volume,
    generate_mcx_config_json,
    run_mcx_simulation,
    load_mcx_fluence,
)
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera


def debug_projection():
    """检查 project_volume_reference 的实际输出"""

    # 创建体积
    vol_size_mm = (30.0, 30.0, 20.0)
    voxel_size_mm = 0.1

    volume_dict = create_homogeneous_volume(
        tissue_mu_a=0.087,
        tissue_mu_sp=4.291,
        tissue_g=0.90,
        tissue_n=1.37,
        vol_size_mm=vol_size_mm,
        voxel_size_mm=voxel_size_mm,
    )

    # 源位置
    source_center = np.array([3.0, 3.0, 2.0])
    source_pos_xy = (
        source_center[0] + vol_size_mm[0] / 2,
        source_center[1] + vol_size_mm[1] / 2,
    )
    source_depth_mm = source_center[2]

    output_dir = Path("pilot/e1b_model_mismatch/results/gt_data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行 MCX
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

    print("=" * 80)
    print("调试 project_volume_reference")
    print("=" * 80)
    print(f"Fluence shape: {fluence.shape}")
    print(f"Source: {source_center}")

    # 测试不同角度的投影
    camera = TurntableCamera(
        {
            "angles": [-90, -60, -30, 0, 30, 60, 90],
            "pose": "prone",
            "camera_distance_mm": 200.0,
            "detector_resolution": [256, 256],
            "projection_type": "orthographic",
            "platform_occlusion": False,
            "fov_mm": 50.0,
        }
    )

    for angle in camera.angles:
        proj, depth_map = project_volume_reference(
            fluence,
            angle,
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=voxel_size_mm,
        )

        # 找到投影峰值
        max_idx = np.unravel_index(np.argmax(proj), proj.shape)

        # 转换为物理坐标（以图像中心为原点）
        pixel_size_mm = camera.fov_mm / camera.detector_resolution[0]
        proj_x = (max_idx[1] - proj.shape[1] / 2) * pixel_size_mm
        proj_y = (max_idx[0] - proj.shape[0] / 2) * pixel_size_mm

        print(f"\nAngle {angle}°:")
        print(f"  Projection peak (pixel): ({max_idx[1]}, {max_idx[0]})")
        print(f"  Projection peak (mm):    ({proj_x:6.2f}, {proj_y:6.2f})")
        print(f"  Max value: {proj.max():.6e}")

    # 清理
    import shutil

    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    debug_projection()
