"""
调试 MCX 体积坐标系统
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


def debug_mcx_coords():
    """检查 MCX 体积和 fluence 的坐标系统"""

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

    print("=" * 80)
    print("MCX 体积坐标系统")
    print("=" * 80)
    print(f"Volume shape: {volume_dict['vol_shape']}")
    print(f"  (nz, ny, nx) = {volume_dict['vol_shape']}")
    print(f"Voxel size: {volume_dict['voxel_size']} mm")
    print(f"Volume size: {vol_size_mm} mm")

    # 源位置
    source_center = np.array([3.0, 3.0, 2.0])  # x, y, depth
    source_pos_xy = (
        source_center[0] + vol_size_mm[0] / 2,
        source_center[1] + vol_size_mm[1] / 2,
    )
    source_depth_mm = source_center[2]

    print(f"\n源位置（世界坐标）: {source_center}")
    print(f"  x={source_center[0]}, y={source_center[1]}, depth={source_center[2]}")
    print(f"MCX 源位置（体积坐标）:")
    print(f"  xy=({source_pos_xy[0]}, {source_pos_xy[1]}), depth={source_depth_mm}")

    # 转换为体素索引
    nx, ny, nz = volume_dict["vol_shape"]
    voxel_x = int(source_pos_xy[0] / voxel_size_mm)
    voxel_y = int(source_pos_xy[1] / voxel_size_mm)
    voxel_z = int(source_depth_mm / voxel_size_mm)

    print(f"\n源体素索引: ({voxel_x}, {voxel_y}, {voxel_z})")
    print(f"  nx={nx}, ny={ny}, nz={nz}")

    # 体积中心（体素索引）
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    print(f"体积中心（体素索引）: ({cx}, {cy}, {cz})")

    # 检查 fluence 形状
    output_dir = Path("pilot/e1b_model_mismatch/results/gt_data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path, _ = generate_mcx_config_json(
        volume_dict=volume_dict,
        source_depth_mm=source_depth_mm,
        source_pos_xy=source_pos_xy,
        n_photons=10000000,  # 较少光子用于调试
        session_id="debug",
        output_dir=output_dir,
    )

    jnii_path = run_mcx_simulation(json_path, mcx_exec="/mnt/f/win-pro/bin/mcx.exe")
    fluence = load_mcx_fluence(jnii_path)

    print(f"\nFluence shape: {fluence.shape}")
    print(f"  (nx, ny, nz) = {fluence.shape}")

    # 找到 fluence 峰值位置
    max_idx = np.unravel_index(np.argmax(fluence), fluence.shape)
    print(f"Fluence 峰值位置（体素索引）: {max_idx}")

    # 转换为物理坐标（以体积中心为原点）
    peak_x = (max_idx[0] - fluence.shape[0] / 2) * voxel_size_mm
    peak_y = (max_idx[1] - fluence.shape[1] / 2) * voxel_size_mm
    peak_z = (max_idx[2] - fluence.shape[2] / 2) * voxel_size_mm

    print(
        f"Fluence 峰值（以体积中心为原点）: ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f}) mm"
    )

    # 对比源位置
    print(f"\n对比:")
    print(
        f"  源位置（世界）: ({source_center[0]}, {source_center[1]}, {source_center[2]})"
    )
    print(f"  峰值位置（fluence）: ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")

    # 清理
    import shutil

    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    debug_mcx_coords()
