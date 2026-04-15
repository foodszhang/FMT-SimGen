"""
从 MCX GT 图像中提取实际峰值位置
"""

import numpy as np
from pathlib import Path
from scipy import ndimage


def find_peak_position(image: np.ndarray) -> tuple:
    """找到图像峰值位置"""
    # 使用质心法
    img_norm = image / image.max() if image.max() > 0 else image
    y_coords, x_coords = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]

    total = img_norm.sum()
    if total > 0:
        cx = (x_coords * img_norm).sum() / total
        cy = (y_coords * img_norm).sum() / total
        return cx, cy
    return image.shape[1] / 2, image.shape[0] / 2


def analyze_mcx_peaks():
    """分析 MCX GT 图像的峰值位置"""

    gt_dir = Path("pilot/e1b_model_mismatch/results/gt_data")

    configs = ["M01", "M03"]
    view_angles = [0, 30, 60, 90]

    print("=" * 80)
    print("MCX GT 峰值位置分析")
    print("=" * 80)

    for config_id in configs:
        gt_path = gt_dir / config_id / f"{config_id}_mcx_gt.npz"

        if not gt_path.exists():
            print(f"\n{config_id}: GT file not found at {gt_path}")
            continue

        data = np.load(gt_path)
        images = data["images"]
        source_center = data["source_center_gt"]

        print(f"\n{'=' * 60}")
        print(f"{config_id}: Source at {source_center}")
        print(f"Image shape: {images.shape}")
        print("=" * 60)

        for i, angle in enumerate(view_angles):
            img = images[i]
            peak_x, peak_y = find_peak_position(img)

            # 峰值物理坐标（以图像中心为原点）
            image_size = img.shape[0]
            pixel_size_mm = 0.15

            phys_x = (peak_x - image_size / 2) * pixel_size_mm
            phys_y = (peak_y - image_size / 2) * pixel_size_mm

            # 图像最大值位置
            max_pos = np.unravel_index(np.argmax(img), img.shape)

            print(f"\nView {angle}°:")
            print(f"  Peak (centroid): ({peak_x:.1f}, {peak_y:.1f}) px")
            print(f"  Peak (max):      ({max_pos[1]}, {max_pos[0]}) px")
            print(f"  Physical pos:    ({phys_x:.2f}, {phys_y:.2f}) mm from center")

            # MCX 期望的位置
            # 旋转源位置
            theta = np.radians(-angle)
            c, s = np.cos(theta), np.sin(theta)
            rotated = np.array(
                [
                    c * source_center[0] + s * source_center[2],
                    source_center[1],
                    -s * source_center[0] + c * source_center[2],
                ]
            )

            # MCX volume 物理坐标
            vol_size_mm = (30.0, 30.0, 20.0)
            src_pos_x = rotated[0] + vol_size_mm[0] / 2
            src_pos_y = rotated[1] + vol_size_mm[1] / 2

            # MCX 期望的像素位置
            expected_px = src_pos_x / pixel_size_mm
            expected_py = src_pos_y / pixel_size_mm

            print(f"  MCX expected:    ({expected_px:.1f}, {expected_py:.1f}) px")
            print(
                f"  Difference:      ({peak_x - expected_px:.1f}, {peak_y - expected_py:.1f}) px"
            )


if __name__ == "__main__":
    analyze_mcx_peaks()
