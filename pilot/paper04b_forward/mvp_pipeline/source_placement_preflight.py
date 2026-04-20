"""Source placement preflight and relocation.

P2/P3/P4 at Y=10 fall in liver/lung instead of soft_tissue.
This script:
1. Checks if source position is in soft_tissue
2. If not, pushes outward along surface normal until soft_tissue is reached
3. Saves corrected positions to configs/positions_y10_soft_tissue.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)

SOFT_TISSUE_LABEL = 1

ORGAN_NAMES = {
    0: "air",
    1: "soft_tissue",
    2: "bone",
    3: "brain",
    4: "heart",
    5: "stomach",
    6: "abdominal",
    7: "liver",
    8: "kidney",
    9: "lung",
}


def load_volume() -> np.ndarray:
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    return volume.reshape(VOLUME_SHAPE_XYZ)


def mm_to_voxel(mm: np.ndarray, center: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(mm / voxel_size + center).astype(int)


def voxel_to_mm(voxel: np.ndarray, center: np.ndarray, voxel_size: float) -> np.ndarray:
    return (voxel - center + 0.5) * voxel_size


def get_label_at_pos(
    pos_mm: np.ndarray, volume: np.ndarray, center: np.ndarray, voxel_size: float
) -> int:
    voxel = mm_to_voxel(pos_mm, center, voxel_size)
    if (
        0 <= voxel[0] < volume.shape[0]
        and 0 <= voxel[1] < volume.shape[1]
        and 0 <= voxel[2] < volume.shape[2]
    ):
        return int(volume[voxel[0], voxel[1], voxel[2]])
    return 0


def find_nearest_tissue_direction(
    pos_mm: np.ndarray,
    volume: np.ndarray,
    center: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Find the direction to nearest tissue from air position."""
    voxel = mm_to_voxel(pos_mm, center, voxel_size)

    best_dir = np.array([0.0, 0.0, 1.0])
    best_dist = float("inf")

    for dz in range(-20, 21):
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                test_voxel = voxel + np.array([dx, dy, dz])
                if (
                    0 <= test_voxel[0] < volume.shape[0]
                    and 0 <= test_voxel[1] < volume.shape[1]
                    and 0 <= test_voxel[2] < volume.shape[2]
                ):
                    if (
                        volume[test_voxel[0], test_voxel[1], test_voxel[2]]
                        == SOFT_TISSUE_LABEL
                    ):
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_dir = np.array([dx, dy, dz], dtype=float)
                            if dist > 0:
                                best_dir /= dist

    return best_dir


def push_to_soft_tissue(
    pos_mm: np.ndarray,
    volume: np.ndarray,
    center: np.ndarray,
    voxel_size: float,
    max_push_mm: float = 5.0,
    step_mm: float = 0.5,
) -> Tuple[np.ndarray, str, int]:
    """Push position along surface normal until soft_tissue is reached.

    Returns (new_pos, action_taken, original_label).
    """
    original_label = get_label_at_pos(pos_mm, volume, center, voxel_size)

    if original_label == SOFT_TISSUE_LABEL:
        return pos_mm, "already_in_soft_tissue", original_label

    logger.info(
        f"Position {pos_mm} is in {ORGAN_NAMES.get(original_label, 'unknown')}, pushing..."
    )

    if original_label == 0:
        direction = find_nearest_tissue_direction(pos_mm, volume, center, voxel_size)
    else:
        binary_mask = volume > 0
        voxel = mm_to_voxel(pos_mm, center, voxel_size)

        gradient = np.zeros(3)
        for ax in range(3):
            for delta in [-1, 1]:
                test_voxel = voxel.copy()
                test_voxel[ax] += delta
                if (
                    0 <= test_voxel[0] < volume.shape[0]
                    and 0 <= test_voxel[1] < volume.shape[1]
                    and 0 <= test_voxel[2] < volume.shape[2]
                ):
                    if binary_mask[test_voxel[0], test_voxel[1], test_voxel[2]]:
                        gradient[ax] -= delta

        if np.linalg.norm(gradient) > 0:
            direction = gradient / np.linalg.norm(gradient)
        else:
            direction = np.array([0, 0, 1])

    n_steps = int(max_push_mm / step_mm)
    for i in range(1, n_steps + 1):
        new_pos = pos_mm + i * step_mm * direction
        new_label = get_label_at_pos(new_pos, volume, center, voxel_size)

        if new_label == SOFT_TISSUE_LABEL:
            logger.info(f"  Found soft_tissue at step {i} (offset {i * step_mm:.1f}mm)")
            return new_pos, f"pushed_{i * step_mm:.1f}mm", original_label

    logger.warning(f"  Could not find soft_tissue within {max_push_mm}mm")
    return pos_mm, f"failed_to_find_soft_tissue", original_label


def main():
    volume = load_volume()
    center = np.array(VOLUME_SHAPE_XYZ) / 2

    positions_y10_original = {
        "P1-dorsal": {"pos_mm": [-0.6, 2.4, 5.8], "best_angle": 0},
        "P2-left": {"pos_mm": [-8.0, 2.4, 1.0], "best_angle": 90},
        "P3-right": {"pos_mm": [6.8, 2.4, 1.0], "best_angle": -90},
        "P4-dorsal-lat": {"pos_mm": [-6.3, 2.4, 5.8], "best_angle": -30},
        "P5-ventral": {"pos_mm": [-0.6, 2.4, -3.8], "best_angle": 180},
    }

    print("=" * 70)
    print("Source Placement Preflight (Y=10mm)")
    print("=" * 70)

    corrected_positions = {}

    for name, info in positions_y10_original.items():
        pos_mm = np.array(info["pos_mm"])
        best_angle = info["best_angle"]

        label = get_label_at_pos(pos_mm, volume, center, VOXEL_SIZE_MM)
        organ = ORGAN_NAMES.get(label, "unknown")

        if label == SOFT_TISSUE_LABEL:
            status = "✓ OK"
            new_pos = pos_mm
            action = "none"
        else:
            new_pos, action, orig_label = push_to_soft_tissue(
                pos_mm, volume, center, VOXEL_SIZE_MM
            )
            if action.startswith("pushed"):
                status = "✓ CORRECTED"
            else:
                status = "✗ FAILED"

        corrected_positions[name] = {
            "pos_mm": new_pos.tolist(),
            "best_angle": best_angle,
            "original_pos_mm": pos_mm.tolist(),
            "original_label": int(label),
            "original_organ": organ,
            "correction_action": action,
        }

        offset = np.linalg.norm(new_pos - pos_mm)
        print(f"{name}:")
        print(f"  Original: {pos_mm} -> {organ}")
        print(f"  Corrected: {new_pos} ({status}, offset={offset:.1f}mm)")
        print(f"  Action: {action}")
        print()

    output_dir = Path("pilot/paper04b_forward/configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "positions_y10_soft_tissue.yaml"
    with open(output_path, "w") as f:
        yaml.dump(corrected_positions, f, default_flow_style=False, sort_keys=False)

    print(f"Saved to {output_path}")
    print("=" * 70)
    print("Summary:")
    for name, info in corrected_positions.items():
        label = get_label_at_pos(
            np.array(info["pos_mm"]), volume, center, VOXEL_SIZE_MM
        )
        status = "✓" if label == SOFT_TISSUE_LABEL else "✗"
        print(f"  {name}: {status} ({ORGAN_NAMES.get(label, '?')})")


if __name__ == "__main__":
    main()
