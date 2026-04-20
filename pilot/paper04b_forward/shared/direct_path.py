"""Direct-path preflight checker for closed-form forward validation.

A view is "direct-path" if:
1. The line segment from source to detector surface point traverses only soft tissue
2. The path length is within superficial regime (≤ 9mm default)

Through-organ views are explicitly out of scope for closed-form validation.

Usage:
    from shared.direct_path import assert_direct_path, is_direct_path

    # Check a single view
    result = is_direct_path(source_pos_mm, view_angle_deg, volume_labels, voxel_size_mm)
    if not result['is_direct']:
        print(f"View excluded: {result['reason']}")

    # Assert (raises exception for non-direct views)
    assert_direct_path(source_pos_mm, view_angle_deg, volume_labels, voxel_size_mm)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SOFT_TISSUE_LABELS = {1}
AIR_LABEL = 0
ORGAN_LABELS = {2, 3, 4, 5, 6, 7, 8, 9}

DEFAULT_MAX_PATH_MM = 9.0
DEFAULT_STEP_MM = 0.1


@dataclass
class DirectPathResult:
    is_direct: bool
    reason: str
    entry_point_mm: Optional[np.ndarray]
    exit_point_mm: Optional[np.ndarray]
    path_labels: Set[int]
    path_length_mm: float
    n_steps: int


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ]
    )


def find_surface_in_direction(
    source_pos_mm: np.ndarray,
    direction: np.ndarray,
    volume_labels: np.ndarray,
    voxel_size_mm: float,
    max_distance_mm: float = 50.0,
) -> Tuple[Optional[np.ndarray], List[int], float]:
    """Ray-march from source in given direction to find surface exit.

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position in mm (XYZ, relative to volume center).
    direction : np.ndarray
        Unit vector direction to march.
    volume_labels : np.ndarray
        Volume labels in XYZ order.
    voxel_size_mm : float
        Voxel size in mm.
    max_distance_mm : float
        Maximum ray-marching distance.

    Returns
    -------
    exit_point_mm : np.ndarray or None
        Exit point on surface, or None if no exit found.
    path_labels : list of int
        Labels encountered along path.
    path_length_mm : float
        Path length from source to exit.
    """
    center = np.array(volume_labels.shape) / 2

    def pos_to_voxel(pos_mm: np.ndarray) -> np.ndarray:
        return np.round(pos_mm / voxel_size_mm + center).astype(int)

    def voxel_in_bounds(voxel: np.ndarray) -> bool:
        return (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        )

    source_voxel = pos_to_voxel(source_pos_mm)
    if not voxel_in_bounds(source_voxel):
        return None, [], 0.0

    source_label = volume_labels[source_voxel[0], source_voxel[1], source_voxel[2]]
    if source_label == 0:
        return None, [], 0.0

    step_mm = DEFAULT_STEP_MM
    n_steps = int(max_distance_mm / step_mm)

    path_labels = [int(source_label)]
    exit_point_mm = None
    path_length_mm = 0.0

    for i in range(1, n_steps + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = pos_to_voxel(pos_mm)

        if not voxel_in_bounds(voxel):
            exit_point_mm = source_pos_mm + (i - 1) * step_mm * direction
            path_length_mm = (i - 1) * step_mm
            break

        label = volume_labels[voxel[0], voxel[1], voxel[2]]

        if label == 0:
            exit_point_mm = pos_mm
            path_length_mm = i * step_mm
            break

        path_labels.append(int(label))

    return exit_point_mm, path_labels, path_length_mm


def is_direct_path(
    source_pos_mm: np.ndarray,
    view_angle_deg: float,
    volume_labels: np.ndarray,
    voxel_size_mm: float,
    max_path_mm: float = DEFAULT_MAX_PATH_MM,
    allowed_labels: Optional[Set[int]] = None,
) -> DirectPathResult:
    """Check if a view is direct-path.

    For a given view angle, we ray-march from the source toward the visible surface
    (the surface facing the camera) and check if the path traverses only soft tissue.

    View angle convention:
    - 0° = dorsal view (camera at +Z, looking toward -Z, sees dorsal surface)
    - 180° = ventral view (camera at -Z, looking toward +Z, sees ventral surface)
    - 90° = left lateral view (camera at +X, looking toward -X)
    - -90° = right lateral view (camera at -X, looking toward +X)

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position in mm.
    view_angle_deg : float
        View angle in degrees.
    volume_labels : np.ndarray
        Volume labels (XYZ order).
    voxel_size_mm : float
        Voxel size in mm.
    max_path_mm : float
        Maximum allowed path length for superficial regime.
    allowed_labels : set of int, optional
        Labels allowed along path. Default: {0, 1} (air and soft tissue).

    Returns
    -------
    DirectPathResult
        Result with is_direct, reason, and path details.
    """
    if allowed_labels is None:
        allowed_labels = SOFT_TISSUE_LABELS | {AIR_LABEL}

    R = rotation_matrix_y(view_angle_deg)
    surface_normal = R @ np.array([0, 0, 1])

    exit_point, path_labels, path_length = find_surface_in_direction(
        source_pos_mm, surface_normal, volume_labels, voxel_size_mm
    )

    if exit_point is None:
        return DirectPathResult(
            is_direct=False,
            reason="No surface exit point found",
            entry_point_mm=None,
            exit_point_mm=None,
            path_labels=set(path_labels),
            path_length_mm=path_length,
            n_steps=len(path_labels),
        )

    unique_labels = set(path_labels)
    forbidden_labels = unique_labels - allowed_labels

    if forbidden_labels:
        organ_names = {
            2: "bone",
            3: "brain",
            4: "heart",
            5: "stomach",
            6: "abdominal",
            7: "liver",
            8: "kidney",
            9: "lung",
        }
        organs = [organ_names.get(l, f"label_{l}") for l in forbidden_labels if l > 1]
        reason = f"Through-organ path: traverses {', '.join(organs)}"
        is_direct = False
    elif path_length > max_path_mm:
        reason = f"Path {path_length:.1f}mm exceeds superficial regime {max_path_mm}mm"
        is_direct = False
    else:
        reason = "Direct path verified"
        is_direct = True

    return DirectPathResult(
        is_direct=is_direct,
        reason=reason,
        entry_point_mm=source_pos_mm.copy(),
        exit_point_mm=exit_point,
        path_labels=unique_labels,
        path_length_mm=path_length,
        n_steps=len(path_labels),
    )


def assert_direct_path(
    source_pos_mm: np.ndarray,
    view_angle_deg: float,
    volume_labels: np.ndarray,
    voxel_size_mm: float,
    max_path_mm: float = DEFAULT_MAX_PATH_MM,
    allowed_labels: Optional[Set[int]] = None,
) -> DirectPathResult:
    """Assert that a view is direct-path, raising exception if not.

    Parameters
    ----------
    Same as is_direct_path.

    Returns
    -------
    DirectPathResult
        Result with path details.

    Raises
    ------
    ValueError
        If view is not direct-path.
    """
    result = is_direct_path(
        source_pos_mm,
        view_angle_deg,
        volume_labels,
        voxel_size_mm,
        max_path_mm,
        allowed_labels,
    )

    if not result.is_direct:
        raise ValueError(
            f"Non-direct-path view rejected: {result.reason}\n"
            f"  Source: {source_pos_mm} mm\n"
            f"  View angle: {view_angle_deg}°\n"
            f"  Path length: {result.path_length_mm:.1f} mm\n"
            f"  Labels: {result.path_labels}"
        )

    return result


def get_direct_views_for_source(
    source_pos_mm: np.ndarray,
    view_angles_deg: List[float],
    volume_labels: np.ndarray,
    voxel_size_mm: float,
    max_path_mm: float = DEFAULT_MAX_PATH_MM,
) -> List[Tuple[float, DirectPathResult]]:
    """Filter view angles to only direct-path views.

    Parameters
    ----------
    source_pos_mm : np.ndarray
        Source position in mm.
    view_angles_deg : list of float
        Candidate view angles.
    volume_labels : np.ndarray
        Volume labels.
    voxel_size_mm : float
        Voxel size.
    max_path_mm : float
        Maximum path length.

    Returns
    -------
    list of (angle, result) tuples
        Only direct-path views with their results.
    """
    direct_views = []
    for angle in view_angles_deg:
        result = is_direct_path(
            source_pos_mm, angle, volume_labels, voxel_size_mm, max_path_mm
        )
        if result.is_direct:
            direct_views.append((angle, result))
        else:
            logger.debug(f"View {angle}° excluded: {result.reason}")
    return direct_views


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    VOLUME_PATH = Path(
        "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
    )
    VOLUME_SHAPE_XYZ = (95, 100, 52)
    VOXEL_SIZE_MM = 0.4

    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)

    print("=" * 70)
    print("Direct-Path Preflight Test")
    print("=" * 70)

    print("\nP1-dorsal source (Y=10mm, Z=5.8mm near dorsal surface):")
    source_pos = np.array([-0.6, 2.4, 5.8])

    for angle in [0, 30, 60, 90, -90, 180]:
        result = is_direct_path(source_pos, angle, volume, VOXEL_SIZE_MM)
        status = "✓ DIRECT" if result.is_direct else "✗ EXCLUDED"
        path_len = (
            f"{result.path_length_mm:.1f}mm" if result.path_length_mm > 0 else "N/A"
        )
        print(f"  {angle:>4}°: {status:<12} path={path_len:<8} {result.reason}")

    print("\nP5-ventral source (Y=10mm, Z=-3.8mm near ventral surface):")
    source_pos_ventral = np.array([-0.6, 2.4, -3.8])

    for angle in [0, 30, 60, 90, -90, 180]:
        result = is_direct_path(source_pos_ventral, angle, volume, VOXEL_SIZE_MM)
        status = "✓ DIRECT" if result.is_direct else "✗ EXCLUDED"
        path_len = (
            f"{result.path_length_mm:.1f}mm" if result.path_length_mm > 0 else "N/A"
        )
        print(f"  {angle:>4}°: {status:<12} path={path_len:<8} {result.reason}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  P1-dorsal: 0° should be DIRECT (source on dorsal surface)")
    print("  P1-dorsal: 180° should be EXCLUDED (through body)")
    print("  P5-ventral: 180° should be DIRECT (source on ventral surface)")
    print("  P5-ventral: 0° should be EXCLUDED (through body)")
    print("=" * 70)
