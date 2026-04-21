"""
Frame contract runtime assertions (H3).

These raise FrameContractViolation — never warn.
All assertions validate against fmt_simgen.frame_contract constants.
"""
from __future__ import annotations

import numpy as np

from fmt_simgen.frame_contract import (
    TRUNK_OFFSET_ATLAS_MM,
    TRUNK_SIZE_MM,
    VOXEL_SIZE_MM,
    TRUNK_GRID_SHAPE,
    VOLUME_CENTER_WORLD,
    assert_in_trunk_bbox as _assert_in_trunk_bbox,
)


class FrameContractViolation(AssertionError):
    """Raised when a frame contract invariant is violated."""
    pass


def assert_vcw(vcw, label: str = "") -> None:
    """Assert volume_center_world matches TRUNK_SIZE_MM / 2."""
    vcw_arr = np.asarray(vcw)
    if not np.allclose(vcw_arr, VOLUME_CENTER_WORLD):
        raise FrameContractViolation(
            f"{label}: volume_center_world={vcw_arr.tolist()} "
            f"!= expected {VOLUME_CENTER_WORLD.tolist()} "
            f"(TRUNK_SIZE_MM / 2 = {TRUNK_SIZE_MM.tolist()})"
        )


def assert_projection_contract(
    vcw,
    voxel_size_mm: float,
    label: str = "",
) -> None:
    """Assert projection parameters match the frame contract."""
    assert_vcw(vcw, label)
    if not np.isclose(voxel_size_mm, VOXEL_SIZE_MM):
        raise FrameContractViolation(
            f"{label}: voxel_size_mm={voxel_size_mm} "
            f"!= VOXEL_SIZE_MM={VOXEL_SIZE_MM}"
        )


def assert_focus_in_trunk(focus_center, tol_mm: float = 1.0) -> None:
    """Assert a focus center is inside the trunk bounding box."""
    center = np.asarray(focus_center).reshape(1, 3)
    try:
        _assert_in_trunk_bbox(center, tol_mm=tol_mm)
    except AssertionError as e:
        raise FrameContractViolation(
            f"Focus center {center.flatten().tolist()} outside trunk bbox "
            f"[0, {TRUNK_SIZE_MM[0]}], [0, {TRUNK_SIZE_MM[1]}], [0, {TRUNK_SIZE_MM[2]}] "
            f"(tol_mm={tol_mm})"
        ) from e


def assert_mcx_volume_shape(volume_shape_xyz, label: str = "") -> None:
    """Assert MCX volume shape matches TRUNK_GRID_SHAPE."""
    if tuple(volume_shape_xyz) != TRUNK_GRID_SHAPE:
        raise FrameContractViolation(
            f"{label}: volume_shape_xyz={volume_shape_xyz} "
            f"!= TRUNK_GRID_SHAPE={TRUNK_GRID_SHAPE}"
        )


def assert_volume_center_world(volume_center_world, label: str = "") -> None:
    """Alias for assert_vcw — backwards compatible name."""
    assert_vcw(volume_center_world, label)