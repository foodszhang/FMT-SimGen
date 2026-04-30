"""
Frame contract re-exports.

All frame-related constants are defined in fmt_simgen.frame_contract.
This module re-exports them for import convenience.
"""
from fmt_simgen.frame_contract import (
    VOXEL_SIZE_MM,
    TRUNK_GRID_SHAPE,
    VOLUME_EXTENTS_MM,
    VOLUME_CENTER_WORLD,
    CAMERA_DISTANCE_MM,
    FOV_MM,
    DETECTOR_RESOLUTION,
    ANGLES,
    world_to_volume_voxel,
    volume_voxel_to_world,
    assert_in_trunk_bbox,
)

__all__ = [
    "VOXEL_SIZE_MM",
    "TRUNK_GRID_SHAPE",
    "VOLUME_EXTENTS_MM",
    "VOLUME_CENTER_WORLD",
    "CAMERA_DISTANCE_MM",
    "FOV_MM",
    "DETECTOR_RESOLUTION",
    "ANGLES",
    "world_to_volume_voxel",
    "volume_voxel_to_world",
    "assert_in_trunk_bbox",
]
