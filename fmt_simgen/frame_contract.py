"""
Canonical frame contract for FMT-SimGen.

All frame-related constants are defined here ONCE and imported everywhere.
DO NOT define frame literals (19, 20, 10.4, 30, 0.2, etc.) in any other file.

Canonical frame: mcx_trunk_local_mm (origin at volume bbox corner)
- Axes: X=right (+19mm left→right), Y=anterior (+20mm anterior→posterior), Z=dorsal (+10.4mm ventral→dorsal)
- Rotation center for projection = (19.0, 20.0, 10.4)
- Voxel grid: (190, 200, 104) voxels @ 0.2mm = (38, 40, 20.8) mm
"""
from __future__ import annotations

import numpy as np
from typing import Final

# ── Volume geometry ──────────────────────────────────────────────────────────────

VOXEL_SIZE_MM: Final[float] = 0.2
"""Voxel size for MCX volume and FEM grid in mm."""

TRUNK_GRID_SHAPE: Final[tuple[int, int, int]] = (190, 200, 104)
"""MCX/DE volume grid shape in XYZ order (X=190, Y=200, Z=104)."""

VOLUME_EXTENTS_MM: Final[np.ndarray] = np.array(
    [38.0, 40.0, 20.8], dtype=np.float64
)
"""Physical extents of the volume in mm: X×Y×Z.
Derived from TRUNK_GRID_SHAPE * VOXEL_SIZE_MM = (190*0.2, 200*0.2, 104*0.2)."""

VOLUME_CENTER_WORLD: Final[np.ndarray] = VOLUME_EXTENTS_MM / 2.0  # (19.0, 20.0, 10.4)
"""Geometric center of the volume in trunk-local mm.
Used as rotation center for orthographic projection (camera looks toward this point)."""

# ── Derived ─────────────────────────────────────────────────────────────────────────

CAMERA_DISTANCE_MM: Final[float] = 200.0
"""Camera-to-rotation-center distance for turntable projection."""

FOV_MM: Final[float] = 80.0
"""Field of view for turntable detector in mm (square detector)."""

DETECTOR_RESOLUTION: Final[tuple[int, int]] = (256, 256)
"""Turntable detector resolution: (width, height) in pixels."""

ANGLES: Final[list[int]] = [-90, -60, -30, 0, 30, 60, 90]
"""Turntable rotation angles in degrees."""


# ── Conversion helpers ────────────────────────────────────────────────────────────

def world_to_volume_voxel(
    world_coords: np.ndarray,
    voxel_size: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Convert trunk-local world mm to integer voxel indices.

    Assumes world origin is at trunk bbox corner (voxel [0,0,0] center = voxel_size/2 mm).
    """
    return np.floor(world_coords / voxel_size).astype(np.int32)


def volume_voxel_to_world(
    voxel_indices: np.ndarray,
    voxel_size: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Convert integer voxel indices to trunk-local world mm (voxel centers)."""
    return (voxel_indices + 0.5) * voxel_size


def assert_in_trunk_bbox(
    coords: np.ndarray,
    tol_mm: float = 3.0,
) -> None:
    """Assert that coordinates are within the trunk bounding box (with tolerance).

    Raises
    ------
    AssertionError
        If any coordinate is outside trunk bbox ± tol_mm.
    """
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    assert (lo >= -tol_mm).all(), (
        f"Coordinates below trunk bbox: min={lo}, expected >={-tol_mm}"
    )
    assert (hi <= VOLUME_EXTENTS_MM + tol_mm).all(), (
        f"Coordinates above trunk bbox: max={hi}, expected <={VOLUME_EXTENTS_MM + tol_mm}"
    )
