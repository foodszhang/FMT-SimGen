#!/usr/bin/env python3
"""Real surface data loader for E1d.

Loads surface mesh from Digimouse atlas and provides local depth computation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SurfaceData:
    """Container for real surface data."""

    nodes: np.ndarray
    surface_node_indices: np.ndarray
    surface_faces: np.ndarray
    surface_coords: np.ndarray
    surface_z_range: Tuple[float, float]


def load_real_surface(
    mesh_path: str = "output/shared/mesh.npz",
) -> SurfaceData:
    """Load real surface from mesh file.

    Args:
        mesh_path: path to mesh.npz file

    Returns:
        SurfaceData with surface coordinates
    """
    data = np.load(mesh_path)

    nodes = data["nodes"]
    surface_node_indices = data["surface_node_indices"]
    surface_faces = data["surface_faces"]

    surface_coords = nodes[surface_node_indices]

    z_min = surface_coords[:, 2].min()
    z_max = surface_coords[:, 2].max()

    return SurfaceData(
        nodes=nodes,
        surface_node_indices=surface_node_indices,
        surface_faces=surface_faces,
        surface_coords=surface_coords,
        surface_z_range=(z_min, z_max),
    )


def compute_local_depth(
    source_point: np.ndarray,
    surface_coords: np.ndarray,
) -> np.ndarray:
    """Compute local depth from source to each surface point.

    For each surface point, the depth is computed as the z-distance
    from the surface point to the source.

    Args:
        source_point: [3] source position in mm
        surface_coords: [N, 3] surface node coordinates

    Returns:
        depths: [N] depth values in mm
    """
    source_z = source_point[2]
    surface_z = surface_coords[:, 2]

    depths = surface_z - source_z

    return depths


def project_source_to_surface(
    source_point: np.ndarray,
    surface_coords: np.ndarray,
    surface_faces: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Find the closest point on surface mesh to source XY projection.

    Args:
        source_point: [3] source position
        surface_coords: [N, 3] surface node coordinates
        surface_faces: [M, 3] surface triangle indices

    Returns:
        projection_xy: [2] XY coordinates of projection
        surface_z_at_projection: z coordinate at projection
    """
    source_xy = source_point[:2]
    surface_xy = surface_coords[:, :2]

    distances = np.linalg.norm(surface_xy - source_xy, axis=1)
    closest_idx = np.argmin(distances)

    projection_xy = surface_xy[closest_idx]
    surface_z_at_projection = surface_coords[closest_idx, 2]

    return projection_xy, surface_z_at_projection


def get_surface_nodes_in_roi(
    source_center: np.ndarray,
    surface_coords: np.ndarray,
    roi_radius_mm: float,
) -> np.ndarray:
    """Get surface node indices within ROI of source.

    Args:
        source_center: [3] source position
        surface_coords: [N, 3] surface coordinates
        roi_radius_mm: ROI radius in mm

    Returns:
        indices: indices of surface nodes in ROI
    """
    source_xy = source_center[:2]
    surface_xy = surface_coords[:, :2]

    distances = np.linalg.norm(surface_xy - source_xy, axis=1)

    indices = np.where(distances <= roi_radius_mm)[0]

    return indices
