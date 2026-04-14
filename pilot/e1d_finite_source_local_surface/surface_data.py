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


def compute_surface_normals(
    surface_coords: np.ndarray,
    surface_faces: np.ndarray,
    surface_node_indices: Optional[np.ndarray] = None,
    normalize: bool = True,
    consistent_orientation: bool = True,
) -> np.ndarray:
    """Compute surface normals from mesh triangles.

    For each node, the normal is computed as the average of adjacent face normals.

    Args:
        surface_coords: [N, 3] surface node coordinates
        surface_faces: [M, 3] triangle indices
        surface_node_indices: [N] mapping from surface_coords index to global node index.
                              If None, assumes surface_faces indices are directly into surface_coords.
        normalize: whether to normalize normals to unit length
        consistent_orientation: whether to ensure normals point outward (positive Z bias)

    Returns:
        normals: [N, 3] surface normal vectors
    """
    n_nodes = surface_coords.shape[0]
    normals = np.zeros((n_nodes, 3), dtype=np.float32)

    if surface_node_indices is not None:
        global_to_surface = {idx: i for i, idx in enumerate(surface_node_indices)}

        v0 = surface_coords[[global_to_surface.get(f, 0) for f in surface_faces[:, 0]]]
        v1 = surface_coords[[global_to_surface.get(f, 0) for f in surface_faces[:, 1]]]
        v2 = surface_coords[[global_to_surface.get(f, 0) for f in surface_faces[:, 2]]]
    else:
        v0 = surface_coords[surface_faces[:, 0]]
        v1 = surface_coords[surface_faces[:, 1]]
        v2 = surface_coords[surface_faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    face_normals = np.cross(edge1, edge2)

    face_areas = np.linalg.norm(face_normals, axis=1)
    face_areas = np.maximum(face_areas, 1e-10)

    face_normals_normalized = face_normals / face_areas[:, np.newaxis]

    if surface_node_indices is not None:
        global_to_surface = {idx: i for i, idx in enumerate(surface_node_indices)}
        for i in range(len(surface_faces)):
            for j in range(3):
                global_idx = surface_faces[i, j]
                if global_idx in global_to_surface:
                    surface_idx = global_to_surface[global_idx]
                    normals[surface_idx] += face_normals_normalized[i]
    else:
        for i in range(len(surface_faces)):
            for j in range(3):
                node_idx = surface_faces[i, j]
                if node_idx < n_nodes:
                    normals[node_idx] += face_normals_normalized[i]

    if normalize:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals = normals / norms

    if consistent_orientation:
        z_component = normals[:, 2]
        negative_z_ratio = np.mean(z_component < 0)
        if negative_z_ratio > 0.5:
            normals = -normals

    return normals


def extract_roi_patch(
    surface_coords: np.ndarray,
    surface_normals: np.ndarray,
    source_center: np.ndarray,
    roi_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ROI patch from surface around source.

    Args:
        surface_coords: [N, 3] all surface node coordinates
        surface_normals: [N, 3] surface normals
        source_center: [3] source position
        roi_radius_mm: ROI radius

    Returns:
        roi_coords: [M, 3] ROI surface coordinates
        roi_normals: [M, 3] ROI surface normals
        roi_indices: [M] indices of ROI nodes in original array
    """
    roi_indices = get_surface_nodes_in_roi(source_center, surface_coords, roi_radius_mm)

    roi_coords = surface_coords[roi_indices]
    roi_normals = surface_normals[roi_indices] if surface_normals is not None else None

    return roi_coords, roi_normals, roi_indices


def fit_local_plane(
    surface_coords: np.ndarray,
    surface_normals: np.ndarray,
    query_point: np.ndarray,
    k_neighbors: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit local plane approximation at query point.

    Uses PCA on neighboring surface points to estimate local tangent plane.

    Args:
        surface_coords: [N, 3] surface coordinates
        surface_normals: [N, 3] surface normals
        query_point: [3] point to fit plane at
        k_neighbors: number of neighbors for plane fitting

    Returns:
        plane_normal: [3] normal of fitted plane
        plane_center: [3] center of fitted plane
        plane_curvature: local curvature estimate
    """
    distances = np.linalg.norm(surface_coords - query_point, axis=1)
    neighbor_indices = np.argsort(distances)[:k_neighbors]
    neighbor_coords = surface_coords[neighbor_indices]

    plane_center = neighbor_coords.mean(axis=0)

    centered = neighbor_coords - plane_center
    cov = centered.T @ centered / k_neighbors

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    plane_normal = eigenvectors[:, 0]

    plane_curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-10)

    if surface_normals is not None:
        avg_normal = surface_normals[neighbor_indices].mean(axis=0)
        if np.dot(plane_normal, avg_normal) < 0:
            plane_normal = -plane_normal

    return plane_normal, plane_center, plane_curvature


def compute_local_frame_distances(
    surface_points_mm: np.ndarray,
    source_points_mm: np.ndarray,
    surface_normals_mm: Optional[np.ndarray] = None,
    mode: str = "local_depth",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances in local surface frame.

    Args:
        surface_points_mm: [N, 3] surface point coordinates
        source_points_mm: [M, 3] source point coordinates
        surface_normals_mm: [N, 3] surface normals (required for local_plane mode)
        mode: "local_depth" or "local_plane"

    Returns:
        depth: [N, M] depth from each surface point to each source point
        rho: [N, M] in-plane distance from each surface point to each source point
    """
    n_surf = surface_points_mm.shape[0]
    n_src = source_points_mm.shape[0]

    surface_exp = surface_points_mm[:, np.newaxis, :]
    source_exp = source_points_mm[np.newaxis, :, :]

    delta = surface_exp - source_exp

    if mode == "local_depth":
        depth = delta[:, :, 2]
        rho = np.sqrt(delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2)

    elif mode == "local_plane":
        if surface_normals_mm is None:
            raise ValueError("surface_normals_mm required for local_plane mode")

        normals_exp = surface_normals_mm[:, np.newaxis, :]

        depth = np.sum(delta * normals_exp, axis=2)

        parallel = delta - depth[:, :, np.newaxis] * normals_exp
        rho = np.linalg.norm(parallel, axis=2)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'local_depth' or 'local_plane'")

    return depth, rho


class AtlasSurfaceData:
    """Container for atlas surface data with precomputed normals."""

    def __init__(
        self,
        mesh_path: str = "output/shared/mesh.npz",
        compute_normals: bool = True,
    ):
        """Load atlas surface data.

        Args:
            mesh_path: path to mesh.npz file
            compute_normals: whether to compute surface normals
        """
        data = np.load(mesh_path)

        self.nodes = data["nodes"]
        self.surface_node_indices = data["surface_node_indices"]
        self.surface_faces = data["surface_faces"]

        self.surface_coords = self.nodes[self.surface_node_indices]

        self.surface_z_range = (
            float(self.surface_coords[:, 2].min()),
            float(self.surface_coords[:, 2].max()),
        )

        if compute_normals:
            self.surface_normals = compute_surface_normals(
                self.surface_coords,
                self.surface_faces,
                surface_node_indices=self.surface_node_indices,
            )
        else:
            self.surface_normals = None

    def get_roi(
        self,
        source_center: np.ndarray,
        roi_radius_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROI surface patch around source.

        Args:
            source_center: [3] source position
            roi_radius_mm: ROI radius

        Returns:
            coords: [M, 3] ROI coordinates
            normals: [M, 3] ROI normals (or None)
            indices: [M] ROI indices
        """
        return extract_roi_patch(
            self.surface_coords,
            self.surface_normals,
            source_center,
            roi_radius_mm,
        )

    def get_surface_z_at_point(self, xy_point: np.ndarray) -> float:
        """Get surface Z at XY projection point.

        Args:
            xy_point: [2] XY coordinates

        Returns:
            z: surface Z at closest point
        """
        distances = np.linalg.norm(self.surface_coords[:, :2] - xy_point, axis=1)
        closest_idx = np.argmin(distances)
        return float(self.surface_coords[closest_idx, 2])
