"""
TurntableCamera: Orthographic camera model for FMT turntable imaging.

Supports turntable rotation around Y-axis, hemisphere platform occlusion,
and surface normal visibility determination. Used for MCX projection channel
and DE channel partial surface extraction.

Coordinate system (FMT-SimGen):
- X: left(-19mm) → right(+19mm), range [0, 38] mm
- Y: anterior/head(high Y) → posterior/tail, range [0, 99] mm
- Z: ventral/belly(-Z) → dorsal/back(+Z), range [0, 21] mm
- Rotation axis = Y axis (head-tail direction)
- View direction rotates in XZ plane: view_dir = (sin(θ), 0, cos(θ))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TurntableCamera:
    """Orthographic camera for turntable FMT imaging.

    Parameters
    ----------
    config : dict
        View configuration with keys:
        - angles: list of int, rotation angles in degrees
        - pose: str, "prone" or "supine"
        - camera_distance_mm: float, camera to turntable distance
        - detector_resolution: list of int, [width, height] in pixels
        - projection_type: str, "orthographic" (only supported)
        - platform_occlusion: bool, whether to apply platform occlusion
    """

    def __init__(self, config: dict) -> None:
        self.angles: list[int] = config.get("angles", [-90, -60, -30, 0, 30, 60, 90])
        self.pose: str = config.get("pose", "prone")
        self.camera_distance_mm: float = config.get("camera_distance_mm", 200.0)
        self.detector_resolution: tuple[int, int] = tuple(config.get("detector_resolution", [256, 256]))
        self.projection_type: str = config.get("projection_type", "orthographic")
        self.platform_occlusion: bool = config.get("platform_occlusion", True)

        # FOV in mm for orthographic projection
        self.fov_mm: float = config.get("fov_mm", 50.0)

        # Pixel size in mm
        self.pixel_size_mm: float = self.fov_mm / self.detector_resolution[0]

        # Platform occlusion threshold (Z coordinate of platform plane)
        # For prone: platform at -Z (ventral), occludes Z < z_center
        # For supine: platform at +Z (dorsal), occludes Z > z_center
        self.z_center: float = config.get("platform_z_center", 10.5)

        logger.info(
            f"TurntableCamera: pose={self.pose}, distance={self.camera_distance_mm}mm, "
            f"detector={self.detector_resolution}, fov={self.fov_mm}mm"
        )

    def compute_surface_normals(
        self,
        nodes: np.ndarray,
        surface_faces: np.ndarray,
    ) -> np.ndarray:
        """Compute outward-pointing surface normals at each surface node.

        Face normals are computed and averaged at nodes. The direction is
        ensured to point outward from the mesh using the mesh centroid
        as reference.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3] in mm.
        surface_faces : np.ndarray
            Surface triangle indices [F×3].

        Returns
        -------
        np.ndarray
            Normalized outward normals at each node [N×3].
            Only surface nodes have non-zero normals.
        """
        num_nodes = nodes.shape[0]
        normals = np.zeros_like(nodes)
        counts = np.zeros(num_nodes, dtype=np.int32)

        # Compute face normals (non-normalized)
        for face in surface_faces:
            n0, n1, n2 = face
            v0 = nodes[n0]
            v1 = nodes[n1]
            v2 = nodes[n2]

            # Edge vectors
            e1 = v1 - v0
            e2 = v2 - v0

            # Cross product gives outward normal for CCW-wound faces
            face_normal = np.cross(e1, e2)

            # Accumulate at each node
            normals[n0] += face_normal
            normals[n1] += face_normal
            normals[n2] += face_normal
            counts[n0] += 1
            counts[n1] += 1
            counts[n2] += 1

        # Normalize by averaging
        for i in range(num_nodes):
            if counts[i] > 0:
                normals[i] /= counts[i]

        # Flip normals to point outward using world +Z as reference for prone pose.
        # For prone imaging (dorsal at +Z), outward normals on dorsal surface
        # should point in +Z direction. If dot(normal, +Z) < 0, flip.
        for i in range(num_nodes):
            if counts[i] > 0 and normals[i, 2] < 0:
                normals[i] = -normals[i]

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        return normals

    def get_view_direction(self, angle_deg: float) -> np.ndarray:
        """Get unit view direction vector for a given rotation angle.

        The view direction lies in the XZ plane and rotates around Y axis.
        At 0°, view from +Z direction (looking down at dorsal side).
        At 90°, view from +X direction (looking at lateral side).

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees.

        Returns
        -------
        np.ndarray
            Unit view direction vector [3].
        """
        angle_rad = np.deg2rad(angle_deg)
        return np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float64)

    def _is_occluded_by_platform(self, node_coords: np.ndarray) -> np.ndarray:
        """Determine if nodes are occluded by the platform.

        Parameters
        ----------
        node_coords : np.ndarray
            Node coordinates [N×3].

        Returns
        -------
        np.ndarray
            Boolean array [N], True if occluded by platform.
        """
        z = node_coords[:, 2]

        if self.pose == "prone":
            # Platform at -Z (ventral), occlude nodes below center
            return z < self.z_center
        else:  # supine
            # Platform at +Z (dorsal), occlude nodes above center
            return z > self.z_center

    def get_visible_surface_nodes(
        self,
        angle_deg: float,
        node_coords: np.ndarray,
        node_normals: np.ndarray,
    ) -> np.ndarray:
        """Get indices of surface nodes visible from a given angle.

        A node is visible if:
        1. Its normal dot view direction > 0 (facing camera)
        2. It is not occluded by the platform

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees.
        node_coords : np.ndarray
            All node coordinates [N×3] in mm.
        node_normals : np.ndarray
            Surface normals [N×3], only surface nodes are non-zero.

        Returns
        -------
        np.ndarray
            Indices of visible surface nodes [V_visible].
        """
        view_dir = self.get_view_direction(angle_deg)

        # Normal-facing check: dot(normal, view_dir) > 0 means node faces camera
        facing_camera = np.dot(node_normals, view_dir) > 0

        # Platform occlusion check
        if self.platform_occlusion:
            occluded = self._is_occluded_by_platform(node_coords)
        else:
            occluded = np.zeros(len(node_coords), dtype=bool)

        # Surface node mask (non-zero normals)
        is_surface = np.linalg.norm(node_normals, axis=1) > 1e-6

        # Combined visibility
        visible = is_surface & facing_camera & ~occluded

        return np.where(visible)[0].astype(np.int32)

    def project_points(
        self,
        points_3d: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        """Project 3D points to 2D detector coordinates (orthographic).

        Uses orthographic projection with the camera looking along
        the view direction. The detector plane is perpendicular to view_dir.

        Parameters
        ----------
        points_3d : np.ndarray
            3D points [N×3] in mm.
        angle_deg : float
            Rotation angle in degrees.

        Returns
        -------
        np.ndarray
            2D detector coordinates [N×2] in pixels.
            Coordinates are in pixel space with origin at detector center.
        """
        view_dir = self.get_view_direction(angle_deg)

        # Build camera coordinate basis
        # u: horizontal on detector (perpendicular to view_dir in XZ)
        # v: vertical on detector (upward = +Z)
        if abs(view_dir[2]) < 0.999:
            u = np.cross(view_dir, np.array([0.0, 1.0, 0.0]))
        else:
            u = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
        u = u / np.linalg.norm(u)
        v = np.array([0.0, 0.0, 1.0])  # Detector vertical = world Z

        # Project: compute coordinates in detector plane
        # For orthographic, we simply take dot products with basis vectors
        x_proj = np.dot(points_3d, u)
        y_proj = np.dot(points_3d, v)

        # Convert to pixel coordinates (origin at center)
        pixels_u = x_proj / self.pixel_size_mm + self.detector_resolution[0] / 2
        pixels_v = y_proj / self.pixel_size_mm + self.detector_resolution[1] / 2

        return np.stack([pixels_u, pixels_v], axis=1)

    def project_volume(
        self,
        volume_3d: np.ndarray,
        angle_deg: float,
        voxel_size_mm: float = 0.2,
        origin: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project a 3D fluence volume to a 2D detector image.

        For each pixel in the detector, casts a ray through the volume
        along the view direction and accumulates the first encountered
        non-zero fluence value (nearest surface).

        Parameters
        ----------
        volume_3d : np.ndarray
            3D fluence volume [X×Y×Z]. Shape depends on cropping and
            downsampling from atlas.
        angle_deg : float
            Rotation angle in degrees.
        voxel_size_mm : float
            Voxel size in mm (default 0.2 from MCX config).
        origin : Optional[np.ndarray]
            Physical origin of volume in mm [3]. If None, assumes
            volume starts at (0, 0, 0).

        Returns
        -------
        np.ndarray
            2D detector image [H×W] in float32.
        """
        proj_h, proj_w = self.detector_resolution[1], self.detector_resolution[0]
        projection = np.zeros((proj_h, proj_w), dtype=np.float32)

        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])

        vol_shape = np.array(volume_3d.shape)

        # Compute detector pixel centers in world coordinates
        det_h, det_w = proj_h, proj_w
        x_range = np.arange(det_w) - det_w / 2 + 0.5
        y_range = np.arange(det_h) - det_h / 2 + 0.5

        view_dir = self.get_view_direction(angle_deg)

        # Camera basis vectors
        if abs(view_dir[2]) < 0.999:
            u = np.cross(view_dir, np.array([0.0, 1.0, 0.0]))
        else:
            u = np.cross(view_dir, np.array([1.0, 0.0, 0.0]))
        u = u / np.linalg.norm(u)
        v = np.array([0.0, 0.0, 1.0])

        # Position of camera in world coords
        camera_pos = -view_dir * self.camera_distance_mm

        # March through detector pixels
        for i, dy in enumerate(y_range):
            for j, dx in enumerate(x_range):
                # Pixel center in world coords
                pixel_world = camera_pos + dx * self.pixel_size_mm * u + dy * self.pixel_size_mm * v

                # Ray direction (from camera through pixel)
                ray_dir = pixel_world - camera_pos
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # Ray-box intersection to find entry/exit
                t_min, t_max = self._ray_aabb_intersect(
                    pixel_world, ray_dir, origin, origin + vol_shape * voxel_size_mm
                )

                if t_min is None or t_max < 0:
                    continue

                # March along ray in volume
                t = max(t_min, 0)
                step = voxel_size_mm * 0.5  # Sub-voxel stepping

                while t < t_max:
                    world_pt = pixel_world + t * ray_dir
                    voxel_idx = ((world_pt - origin) / voxel_size_mm).astype(np.int32)

                    if (0 <= voxel_idx[0] < volume_3d.shape[0] and
                        0 <= voxel_idx[1] < volume_3d.shape[1] and
                        0 <= voxel_idx[2] < volume_3d.shape[2]):
                        fluence = volume_3d[voxel_idx[0], voxel_idx[1], voxel_idx[2]]
                        if fluence > 0:
                            projection[i, j] = fluence
                            break
                    t += step

        return projection

    def _ray_aabb_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> tuple[Optional[float], Optional[float]]:
        """Ray-axis-aligned bounding box intersection.

        Returns
        -------
        tuple
            (t_min, t_max) or (None, None) if no intersection.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (box_min - origin) / direction
            t2 = (box_max - origin) / direction

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        t_enter = np.max(t_min)
        t_exit = np.min(t_max)

        if t_exit < 0 or t_enter > t_exit:
            return None, None

        return t_enter, t_exit

    def get_all_visible_nodes_per_angle(
        self,
        node_coords: np.ndarray,
        node_normals: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Compute visible surface nodes for all configured angles.

        Parameters
        ----------
        node_coords : np.ndarray
            Node coordinates [N×3] in mm.
        node_normals : np.ndarray
            Surface normals [N×3].

        Returns
        -------
        dict
            Mapping from angle (int) to visible node indices array.
        """
        results = {}
        for angle in self.angles:
            visible = self.get_visible_surface_nodes(angle, node_coords, node_normals)
            results[angle] = visible
            logger.info(f"Angle {angle:4d}°: {len(visible)} visible nodes")
        return results
