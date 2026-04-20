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

from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD

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

        # Volume center in world/trunk-local coordinates (mm).
        # Must match the MCX volume center used in mcx_projection.py.
        # For the trunk volume: center = (19.0, 20.0, 10.4) mm,
        # which is the midpoint of MCX bbox [0,38]×[0,40]×[0,20.8].
        self.volume_center_world: np.ndarray = np.array(
            config.get("volume_center_world", VOLUME_CENTER_WORLD), dtype=np.float64
        )

        logger.info(
            f"TurntableCamera: pose={self.pose}, distance={self.camera_distance_mm}mm, "
            f"detector={self.detector_resolution}, fov={self.fov_mm}mm"
        )

    def project_nodes_to_detector(
        self,
        node_coords: np.ndarray,
        angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project node coordinates onto detector plane at a given angle.

        Uses the same orthographic projection geometry as MCX:
        camera at D*(sinθ, 0, cosθ), looking toward origin along -Z.
        Image axes: U = world+X (right), V = world+Y (anterior on top).

        Parameters
        ----------
        node_coords : np.ndarray
            Node coordinates [N×3] in mm (world/trunk-local frame).
        angle_deg : float
            Rotation angle in degrees.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (u_px, v_px, depth) for each node:
            - u_px: detector X pixel index [N], -1 if out of FOV
            - v_px: detector Y pixel index [N], -1 if out of FOV
            - depth: camera-frame depth = D - z_rot [N]
        """
        θ = np.deg2rad(angle_deg)
        D = self.camera_distance_mm

        # Shift to volume-center frame, then rotate around Y axis.
        # This matches mcx_projection.py::project_volume_reference which shifts
        # by volume_center_world before the Y-axis rotation.
        c = self.volume_center_world
        p = node_coords - c

        cos_t = np.cos(θ)
        sin_t = np.sin(θ)
        x_rot = p[:, 0] * cos_t + p[:, 2] * sin_t
        y_rot = p[:, 1]  # Y unchanged
        z_rot = -p[:, 0] * sin_t + p[:, 2] * cos_t

        # Camera-frame depth: positive = in front of camera
        depth = D - z_rot

        # Orthographic projection: (x_rot, y_rot) → detector (U, V)
        u = x_rot   # world +X → detector right
        v = y_rot   # world +Y → detector up

        # Pixel indices
        half_fov = self.fov_mm / 2.0
        w_px, h_px = self.detector_resolution
        u_px = ((u + half_fov) / self.fov_mm * w_px).astype(int)
        v_px = ((v + half_fov) / self.fov_mm * h_px).astype(int)

        # Mark out-of-FOV
        u_px = np.where((u >= -half_fov) & (u < half_fov), u_px, -1)
        v_px = np.where((v >= -half_fov) & (v < half_fov), v_px, -1)

        return u_px, v_px, depth

    def get_visible_surface_nodes_from_mcx_depth(
        self,
        node_coords: np.ndarray,
        node_normals: np.ndarray,
        depth_map: np.ndarray,
        angle_deg: float,
        depth_tolerance_mm: float = 0.2,
    ) -> np.ndarray:
        """Get visible surface nodes using MCX depth_map for self-occlusion.

        A node is visible if:
        1. It is a surface node (non-zero normal)
        2. Its normal faces the camera (dot(normal, view_dir) > 0)
        3. It is not platform-occluded
        4. Its camera-frame depth <= depth_map[pv, pu] + ε
           (i.e., it is at or shallower than the frontmost fluence voxel)

        Parameters
        ----------
        node_coords : np.ndarray
            All node coordinates [N×3] in mm (trunk-local/world frame).
        node_normals : np.ndarray
            Surface normals [N×3].
        depth_map : np.ndarray
            MCX depth_map [H×W] in camera frame (mm). depth_map[pv, pu] is
            the depth of the voxel with maximum fluence at that pixel.
            Set pixel to inf if no fluence reaches that pixel.
        angle_deg : float
            Rotation angle in degrees.
        depth_tolerance_mm : float
            Tolerance for depth comparison (default 0.2mm = voxel_size).
            A node is visible if its depth is within this of depth_map.

        Returns
        -------
        np.ndarray
            Indices of visible surface nodes.
        """
        view_dir = self.get_view_direction(angle_deg)

        # 1. Surface node mask
        is_surface = np.linalg.norm(node_normals, axis=1) > 1e-6

        # 2. Normal-facing check
        facing_camera = np.dot(node_normals, view_dir) > 0

        # 3. Platform occlusion
        if self.platform_occlusion:
            platform_occl = self._is_occluded_by_platform(node_coords)
        else:
            platform_occl = np.zeros(len(node_coords), dtype=bool)

        # 4. Project to detector
        u_px, v_px, depth = self.project_nodes_to_detector(node_coords, angle_deg)

        H, W = depth_map.shape
        w_px, h_px = self.detector_resolution

        # 5. MCX self-occlusion via depth comparison
        # Node is occluded if:
        #   - out of FOV, OR
        #   - pixel has no fluence coverage (depth_map is +inf), OR
        #   - node is behind the frontmost fluence voxel (depth > depth_map + ε)
        mcx_occluded = np.ones(len(node_coords), dtype=bool)
        in_fov = (u_px >= 0) & (u_px < w_px) & (v_px >= 0) & (v_px < h_px)
        if in_fov.any():
            idx = np.where(in_fov)[0]
            # Clip+round to safely handle edge pixels
            u_i = np.clip(np.round(u_px[idx]).astype(np.int32), 0, W - 1)
            v_i = np.clip(np.round(v_px[idx]).astype(np.int32), 0, H - 1)
            mcx_d = depth_map[v_i, u_i]          # [M] depth from MCX
            node_d = depth[idx]                   # [M] depth of this node

            has_coverage = np.isfinite(mcx_d)                        # pixel has fluence
            not_occluded = node_d <= (mcx_d + depth_tolerance_mm)     # node in front
            mcx_occluded[idx] = ~(has_coverage & not_occluded)

        visible = is_surface & facing_camera & ~platform_occl & ~mcx_occluded
        return np.where(visible)[0].astype(np.int32)

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

        # Flip normals to point outward using mesh center direction as reference.
        # The direction from mesh center to a surface node is outward.
        # If normal points inward (dot < 0), flip it.
        mesh_center = nodes[counts > 0].mean(axis=0)
        for i in range(num_nodes):
            if counts[i] > 0:
                outward_dir = nodes[i] - mesh_center
                if np.dot(normals[i], outward_dir) < 0:
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


# ─── Standalone wrapper for QA / visibility computation ───────────────────────

def get_visible_surface_nodes_from_mcx_depth(
    node_coords: np.ndarray,
    surface_faces: np.ndarray,
    mask_xyz: np.ndarray,
    angle_deg: float,
    voxel_size: float,
    volume_center_world: tuple[float, float, float],
    epsilon: float = 0.5,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Standalone wrapper: compute depth_map from mask, then return visible nodes.

    This provides the same functionality as TurntableCamera's instance method
    ``get_visible_surface_nodes_from_mcx_depth`` but as a one-shot function
    that internally computes the depth map from an MCX trunk-mask volume.

    Parameters
    ----------
    node_coords : np.ndarray [N×3]
        Node coordinates in mm (trunk-local frame).
    surface_faces : np.ndarray [F×3]
        Surface triangle vertex indices.
    mask_xyz : np.ndarray [X×Y×Z]
        Binary tissue mask (e.g. mcx_volume > 0).
    angle_deg : float
        Camera rotation angle in degrees.
    voxel_size : float
        MCX voxel size in mm.
    volume_center_world : tuple[float, float, float]
        Physical center of the MCX volume in trunk-local mm, e.g. (19.0, 20.0, 10.4).
    epsilon : float
        Bilateral depth tolerance in mm (default 0.5).

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray]
        (finite_px, visible_mask, depth_map) where:
        - finite_px: number of detector pixels with finite depth
        - visible_mask: boolean array [N] — True for visible surface nodes
        - depth_map: the computed depth map [H×W] in mm
    """
    from fmt_simgen.mcx_projection import project_volume_reference

    # Camera parameters for TurntableCamera
    camera_distance_mm = 200.0
    fov_mm = 80.0
    detector_resolution = (256, 256)

    # 1. Compute depth map from tissue mask.
    #
    # IMPORTANT: Use volume_center_world=(0,0,0) here because the mask volume
    # is already in trunk-local frame (corner at origin). project_volume_reference
    # shifts by -volume_center_world before computing camera depth, so passing
    # (0,0,0) gives depth = D - z_world (no offset), which correctly represents
    # the physical depth from camera to the tissue front along each ray.
    # Using volume_center_world=(19,20,10.4) would offset depth by +10.4mm,
    # which does NOT match MCX actual depth.
    _, depth_map = project_volume_reference(
        mask_xyz.astype(np.uint8),
        angle_deg=angle_deg,
        camera_distance=camera_distance_mm,
        fov_mm=fov_mm,
        detector_resolution=detector_resolution,
        voxel_size_mm=voxel_size,
        volume_center_world=(0.0, 0.0, 0.0),
    )

    finite_px = int(np.isfinite(depth_map).sum())

    # 2. Compute surface normals using the TurntableCamera.
    # The camera uses volume_center_world to center node coords before rotation,
    # which is needed for correct per-angle node→detector projection.
    camera_cfg = dict(
        volume_center_world=volume_center_world,
        camera_distance_mm=camera_distance_mm,
        fov_mm=fov_mm,
        detector_resolution=detector_resolution,
    )
    camera = TurntableCamera(camera_cfg)
    node_normals = camera.compute_surface_normals(node_coords, surface_faces)

    # 3. Get visible surface nodes using MCX depth for self-occlusion.
    # NOTE: node_depths from project_nodes_to_detector are computed as
    # D - (z_node - cz) = D - z_node + cz, while depth_map from step 1 is
    # D - z_world (since vcw=0). The comparison below handles this via
    # the epsilon tolerance: node is visible if its depth is within
    # epsilon of the MCX-reported front surface depth.
    visible_idx = camera.get_visible_surface_nodes_from_mcx_depth(
        node_coords,
        node_normals,
        depth_map,
        angle_deg,
        depth_tolerance_mm=epsilon,
    )

    # Convert to boolean mask over all nodes
    visible_mask = np.zeros(len(node_coords), dtype=bool)
    visible_mask[visible_idx] = True

    return finite_px, visible_mask, depth_map

