"""
MeshGenerator: Tetrahedral mesh generation from atlas volume.

This module handles:
- Adaptive size field computation for mesh density control
- iso2mesh-based tetrahedral mesh generation (using cgalmesh method)
- Surface triangle extraction
- Mesh quality checking
- Save/load mesh data as .npz
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

import numpy as np
import iso2mesh

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Container for mesh data."""

    nodes: np.ndarray
    elements: np.ndarray
    tissue_labels: np.ndarray
    surface_faces: np.ndarray
    surface_node_indices: np.ndarray


@dataclass
class MeshQualityReport:
    """Mesh quality metrics."""

    num_nodes: int
    num_elements: int
    num_surface_faces: int
    min_element_volume: float
    max_element_volume: float
    mean_element_volume: float
    num_degenerate_elements: int
    aspect_ratio_min: float
    aspect_ratio_max: float
    element_volume_distribution: Dict[int, int]


class MeshGenerator:
    """Generate tetrahedral FEM meshes from atlas volumes with adaptive refinement."""

    def __init__(self, config: Dict):
        """Initialize mesh generator.

        Parameters
        ----------
        config : Dict
            Configuration dictionary with mesh parameters:
            - target_nodes: int
            - surface_maxvol: float
            - deep_maxvol: float
            - roi_maxvol: float
            - output_path: str
        """
        self.config = config
        self.target_nodes = config.get("target_nodes", 10000)
        self.surface_maxvol = config.get("surface_maxvol", 0.5)
        self.deep_maxvol = config.get("deep_maxvol", 5.0)
        self.roi_maxvol = config.get("roi_maxvol", 1.0)
        self.output_path = Path(config.get("output_path", "assets/mesh/"))

    def generate(
        self,
        atlas_volume: np.ndarray,
        voxel_size: float = 0.1,
        tissue_labels: Optional[np.ndarray] = None,
        downsample_factor: int = 8,
        crop_to_trunk: bool = True,
    ) -> MeshData:
        """Generate tetrahedral mesh from atlas volume.

        Parameters
        ----------
        atlas_volume : np.ndarray
            3D tissue label array [X×Y×Z].
        voxel_size : float, default 0.1
            Voxel size in mm.
        tissue_labels : Optional[np.ndarray], default None
            Not used, kept for API compatibility.
        downsample_factor : int, default 8
            Downsampling factor for the volume.
        crop_to_trunk : bool, default True
            If True, crop mesh to trunk bounding box (for atlas-origin meshes).
            If False, skip crop (for meshes already in trunk-local frame).

        Returns
        -------
        MeshData
            Named tuple containing nodes, elements, tissue_labels, surface_faces.
        """
        logger.info(
            f"Starting mesh generation with downsample_factor={downsample_factor}"
        )

        orig_shape = atlas_volume.shape
        logger.info(f"Original atlas shape: {orig_shape}")

        downsample_shape = (
            int(np.ceil(orig_shape[0] / downsample_factor)),
            int(np.ceil(orig_shape[1] / downsample_factor)),
            int(np.ceil(orig_shape[2] / downsample_factor)),
        )
        logger.info(f"Downsampled shape: {downsample_shape}")

        downsampled_vol = self._downsample_volume(atlas_volume, downsample_shape)

        effective_voxel_size = voxel_size * downsample_factor
        logger.info(
            f"Effective voxel size after downsampling: {effective_voxel_size} mm"
        )

        logger.info("Calling iso2mesh vol2mesh with cgalmesh method...")
        node, elem, face = iso2mesh.vol2mesh(
            downsampled_vol,
            ix=np.arange(0, downsample_shape[0]),
            iy=np.arange(0, downsample_shape[1]),
            iz=np.arange(0, downsample_shape[2]),
            opt={},
            maxvol=self._estimate_maxvol(downsample_shape, effective_voxel_size),
            dofix=True,
            method="cgalmesh",
        )

        logger.info(f"Mesh generated: nodes={node.shape[0]}, elements={elem.shape[0]}")

        nodes_phys = node[:, :3] * effective_voxel_size
        tissue_labels_elem = elem[:, 4].astype(np.int32)

        tet_elements_raw = elem[:, :4].astype(np.int32) - 1
        tet_elements_raw = self._ensure_tet_orientation(nodes_phys, tet_elements_raw)

        # === 新的:先合并节点,再提取边界 ===
        merged_nodes, merged_elements, surface_faces, surface_node_indices = \
            self._merge_dup_nodes_and_extract_boundary(
                nodes_phys, tet_elements_raw, tissue_labels_elem,
                tol=max(effective_voxel_size * 1e-3, 1e-6),
            )

        logger.info(
            f"节点合并: {len(nodes_phys)} → {len(merged_nodes)} "
            f"(复制率 {len(nodes_phys)/len(merged_nodes):.2f}×)"
        )
        logger.info(f"边界面: {len(surface_faces)} 三角形, "
                    f"{len(surface_node_indices)} unique 节点, "
                    f"F/V = {len(surface_faces)/len(surface_node_indices):.2f}")

        quality = self.check_quality(merged_nodes, merged_elements)
        logger.info(
            f"Mesh quality: nodes={quality.num_nodes}, "
            f"elements={quality.num_elements}, "
            f"degenerate={quality.num_degenerate_elements}"
        )

        mesh_data = MeshData(
            nodes=merged_nodes,
            elements=merged_elements,
            tissue_labels=tissue_labels_elem,
            surface_faces=surface_faces,
            surface_node_indices=surface_node_indices,
        )

        # ── Crop to trunk region ───────────────────────────────────────────────
        if crop_to_trunk:
            mesh_data = self._crop_to_trunk(mesh_data)

        return mesh_data

    # -------------------------------------------------------------------------
    # Trunk cropping
    # -------------------------------------------------------------------------
    # Trunk bounding box in atlas physical coordinates (mm).
    # MUST match mcx_volume.py crop: y_start=340, y_end=740 (voxel at 0.1mm)
    # = Y ∈ [34, 74] mm atlas = the same torso region as the MCX volume crop.
    TRUNK_BBOX_ATLAS = {
        "x": (-1.0, 39.0),    # trunk X extent: 0..38mm
        "y": (34.0, 74.0),    # trunk Y extent: 34..74mm atlas (matches MCX crop y_start=340,y_end=740)
        "z": (-1.0, 21.8),    # trunk Z extent: 0..20.8mm
    }

    def _crop_to_trunk(self, mesh_data: "MeshData") -> "MeshData":
        """Crop mesh to trunk bounding box (atlas physical mm).

        Only keeps nodes within the trunk bbox and remaps element connectivity.
        Returns original mesh_data if too few nodes remain after crop.
        """
        from dataclasses import replace

        nodes = mesh_data.nodes
        elements = mesh_data.elements
        bbox = self.TRUNK_BBOX_ATLAS

        in_bbox = (
            (nodes[:, 0] >= bbox["x"][0]) & (nodes[:, 0] <= bbox["x"][1]) &
            (nodes[:, 1] >= bbox["y"][0]) & (nodes[:, 1] <= bbox["y"][1]) &
            (nodes[:, 2] >= bbox["z"][0]) & (nodes[:, 2] <= bbox["z"][1])
        )
        keep_idx = np.where(in_bbox)[0]
        n_orig = len(nodes)

        if len(keep_idx) < 100:
            logger.warning(
                f"  Trunk crop: only {len(keep_idx)}/{n_orig} nodes, keeping full mesh"
            )
            return mesh_data

        # Remap old node indices to new consecutive indices
        old_to_new = np.full(n_orig, -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

        # Filter elements: all 4 nodes must be in bbox
        elem_mask = in_bbox[elements].all(axis=1)
        cropped_elements = old_to_new[elements[elem_mask]]
        cropped_nodes = nodes[keep_idx]
        cropped_tissue_labels = mesh_data.tissue_labels[elem_mask]

        # Surface faces: all 3 nodes must be in bbox
        sf_mask = in_bbox[mesh_data.surface_faces].all(axis=1)
        cropped_surface_faces = old_to_new[mesh_data.surface_faces[sf_mask]]
        surface_node_indices = np.unique(cropped_surface_faces)

        logger.info(
            f"  Trunk crop: {n_orig} → {len(cropped_nodes)} nodes, "
            f"{len(elements)} → {len(cropped_elements)} elements"
        )

        return replace(
            mesh_data,
            nodes=cropped_nodes,
            elements=cropped_elements,
            tissue_labels=cropped_tissue_labels,
            surface_faces=cropped_surface_faces,
            surface_node_indices=surface_node_indices,
        )

    def _downsample_volume(
        self, volume: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Downsample volume using nearest neighbor interpolation.

        Parameters
        ----------
        volume : np.ndarray
            Original volume.
        target_shape : Tuple[int, int, int]
            Target shape after downsampling.

        Returns
        -------
        np.ndarray
            Downsampled volume.
        """
        from scipy.ndimage import zoom

        factors = (
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            target_shape[2] / volume.shape[2],
        )

        downsampled = zoom(volume, factors, order=0)

        downsampled = np.round(downsampled).astype(np.uint8)

        return downsampled

    def _estimate_maxvol(
        self, volume_shape: Tuple[int, int, int], voxel_size: float
    ) -> float:
        """Estimate maxvol parameter based on target node count.

        This is a heuristic estimate. The actual optimal maxvol may differ
        and can be found through binary search.

        Parameters
        ----------
        volume_shape : Tuple[int, int, int]
            Volume shape after downsampling.
        voxel_size : float
            Effective voxel size in mm.

        Returns
        -------
        float
            Estimated maxvol in mm^3.
        """
        num_voxels = volume_shape[0] * volume_shape[1] * volume_shape[2]
        voxel_volume = voxel_size**3

        total_volume = num_voxels * voxel_volume

        target_tets = self.target_nodes * 5

        maxvol_estimate = total_volume / target_tets

        maxvol_estimate = max(maxvol_estimate, self.surface_maxvol)
        maxvol_estimate = min(maxvol_estimate, self.deep_maxvol)

        logger.info(
            f"Estimated maxvol={maxvol_estimate:.4f} mm³ "
            f"(target ~{self.target_nodes} nodes, "
            f"~{target_tets} tets)"
        )

        return maxvol_estimate

    @staticmethod
    def _ensure_tet_orientation(
        nodes: np.ndarray, tet_elements: np.ndarray
    ) -> np.ndarray:
        """Fix iso2mesh tet vertex ordering to right-hand rule.

        iso2mesh/cgalmesh does not guarantee tet vertex order produces
        positive determinant. This method detects and swaps vertex 2/3
        for tets with negative volume.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N, 3].
        tet_elements : np.ndarray
            Tetrahedron node indices [T, 4], 0-based.

        Returns
        -------
        np.ndarray
            Tet elements with positive determinants [T, 4].
        """
        v0 = nodes[tet_elements[:, 0]]
        v1 = nodes[tet_elements[:, 1]]
        v2 = nodes[tet_elements[:, 2]]
        v3 = nodes[tet_elements[:, 3]]

        cross23 = np.cross(v1 - v0, v2 - v0)
        det = np.sum(cross23 * (v3 - v0), axis=1)

        bad = det < 0
        if bad.sum() == 0:
            return tet_elements

        result = tet_elements.copy()
        result[bad, 2] = tet_elements[bad, 3]
        result[bad, 3] = tet_elements[bad, 2]
        logger.info(f"  Fixed orientation on {bad.sum()}/{len(tet_elements)} tets")
        return result

    @staticmethod
    def _extract_exterior_faces_fast(tet_elements: np.ndarray) -> np.ndarray:
        """Extract exterior hull faces via count==1 filter, O(T log T) vectorized.

        A face is an exterior hull face iff it appears in exactly one tet.
        Each tet supplies 4 oriented faces with the "absent vertex" winding:
          absent v0 → [v1, v2, v3]
          absent v1 → [v0, v3, v2]
          absent v2 → [v0, v1, v3]
          absent v3 → [v0, v2, v1]

        This automatically produces outward-pointing normals.

        Parameters
        ----------
        tet_elements : np.ndarray
            Tetrahedron node indices [T, 4], 0-based, right-hand oriented.

        Returns
        -------
        np.ndarray
            Exterior hull faces [F_ext, 3], int32, 0-based.
        """
        T = tet_elements.shape[0]

        f0 = tet_elements[:, [1, 2, 3]]
        f1 = tet_elements[:, [0, 3, 2]]
        f2 = tet_elements[:, [0, 1, 3]]
        f3 = tet_elements[:, [0, 2, 1]]

        all_faces = np.concatenate([f0, f1, f2, f3], axis=0).astype(np.int32)
        all_tets = np.repeat(np.arange(T, dtype=np.int32), 4)
        all_fids = np.tile(np.arange(4, dtype=np.int32), T)

        # Canonicalize: sort node indices within each face
        sorted_faces = np.sort(all_faces, axis=1)

        # Group identical sorted faces; lexsort by sorted face only (no tiebreaker)
        order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
        sorted_faces = sorted_faces[order]
        all_tets = all_tets[order]
        all_fids = all_fids[order]

        # Find boundaries between different face groups
        diff = np.diff(sorted_faces, axis=0, prepend=[[-1, -1, -1]])
        is_diff = np.any(diff != 0, axis=1)  # True at boundary starts

        run_start = np.zeros(len(sorted_faces), dtype=bool)
        run_start[0] = True
        run_start[1:] = is_diff[1:]

        run_id = np.cumsum(run_start)
        n_runs = int(run_id[-1])
        run_counts = np.bincount(run_id, minlength=n_runs + 1)

        exterior_mask = run_counts[run_id] == 1

        ext_tets = all_tets[exterior_mask]
        ext_fids = all_fids[exterior_mask]

        result = np.zeros((len(ext_tets), 3), dtype=np.int32)
        for i in range(len(ext_tets)):
            tet = tet_elements[ext_tets[i]]
            fid = ext_fids[i]
            if fid == 0:
                result[i] = [tet[1], tet[2], tet[3]]
            elif fid == 1:
                result[i] = [tet[0], tet[3], tet[2]]
            elif fid == 2:
                result[i] = [tet[0], tet[1], tet[3]]
            else:
                result[i] = [tet[0], tet[2], tet[1]]

        return result

    @staticmethod
    def _merge_dup_nodes_and_extract_boundary(
        nodes: np.ndarray,
        tet_elements: np.ndarray,
        tissue_labels: np.ndarray,
        tol: float = 1e-6,
    ) -> tuple:
        """Merge CGAL-duplicated nodes, then extract exterior hull via count==1.

        CGAL's mesh generator replicates nodes at multi-label boundaries.
        This function: (1) merges duplicates by rounding coords, then (2) runs
        the preAmiraMesh.m count==1 exterior hull filter on the merged topology.

        Returns (merged_nodes, merged_elements, boundary_faces, surface_node_idx).
        """
        # ── 1. Merge duplicate nodes ──
        keys = np.round(nodes / tol).astype(np.int64)
        _, uniq_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        merged_nodes = nodes[uniq_idx]
        merged_elements = inverse[tet_elements].astype(np.int32)

        # ── 2. Build all 4T oriented faces ──
        T = merged_elements.shape[0]
        faces = np.concatenate([
            merged_elements[:, [0, 1, 2]],
            merged_elements[:, [0, 1, 3]],
            merged_elements[:, [0, 2, 3]],
            merged_elements[:, [1, 2, 3]],
        ], axis=0)
        tet_id = np.tile(np.arange(T, dtype=np.int32), 4)
        local_id = np.repeat(np.arange(4, dtype=np.int32), T)

        # ── 3. Count appearances via sorted-group ──
        sorted_faces = np.sort(faces, axis=1)
        order = np.lexsort(sorted_faces.T[::-1])
        sf = sorted_faces[order]
        tid = tet_id[order]
        lid = local_id[order]

        diff = np.any(np.diff(sf, axis=0) != 0, axis=1)
        run_start = np.concatenate([[True], diff])
        run_id = np.cumsum(run_start) - 1
        counts = np.bincount(run_id)
        is_bdry = counts[run_id] == 1

        bnd_tet = tid[is_bdry]
        bnd_lid = lid[is_bdry]

        # ── 4. Reconstruct outward-winding faces ──
        LOCAL_FACES = np.array([
            [1, 2, 3],    # absent v0
            [0, 3, 2],    # absent v1
            [0, 1, 3],    # absent v2
            [0, 2, 1],    # absent v3
        ], dtype=np.int32)
        boundary_faces = merged_elements[
            bnd_tet[:, None], LOCAL_FACES[bnd_lid]
        ].astype(np.int32)

        surface_node_idx = np.unique(boundary_faces).astype(np.int32)

        return merged_nodes, merged_elements, boundary_faces, surface_node_idx
        """Extract surface triangles from tetrahedral mesh.

        Boundary faces are triangles that appear exactly once across all
        tetrahedra faces.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3].
        elements : np.ndarray
            Tetrahedron node indices [M×4].

        Returns
        -------
        np.ndarray
            Surface triangle indices [F×3].
        """
        num_elements = elements.shape[0]
        face_counts: Dict[Tuple[int, int, int], int] = {}

        for elem_idx in range(num_elements):
            n0, n1, n2, n3 = elements[elem_idx]

            faces = [
                tuple(sorted([n0, n1, n2])),
                tuple(sorted([n0, n1, n3])),
                tuple(sorted([n0, n2, n3])),
                tuple(sorted([n1, n2, n3])),
            ]

            for face in faces:
                face_counts[face] = face_counts.get(face, 0) + 1

        surface_faces = np.array(
            [list(face) for face, count in face_counts.items() if count == 1],
            dtype=np.int32,
        )

        return surface_faces

    def check_quality(
        self, nodes: np.ndarray, elements: np.ndarray
    ) -> MeshQualityReport:
        """Check tetrahedral mesh quality.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3].
        elements : np.ndarray
            Tetrahedron node indices [M×4].

        Returns
        -------
        MeshQualityReport
            Mesh quality metrics.
        """
        num_nodes = nodes.shape[0]
        num_elements = elements.shape[0]

        volumes = self._compute_tetrahedron_volumes(nodes, elements)

        degenerate_mask = volumes < 1e-10
        num_degenerate = np.sum(degenerate_mask)

        aspect_ratios = self._compute_aspect_ratios(nodes, elements)

        label_distribution: Dict[int, int] = {}
        for i in range(num_elements):
            elem_label = i
            label_distribution[elem_label] = 1

        return MeshQualityReport(
            num_nodes=num_nodes,
            num_elements=num_elements,
            num_surface_faces=0,
            min_element_volume=float(np.min(volumes)),
            max_element_volume=float(np.max(volumes)),
            mean_element_volume=float(np.mean(volumes)),
            num_degenerate_elements=num_degenerate,
            aspect_ratio_min=float(np.min(aspect_ratios)),
            aspect_ratio_max=float(np.max(aspect_ratios)),
            element_volume_distribution=label_distribution,
        )

    def _compute_tetrahedron_volumes(
        self, nodes: np.ndarray, elements: np.ndarray
    ) -> np.ndarray:
        """Compute volumes of all tetrahedra.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3].
        elements : np.ndarray
            Tetrahedron node indices [M×4].

        Returns
        -------
        np.ndarray
            Volume of each tetrahedron [M].
        """
        num_elements = elements.shape[0]
        volumes = np.zeros(num_elements)

        for i in range(num_elements):
            n0, n1, n2, n3 = elements[i]
            v0 = nodes[n0]
            v1 = nodes[n1]
            v2 = nodes[n2]
            v3 = nodes[n3]

            v1v0 = v1 - v0
            v2v0 = v2 - v0
            v3v0 = v3 - v0

            volume = np.abs(np.dot(v1v0, np.cross(v2v0, v3v0))) / 6.0
            volumes[i] = volume

        return volumes

    def _compute_aspect_ratios(
        self, nodes: np.ndarray, elements: np.ndarray
    ) -> np.ndarray:
        """Compute aspect ratios of all tetrahedra.

        Aspect ratio = longest_edge / shortest_edge

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3].
        elements : np.ndarray
            Tetrahedron node indices [M×4].

        Returns
        -------
        np.ndarray
            Aspect ratio of each tetrahedron [M].
        """
        num_elements = elements.shape[0]
        aspect_ratios = np.zeros(num_elements)

        for i in range(num_elements):
            n0, n1, n2, n3 = elements[i]
            pts = nodes[[n0, n1, n2, n3]]

            edges = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[0]),
                np.linalg.norm(pts[3] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
            ]

            aspect_ratios[i] = max(edges) / (min(edges) + 1e-10)

        return aspect_ratios

    def save(self, mesh_data: MeshData, filename: str) -> Path:
        """Save mesh data to .npz file.

        Parameters
        ----------
        mesh_data : MeshData
            Mesh data to save.
        filename : str
            Output filename (without extension).

        Returns
        -------
        Path
            Path to saved file.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        filepath = self.output_path / f"{filename}.npz"

        np.savez(
            filepath,
            nodes=mesh_data.nodes,
            elements=mesh_data.elements,
            tissue_labels=mesh_data.tissue_labels,
            surface_faces=mesh_data.surface_faces,
            surface_node_indices=mesh_data.surface_node_indices,
        )

        logger.info(f"Mesh saved to: {filepath}")
        return filepath

    def save_exterior(
        self,
        mesh_data: MeshData,
        exterior_surface_faces: np.ndarray,
        filename: str,
    ) -> Path:
        """Save mesh data including exterior hull faces to .npz file.

        Parameters
        ----------
        mesh_data : MeshData
            Mesh data to save.
        exterior_surface_faces : np.ndarray [E×3]
            Exterior hull faces (adjacent to exactly 1 tet).
        filename : str
            Output filename (without extension).

        Returns
        -------
        Path
            Path to saved file.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        filepath = self.output_path / f"{filename}.npz"

        np.savez(
            filepath,
            nodes=mesh_data.nodes,
            elements=mesh_data.elements,
            tissue_labels=mesh_data.tissue_labels,
            surface_faces=mesh_data.surface_faces,
            surface_node_indices=mesh_data.surface_node_indices,
            exterior_surface_faces=exterior_surface_faces,
        )

        logger.info(f"Mesh saved to: {filepath}")
        return filepath

    def load(self, filename: str) -> MeshData:
        """Load mesh data from .npz file.

        Parameters
        ----------
        filename : str
            Path to .npz file.

        Returns
        -------
        MeshData
            Loaded mesh data.
        """
        data = np.load(filename)
        return MeshData(
            nodes=data["nodes"],
            elements=data["elements"],
            tissue_labels=data["tissue_labels"],
            surface_faces=data["surface_faces"],
            surface_node_indices=data["surface_node_indices"],
        )
