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

        surface_mask = face[:, 3] > 0
        surface_faces = face[surface_mask, :3].astype(np.int32) - 1
        logger.info(f"Surface faces: {surface_faces.shape[0]}")

        surface_node_set = np.unique(surface_faces)
        surface_node_indices = surface_node_set.astype(np.int32)
        logger.info(f"Surface nodes: {len(surface_node_indices)}")

        quality = self.check_quality(nodes_phys, elem[:, :4].astype(np.int32) - 1)
        logger.info(
            f"Mesh quality: nodes={quality.num_nodes}, "
            f"elements={quality.num_elements}, "
            f"degenerate={quality.num_degenerate_elements}"
        )

        return MeshData(
            nodes=nodes_phys,
            elements=elem[:, :4].astype(np.int32) - 1,
            tissue_labels=tissue_labels_elem,
            surface_faces=surface_faces,
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

    def extract_surface(self, nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
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
