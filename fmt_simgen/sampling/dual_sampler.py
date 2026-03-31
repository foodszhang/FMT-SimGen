"""
DualSampler: Sample ground truth at both FEM nodes and voxel grid.

This module implements dual-carrier GT sampling:
- gt_nodes: sampled at FEM node coordinates
- gt_voxels: sampled at voxel grid centers

Both GTs come from the same analytic tumor function, ensuring alignment.
"""

import numpy as np
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class VoxelGridConfig:
    """Configuration for voxel grid."""

    shape: Tuple[int, int, int]
    spacing: float
    offset: np.ndarray


class DualSampler:
    """Sample ground truth at FEM nodes and voxel grid."""

    def __init__(self, nodes: np.ndarray, voxel_grid_config: VoxelGridConfig):
        """Initialize dual sampler.

        Parameters
        ----------
        nodes : np.ndarray
            FEM node coordinates [N_n × 3] in mm.
        voxel_grid_config : VoxelGridConfig
            Configuration for voxel grid.
        """
        self.nodes = nodes
        self.voxel_grid_config = voxel_grid_config

    def sample_to_nodes(self, tumor_sample) -> np.ndarray:
        """Sample tumor at FEM node coordinates.

        Parameters
        ----------
        tumor_sample : TumorSample
            Tumor sample with analytic foci.

        Returns
        -------
        np.ndarray
            Ground truth at nodes [N_n].
        """
        return tumor_sample.evaluate(self.nodes)

    def sample_to_voxels(self, tumor_sample) -> np.ndarray:
        """Sample tumor at voxel grid centers.

        Parameters
        ----------
        tumor_sample : TumorSample
            Tumor sample with analytic foci.

        Returns
        -------
        np.ndarray
            Ground truth at voxels [Nx × Ny × Nz].
        """
        nx, ny, nz = self.voxel_grid_config.shape
        spacing = self.voxel_grid_config.spacing
        offset = self.voxel_grid_config.offset

        grid_x = np.arange(nx) * spacing + offset[0] + spacing / 2
        grid_y = np.arange(ny) * spacing + offset[1] + spacing / 2
        grid_z = np.arange(nz) * spacing + offset[2] + spacing / 2

        gz, gy, gx = np.meshgrid(grid_z, grid_y, grid_x, indexing="ij")

        coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        values = tumor_sample.evaluate(coords)

        return values.reshape(nx, ny, nz)

    def sample_dual(self, tumor_sample) -> Dict[str, np.ndarray]:
        """Sample tumor at both FEM nodes and voxel grid.

        Parameters
        ----------
        tumor_sample : TumorSample
            Tumor sample with analytic foci.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'gt_nodes' and 'gt_voxels'.
        """
        return {
            "gt_nodes": self.sample_to_nodes(tumor_sample),
            "gt_voxels": self.sample_to_voxels(tumor_sample),
        }
