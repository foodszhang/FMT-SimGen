"""
DualSampler: Sample ground truth at both FEM nodes and voxel grid.

This module implements dual-carrier GT sampling:
- gt_nodes: sampled at FEM node coordinates
- gt_voxels: sampled at voxel grid centers

Both GTs come from the same analytic tumor function, ensuring alignment.
"""

import logging
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

        # Precompute voxel grid coordinates once (not per-sample)
        nx, ny, nz = voxel_grid_config.shape
        spacing = voxel_grid_config.spacing
        offset = voxel_grid_config.offset
        grid_x = np.arange(nx, dtype=np.float32) * np.float32(spacing) + np.float32(offset[0]) + np.float32(spacing / 2)
        grid_y = np.arange(ny, dtype=np.float32) * np.float32(spacing) + np.float32(offset[1]) + np.float32(spacing / 2)
        grid_z = np.arange(nz, dtype=np.float32) * np.float32(spacing) + np.float32(offset[2]) + np.float32(spacing / 2)
        gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        self._voxel_coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)
        del gx, gy, gz  # free meshgrid intermediates immediately
        logger.info(
            f"  Voxel grid: {nx}×{ny}×{nz} = {len(self._voxel_coords)} points (precomputed float32, "
            f"{self._voxel_coords.nbytes / 1e9:.2f} GB)"
        )

    def sample_to_nodes(self, tumor_sample) -> np.ndarray:
        """Sample tumor at FEM node coordinates.

        Parameters
        ----------
        tumor_sample : TumorSample
            Tumor sample with analytic foci.

        Returns
        -------
        np.ndarray
            Ground truth at nodes [N_n] (float32).
        """
        return tumor_sample.evaluate(self.nodes).astype(np.float32)

    def sample_to_voxels(self, tumor_sample) -> np.ndarray:
        """Sample tumor at voxel grid centers.

        Parameters
        ----------
        tumor_sample : TumorSample
            Tumor sample with analytic foci.

        Returns
        -------
        np.ndarray
            Ground truth at voxels [Nx × Ny × Nz] (float32).
        """
        nx, ny, nz = self.voxel_grid_config.shape
        values = tumor_sample.evaluate(self._voxel_coords)
        return values.reshape(nx, ny, nz).astype(np.float32)

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
