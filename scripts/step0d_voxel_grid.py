#!/usr/bin/env python3
"""
Step 0d: Define Voxel Grid

This script defines the voxel grid for GT sampling, aligned with the atlas.

The voxel grid uses the same resolution as the original atlas (0.1mm) but
is cropped to the ROI defined by the mesh's physical bounds.

Usage:
    python scripts/step0d_voxel_grid.py

Output:
    output/shared/voxel_grid.npz
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"


def define_voxel_grid(
    nodes: np.ndarray,
    spacing: float = 0.1,
    margin_mm: float = 2.0,
) -> dict:
    """Define a voxel grid aligned with the atlas.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates [N×3] in mm.
    spacing : float
        Voxel spacing in mm.
    margin_mm : float
        Margin to add around the mesh bounds.

    Returns
    -------
    dict
        Grid definition with shape, origin, spacing, etc.
    """
    node_min = nodes.min(axis=0) - margin_mm
    node_max = nodes.max(axis=0) + margin_mm

    origin = node_min
    shape = np.ceil((node_max - node_min) / spacing).astype(int)

    grid_centers = np.meshgrid(
        np.arange(shape[0]) * spacing + origin[0] + spacing / 2,
        np.arange(shape[1]) * spacing + origin[1] + spacing / 2,
        np.arange(shape[2]) * spacing + origin[2] + spacing / 2,
        indexing="ij",
    )

    voxel_centers = np.stack(
        [grid_centers[0].ravel(), grid_centers[1].ravel(), grid_centers[2].ravel()],
        axis=1,
    )

    grid_def = {
        "shape": shape,
        "origin": origin,
        "spacing": spacing,
        "voxel_centers": voxel_centers,
        "num_voxels": int(np.prod(shape)),
    }

    logger.info(f"Voxel grid defined:")
    logger.info(f"  shape: {shape}")
    logger.info(f"  origin: {origin}")
    logger.info(f"  spacing: {spacing} mm")
    logger.info(f"  num_voxels: {grid_def['num_voxels']}")
    logger.info(
        f"  physical bounds X: [{origin[0]}, {origin[0] + shape[0] * spacing}] mm"
    )
    logger.info(
        f"  physical bounds Y: [{origin[1]}, {origin[1] + shape[1] * spacing}] mm"
    )
    logger.info(
        f"  physical bounds Z: [{origin[2]}, {origin[2] + shape[2] * spacing}] mm"
    )

    return grid_def


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Step 0d: Voxel Grid Definition")
    logger.info("=" * 60)

    mesh_path = OUTPUT_DIR / "mesh.npz"
    logger.info(f"Loading mesh from: {mesh_path}")
    mesh_data = np.load(mesh_path, allow_pickle=True)

    nodes = mesh_data["nodes"]
    logger.info(f"Mesh nodes: {nodes.shape[0]}")

    spacing = 0.1

    grid_def = define_voxel_grid(nodes, spacing=spacing, margin_mm=2.0)

    output_path = OUTPUT_DIR / "voxel_grid.npz"
    np.savez(
        output_path,
        shape=grid_def["shape"],
        origin=grid_def["origin"],
        spacing=grid_def["spacing"],
        voxel_centers=grid_def["voxel_centers"],
        num_voxels=grid_def["num_voxels"],
    )

    logger.info(f"\nVoxel grid saved to: {output_path}")

    logger.info("=" * 60)
    logger.info("VOXEL GRID DEFINITION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
