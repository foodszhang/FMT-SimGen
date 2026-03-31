#!/usr/bin/env python3
"""
Step 0e: Graph Laplacian Computation

Computes standard and multi-scale kernel graph Laplacian matrices
for MS-GDUN training.

Output:
    output/shared/graph_laplacian.npz

Usage:
    python scripts/step0e_graph_laplacian.py
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp

from fmt_simgen.graph_laplacian import (
    build_surface_adjacency,
    remap_surface_indices,
    compute_topological_laplacian,
    compute_kernel_laplacian_kdtree,
    compute_laplacian_statistics,
    compute_kernel_neighbors_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"


def main():
    logger.info("=" * 60)
    logger.info("Step 0e: Graph Laplacian Computation")
    logger.info("=" * 60)

    logger.info("Loading mesh data...")
    mesh_data = np.load(OUTPUT_DIR / "mesh.npz", allow_pickle=True)
    nodes = mesh_data["nodes"]
    surface_faces = mesh_data["surface_faces"]
    surface_node_indices = mesh_data["surface_node_indices"]

    n_total = nodes.shape[0]
    n_surface = len(surface_node_indices)

    logger.info(f"  Total nodes: {n_total}")
    logger.info(f"  Surface nodes: {n_surface}")
    logger.info(f"  Surface faces: {surface_faces.shape[0]}")

    logger.info("Remapping surface faces to local indices...")
    surface_faces_local = remap_surface_indices(surface_faces, surface_node_indices)

    logger.info("Building surface adjacency matrix...")
    adj = build_surface_adjacency(surface_faces_local, n_surface)

    logger.info("Computing standard topological Laplacian...")
    Lap = compute_topological_laplacian(adj)

    logger.info("Computing multi-scale kernel Laplacian (this may take a few minutes)...")
    surface_coords = nodes[surface_node_indices]

    radii = (1.0, 2.0, 3.0, 4.0)
    kernel_laps = compute_kernel_laplacian_kdtree(surface_coords, radii)

    logger.info("=" * 60)
    logger.info("VALIDATION STATISTICS")
    logger.info("=" * 60)

    lap_stats = compute_laplacian_statistics(Lap, "Standard Laplacian")
    logger.info(f"\nStandard Laplacian (Lap):")
    logger.info(f"  Shape: {lap_stats['shape']}")
    logger.info(f"  Non-zeros: {lap_stats['nnz']}")
    logger.info(f"  Sparsity: {lap_stats['sparsity']*100:.2f}%")
    logger.info(f"  Max |row sum|: {lap_stats['row_sum_max']:.2e}")
    logger.info(f"  Symmetry max: {lap_stats['sym_max']:.2e}")

    for idx, r in enumerate(radii):
        n_lap_key = f"n_Lap{idx}"
        L = kernel_laps[n_lap_key]
        stats = compute_laplacian_statistics(L, f"Kernel r={r}mm ({n_lap_key})")
        logger.info(f"\n{n_lap_key} (r={r}mm):")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Non-zeros: {stats['nnz']}")
        logger.info(f"  Sparsity: {stats['sparsity']*100:.2f}%")
        logger.info(f"  Max |row sum|: {stats['row_sum_max']:.2e}")
        logger.info(f"  Symmetry max: {stats['sym_max']:.2e}")

        neighbor_stats = compute_kernel_neighbors_stats(surface_coords, r)
        logger.info(f"  Avg neighbors: {neighbor_stats['mean_neighbors']:.1f}")
        logger.info(f"  Min/Max neighbors: {neighbor_stats['min_neighbors']}/{neighbor_stats['max_neighbors']}")

    logger.info("=" * 60)
    logger.info("Saving Graph Laplacian matrices...")
    logger.info("=" * 60)

    output_path = OUTPUT_DIR / "graph_laplacian.npz"

    matrices_to_save = {
        "Lap": Lap,
        "n_Lap0": kernel_laps["n_Lap0"],
        "n_Lap1": kernel_laps["n_Lap1"],
        "n_Lap2": kernel_laps["n_Lap2"],
        "n_Lap3": kernel_laps["n_Lap3"],
        "surface_node_indices": surface_node_indices,
    }

    sp.save_npz(str(output_path.with_suffix(".Lap.npz")), Lap)
    sp.save_npz(str(output_path.with_suffix(".n_Lap0.npz")), kernel_laps["n_Lap0"])
    sp.save_npz(str(output_path.with_suffix(".n_Lap1.npz")), kernel_laps["n_Lap1"])
    sp.save_npz(str(output_path.with_suffix(".n_Lap2.npz")), kernel_laps["n_Lap2"])
    sp.save_npz(str(output_path.with_suffix(".n_Lap3.npz")), kernel_laps["n_Lap3"])
    np.savez(
        output_path,
        **{k: v for k, v in matrices_to_save.items()}
    )

    logger.info(f"Graph Laplacian saved to: {output_path}")
    logger.info("=" * 60)
    logger.info("GRAPH LAPLACIAN COMPUTATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
