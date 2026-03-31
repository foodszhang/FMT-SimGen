#!/usr/bin/env python3
"""
Step 0c: FEM System Matrix Assembly

This script:
1. Loads mesh from output/shared/mesh.npz
2. Loads optical parameters from config
3. Assembles FEM matrices (K, C, F, B, M)
4. Computes forward matrix A = M^{-1} * F
5. Saves system matrix to output/shared/system_matrix.npz

Usage:
    python scripts/step0c_fem_matrix.py [--compute_A]

Output:
    output/shared/system_matrix.*.npz
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from fmt_simgen.mesh.mesh_generator import MeshGenerator
from fmt_simgen.physics.optical_params import OpticalParameterManager
from fmt_simgen.physics.fem_solver import FEMSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Step 0c: FEM System Matrix Assembly")
    parser.add_argument(
        "--compute_A",
        action="store_true",
        help="Compute forward matrix A (takes a few minutes)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation test after assembly",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 0c: FEM System Matrix Assembly")
    logger.info("=" * 60)

    config = load_config()

    mesh_path = OUTPUT_DIR / "mesh.npz"
    logger.info(f"Loading mesh from: {mesh_path}")
    mesh_data = np.load(mesh_path, allow_pickle=True)

    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    tissue_labels = mesh_data["tissue_labels"]
    surface_faces = mesh_data["surface_faces"]

    logger.info(f"Mesh loaded: nodes={nodes.shape[0]}, elements={elements.shape[0]}")
    logger.info(f"  surface_faces={surface_faces.shape[0]}")
    logger.info(f"  tissue labels: {np.unique(tissue_labels)}")

    physics_config = config.get("physics", {})
    tissues_config = physics_config.get("tissues", {})
    n = physics_config.get("n", 1.37)

    logger.info("Creating optical parameter manager...")
    opt_manager = OpticalParameterManager(tissues_config, n=n)

    logger.info("Creating FEM solver...")
    solver = FEMSolver(
        nodes=nodes,
        elements=elements,
        surface_faces=surface_faces,
        tissue_labels=tissue_labels,
        opt_params_manager=opt_manager,
    )

    logger.info("Assembling system matrix...")
    matrices = solver.assemble_system_matrix()

    if args.validate:
        logger.info("\nRunning validation (point source test)...")
        validation_results = solver.validate()
        logger.info(f"Validation results: {validation_results}")

    if args.compute_A:
        logger.info("\nComputing forward matrix A = M^{-1} * F...")
        logger.info("This may take several minutes...")
        A = solver.compute_forward_matrix()
        logger.info(f"Forward matrix shape: {A.shape}")

        A_sparsity = np.count_nonzero(A) / A.size * 100
        logger.info(f"Forward matrix sparsity: {A_sparsity:.2f}%")
        logger.info(f"Forward matrix max: {A.max():.6f}, min: {A.min():.6f}")

    logger.info("\nSaving system matrix...")
    solver.save_system_matrix(str(OUTPUT_DIR / "system_matrix"))

    logger.info("=" * 60)
    logger.info("FEM MATRIX ASSEMBLY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output files: {OUTPUT_DIR / 'system_matrix.*.npz'}")


if __name__ == "__main__":
    main()
