#!/usr/bin/env python3
"""
Step 3: Verify dataset correctness.

This script performs visual checks on generated data:
- Verify mesh quality
- Check system matrix properties
- Visualize sample tumors
- Compare dual GT alignment

Usage:
    python scripts/03_verify_dataset.py [--sample_idx 0]
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from fmt_simgen.dataset.builder import DatasetBuilder
from fmt_simgen.utils.io import load_json


def verify_mesh(builder: DatasetBuilder):
    """Verify mesh data quality."""
    print("\n--- Mesh Verification ---")

    mesh_data = builder._mesh_data

    print(f"  Nodes: {mesh_data.nodes.shape}")
    print(f"  Elements: {mesh_data.elements.shape}")
    print(f"  Surface faces: {mesh_data.surface_faces.shape}")
    print(f"  Tissue labels: {np.unique(mesh_data.tissue_labels)}")

    node_bounds = (
        f"X: [{mesh_data.nodes[:, 0].min():.2f}, {mesh_data.nodes[:, 0].max():.2f}] mm, "
        f"Y: [{mesh_data.nodes[:, 1].min():.2f}, {mesh_data.nodes[:, 1].max():.2f}] mm, "
        f"Z: [{mesh_data.nodes[:, 2].min():.2f}, {mesh_data.nodes[:, 2].max():.2f}] mm"
    )
    print(f"  Node bounds: {node_bounds}")


def verify_system_matrix(builder: DatasetBuilder):
    """Verify system matrix properties."""
    print("\n--- System Matrix Verification ---")

    matrices = builder._fem_solver._matrices

    M = matrices.M.tocsr()

    print(f"  Matrix shape: {M.shape}")
    print(f"  Non-zeros: {M.nnz}")
    print(f"  Sparsity: {M.nnz / (M.shape[0] * M.shape[1]) * 100:.2f}%")

    diag = np.array(M.diagonal())
    print(f"  Diagonal range: [{diag.min():.4f}, {diag.max():.4f}]")

    eigenvalues = np.linalg.eigvalsh(M.toarray())
    print(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    print(f"  Eigenvalue condition: {eigenvalues.max() / eigenvalues.min():.2f}")


def verify_sample(builder: DatasetBuilder, sample_idx: int, data_path: Path):
    """Verify a single sample."""
    print(f"\n--- Sample {sample_idx} Verification ---")

    sample_dir = data_path / f"sample_{sample_idx:04d}"

    if not sample_dir.exists():
        print(f"  Sample directory not found: {sample_dir}")
        return

    meas_b = np.load(sample_dir / "measurement_b.npy")
    gt_nodes = np.load(sample_dir / "gt_nodes.npy")
    gt_voxels = np.load(sample_dir / "gt_voxels.npy")
    tumor_params = load_json(sample_dir / "tumor_params.json")

    print(
        f"  measurement_b shape: {meas_b.shape}, range: [{meas_b.min():.4f}, {meas_b.max():.4f}]"
    )
    print(
        f"  gt_nodes shape: {gt_nodes.shape}, range: [{gt_nodes.min():.4f}, {gt_nodes.max():.4f}]"
    )
    print(
        f"  gt_voxels shape: {gt_voxels.shape}, range: [{gt_voxels.min():.4f}, {gt_voxels.max():.4f}]"
    )
    print(f"  Num foci: {tumor_params['num_foci']}")

    nonzero_nodes = np.sum(gt_nodes > 0.01)
    nonzero_voxels = np.sum(gt_voxels > 0.01)
    print(f"  Non-zero nodes: {nonzero_nodes}, non-zero voxels: {nonzero_voxels}")


def main():
    parser = argparse.ArgumentParser(description="Verify FMT dataset")
    parser.add_argument(
        "--sample_idx",
        "-i",
        type=int,
        default=0,
        help="Sample index to verify (default: 0)",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    builder = DatasetBuilder(config)

    print("=" * 60)
    print("Step 3: Verifying dataset")
    print("=" * 60)

    builder.build_shared_assets()

    verify_mesh(builder)
    verify_system_matrix(builder)

    data_path = Path(config.get("dataset", {}).get("output_path", "data/"))
    if data_path.exists():
        verify_sample(builder, args.sample_idx, data_path)
    else:
        print(f"\nNo data directory found at {data_path}")
        print("Run 02_generate_dataset.py first to generate samples.")


if __name__ == "__main__":
    main()
