#!/usr/bin/env python3
"""
Regenerate all 200 dataset samples with deterministic foci counts.

This script generates all 200 samples from scratch with:
- Deterministic foci allocation: 60x1-foci, 70x2-foci, 70x3-foci
- Proper JSON tumor_params.json (using json.dump, not yaml.dump)
- Seed 42 for reproducibility

Usage:
    python scripts/03_regenerate_dataset.py
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import scipy.sparse as sp
import yaml
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.physics.optical_params import OpticalParameterManager
from fmt_simgen.tumor.tumor_generator import TumorGenerator
from fmt_simgen.sampling.dual_sampler import DualSampler, VoxelGridConfig


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(config["dataset"]["output_path"])
    samples_dir = output_path / "samples"

    # Clear existing samples directory
    if samples_dir.exists():
        print(f"Removing existing samples directory: {samples_dir}")
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Find shared directory (try multiple possible locations)
    possible_shared_dirs = [
        Path("output/shared"),
        output_path / "shared",
        Path("assets/shared"),
    ]
    shared_dir = None
    for d in possible_shared_dirs:
        if d.exists() and (d / "mesh.npz").exists():
            shared_dir = d
            break
    if shared_dir is None:
        raise FileNotFoundError(
            f"Cannot find shared mesh directory. Tried: {possible_shared_dirs}. "
            f"Please ensure mesh and system matrix exist."
        )
    print(f"Using shared directory: {shared_dir}")

    # Load mesh
    mesh_data = np.load(shared_dir / "mesh.npz")
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    surface_faces = mesh_data["surface_faces"]
    tissue_labels = mesh_data["tissue_labels"]

    # Setup physics
    opt_mgr = OpticalParameterManager(config["physics"]["tissues"], n=config["physics"]["n"])
    solver = FEMSolver(nodes, elements, surface_faces, tissue_labels, opt_mgr)
    A = sp.load_npz(shared_dir / "system_matrix.A.npz")
    solver._forward_matrix = A

    # Voxel grid config (matches existing samples: 150x151x150, spacing=0.2mm)
    shape = (150, 151, 150)
    spacing = 0.2
    node_min = nodes.min(axis=0)
    voxel_cfg = VoxelGridConfig(shape=shape, spacing=spacing, offset=node_min)

    sampler = DualSampler(nodes=nodes, voxel_grid_config=voxel_cfg)

    # Mesh bbox for tumor generator
    mesh_min = nodes.min(axis=0)
    mesh_max = nodes.max(axis=0)
    mesh_bbox = {"min": mesh_min.tolist(), "max": mesh_max.tolist()}

    # Tumor generator
    tumor_gen = TumorGenerator(config["tumor"], mesh_bbox=mesh_bbox)

    # Deterministic foci allocation
    n_total = 200
    n_1foci = 60  # 30%
    n_2foci = 70  # 35%
    n_3foci = 70  # 35%

    np.random.seed(42)
    foci_counts = [1] * n_1foci + [2] * n_2foci + [3] * n_3foci
    np.random.shuffle(foci_counts)

    print(f"Generating {n_total} samples with foci distribution:")
    print(f"  1-foci: {n_1foci} (30%)")
    print(f"  2-foci: {n_2foci} (35%)")
    print(f"  3-foci: {n_3foci} (35%)")
    print(f"Output: {samples_dir}")

    for i, n_foci in enumerate(foci_counts):
        sample_dir = samples_dir / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)

        # Generate tumor with forced foci count
        tumor = tumor_gen.generate_sample(num_foci=n_foci)

        # Dual sampling
        gt_voxels = sampler.sample_to_voxels(tumor)
        gt_nodes = sampler.sample_to_nodes(tumor)

        # Forward measurement
        b = solver.forward(gt_nodes)

        # Save tumor_params.json with explicit num_foci field
        params = tumor.to_dict()
        params["num_foci"] = n_foci
        with open(sample_dir / "tumor_params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)

        # Save other arrays
        np.save(sample_dir / "measurement_b.npy", b.astype(np.float32))
        np.save(sample_dir / "gt_nodes.npy", gt_nodes.astype(np.float32))
        np.save(sample_dir / "gt_voxels.npy", gt_voxels.astype(np.float32))

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{n_total} samples...")

    print(f"\nDone! All {n_total} samples saved to {samples_dir}")

    # Verify foci distribution
    foci_counts_verify = {}
    for i in range(n_total):
        params = json.load(open(samples_dir / f"sample_{i:04d}" / "tumor_params.json"))
        n = params.get("num_foci", len(params.get("foci", [])))
        foci_counts_verify[n] = foci_counts_verify.get(n, 0) + 1
    print(f"\nVerification - foci distribution: {foci_counts_verify}")


if __name__ == "__main__":
    main()
