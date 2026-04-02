#!/usr/bin/env python3
"""
Generate additional dataset samples (up to 200 total).
Uses the existing output/shared mesh and system matrix.

Usage:
    python scripts/02b_generate_more_samples.py --num_samples 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import scipy.sparse as sp
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.physics.optical_params import OpticalParameterManager
from fmt_simgen.tumor.tumor_generator import TumorGenerator
from fmt_simgen.sampling.dual_sampler import DualSampler, VoxelGridConfig


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(config["dataset"]["output_path"])
    target_total = 200

    # Load mesh
    mesh_data = np.load("output/shared/mesh.npz")
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    surface_faces = mesh_data["surface_faces"]
    tissue_labels = mesh_data["tissue_labels"]

    # Setup physics
    opt_mgr = OpticalParameterManager(config["physics"]["tissues"], n=config["physics"]["n"])
    solver = FEMSolver(nodes, elements, surface_faces, tissue_labels, opt_mgr)
    A = sp.load_npz("output/shared/system_matrix.A.npz")
    solver._forward_matrix = A

    # Tumor generator
    tumor_gen = TumorGenerator(config["tumor"])

    # Voxel grid config (matches existing samples: 150x151x150, spacing=0.2mm)
    shape = (150, 151, 150)
    spacing = 0.2
    node_min = nodes.min(axis=0)
    voxel_cfg = VoxelGridConfig(shape=shape, spacing=spacing, offset=node_min)

    sampler = DualSampler(nodes=nodes, voxel_grid_config=voxel_cfg)

    # Find existing samples
    existing = sorted([
        int(d.name.split("_")[1])
        for d in output_path.iterdir()
        if d.name.startswith("sample_")
    ])
    start_idx = (existing[-1] + 1) if existing else 0
    n_generate = target_total - start_idx

    if n_generate <= 0:
        print(f"Already have {start_idx} samples ({target_total} target). Nothing to generate.")
        return

    print(f"Generating {n_generate} samples (index {start_idx} to {target_total - 1})")
    print(f"Mesh: {nodes.shape[0]} nodes, A: {A.shape}")
    print(f"Voxel grid: {shape}, spacing: {spacing}mm")

    for i in range(start_idx, target_total):
        sample_dir = output_path / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)

        # Generate tumor
        tumor = tumor_gen.generate_sample()

        # Dual sampling
        gt_voxels = sampler.sample_to_voxels(tumor)
        gt_nodes = sampler.sample_to_nodes(tumor)

        # Forward measurement
        b = solver.forward(gt_nodes)

        # Save
        np.save(sample_dir / "measurement_b.npy", b.astype(np.float32))
        np.save(sample_dir / "gt_nodes.npy", gt_nodes.astype(np.float32))
        np.save(sample_dir / "gt_voxels.npy", gt_voxels.astype(np.float32))
        with open(sample_dir / "tumor_params.json", "w") as f:
            yaml.dump(tumor.to_dict(), f)

        if (i - start_idx + 1) % 20 == 0:
            print(f"  Generated {i - start_idx + 1}/{n_generate} samples...")

    print(f"Done! Total samples: {target_total}")


if __name__ == "__main__":
    main()
