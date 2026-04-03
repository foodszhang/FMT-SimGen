#!/usr/bin/env python3
"""
Regenerate all 200 dataset samples with deterministic foci and depth tiers.

This script generates all 200 samples from scratch with:
- Deterministic foci allocation: 60x1-foci, 70x2-foci, 70x3-foci
- Deterministic depth tier allocation: 70xshallow + 80xmedium + 50xdeep
- Proper JSON tumor_params.json (using json.dump, not yaml.dump)
- Organ boundary constraint validation
- Seed 42 for foci, seed 43 for depth tiers (decoupled)

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

from datetime import datetime


# Depth tier configuration (from config)
DEPTH_RANGES = {
    "shallow": (1.5, 3.5),
    "medium":  (3.5, 6.0),
    "deep":    (6.0, 8.0),
}


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(config["dataset"]["output_path"])
    samples_dir = output_path / "samples"
    manifest_path = Path("output/dataset_manifest.json")

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

    # Tumor generator (with mesh nodes for organ constraint)
    tumor_gen = TumorGenerator(
        config["tumor"],
        mesh_bbox=mesh_bbox,
        mesh_nodes=nodes,
        tissue_labels=tissue_labels,
        elements=elements,
    )

    # ============ Deterministic foci allocation ============
    n_total = 200
    n_1foci = 60  # 30%
    n_2foci = 70  # 35%
    n_3foci = 70  # 35%

    np.random.seed(42)
    foci_counts = [1] * n_1foci + [2] * n_2foci + [3] * n_3foci
    np.random.shuffle(foci_counts)

    # ============ Deterministic depth tier allocation ============
    depth_tiers = (
        ["shallow"] * 70 +
        ["medium"] * 80 +
        ["deep"] * 50
    )
    np.random.seed(43)  # Different seed to decouple from foci shuffle
    np.random.shuffle(depth_tiers)

    print(f"Generating {n_total} samples:")
    print(f"  Foci: 1-foci={n_1foci}, 2-foci={n_2foci}, 3-foci={n_3foci}")
    print(f"  Depth: shallow=70, medium=80, deep=50")
    print(f"Output: {samples_dir}")

    # Store sample metadata for manifest
    sample_metadata = {}

    for i, (n_foci, tier) in enumerate(zip(foci_counts, depth_tiers)):
        sample_dir = samples_dir / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)

        # Sample depth within tier range
        lo, hi = DEPTH_RANGES[tier]
        depth_mm = np.random.uniform(lo, hi)

        # Generate tumor with forced foci count and depth
        tumor = tumor_gen.generate_sample(
            num_foci=n_foci,
            depth_mm=depth_mm,
            depth_tier=tier,
        )

        # Dual sampling
        gt_voxels = sampler.sample_to_voxels(tumor)
        gt_nodes = sampler.sample_to_nodes(tumor)

        # Forward measurement
        b = solver.forward(gt_nodes)

        # Save tumor_params.json with full metadata
        params = tumor.to_dict()
        params["num_foci"] = n_foci
        params["depth_tier"] = tier
        params["depth_mm"] = round(depth_mm, 4)
        params["organ_constraint_passed"] = getattr(tumor, '_organ_constraint_passed', True)
        with open(sample_dir / "tumor_params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)

        # Save other arrays
        np.save(sample_dir / "measurement_b.npy", b.astype(np.float32))
        np.save(sample_dir / "gt_nodes.npy", gt_nodes.astype(np.float32))
        np.save(sample_dir / "gt_voxels.npy", gt_voxels.astype(np.float32))

        # Store for manifest
        sample_metadata[f"sample_{i:04d}"] = {
            "num_foci": n_foci,
            "depth_tier": tier,
            "depth_mm": round(depth_mm, 4),
            "organ_constraint_passed": params["organ_constraint_passed"],
            "foci_details": params.get("foci", []),
        }

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{n_total} samples...")

    print(f"\nDone! All {n_total} samples saved to {samples_dir}")

    # ============ Verify distributions ============
    foci_verify = {1: 0, 2: 0, 3: 0}
    depth_verify = {"shallow": 0, "medium": 0, "deep": 0}
    organ_failures = 0

    for i in range(n_total):
        params = json.load(open(samples_dir / f"sample_{i:04d}" / "tumor_params.json"))
        n = params.get("num_foci", len(params.get("foci", [])))
        foci_verify[n] = foci_verify.get(n, 0) + 1
        tier = params.get("depth_tier", "unknown")
        depth_verify[tier] = depth_verify.get(tier, 0) + 1
        if not params.get("organ_constraint_passed", True):
            organ_failures += 1

    print(f"\nVerification:")
    print(f"  Foci distribution: {foci_verify}")
    print(f"  Depth tier distribution: {depth_verify}")
    print(f"  Organ constraint failures: {organ_failures}")

    # ============ Create stratified split ============
    print("\nCreating stratified split...")

    # Group by foci count
    by_foci = {1: [], 2: [], 3: []}
    for name, info in sample_metadata.items():
        n = info["num_foci"]
        if n in by_foci:
            by_foci[n].append(name)

    # Shuffle within each group
    rng = np.random.default_rng(42)
    for n in by_foci:
        rng.shuffle(by_foci[n])

    # Stratified split: 80/20
    train_counts = {1: 48, 2: 56, 3: 56}
    val_counts = {1: 12, 2: 14, 3: 14}

    train_samples = []
    val_samples = []

    for n_foci, count in train_counts.items():
        train_samples.extend(by_foci[n_foci][:count])

    for n_foci, count in val_counts.items():
        val_samples.extend(by_foci[n_foci][:count])

    # Verify val coverage
    val_by_foci = {1: 0, 2: 0, 3: 0}
    val_by_depth = {"shallow": 0, "medium": 0, "deep": 0}
    for name in val_samples:
        val_by_foci[sample_metadata[name]["num_foci"]] += 1
        val_by_depth[sample_metadata[name]["depth_tier"]] += 1

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples ({val_by_foci})")
    print(f"  Val by depth tier: {val_by_depth}")

    # Write split files
    splits_dir = Path("train/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train.txt", "w") as f:
        for name in sorted(train_samples):
            f.write(name + "\n")

    with open(splits_dir / "val.txt", "w") as f:
        for name in sorted(val_samples):
            f.write(name + "\n")

    with open(splits_dir / "train_with_foci.txt", "w") as f:
        for name in sorted(train_samples):
            n = sample_metadata[name]["num_foci"]
            tier = sample_metadata[name]["depth_tier"]
            f.write(f"{name}\t{n}\t{tier}\n")

    with open(splits_dir / "val_with_foci.txt", "w") as f:
        for name in sorted(val_samples):
            n = sample_metadata[name]["num_foci"]
            tier = sample_metadata[name]["depth_tier"]
            f.write(f"{name}\t{n}\t{tier}\n")

    # ============ Create manifest ============
    train_by_depth = {"shallow": 0, "medium": 0, "deep": 0}
    for name in train_samples:
        train_by_depth[sample_metadata[name]["depth_tier"]] += 1

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": n_total,
        "seed_foci": 42,
        "seed_depth": 43,
        "foci_distribution": {
            "1": n_1foci,
            "2": n_2foci,
            "3": n_3foci,
        },
        "depth_distribution": {
            "shallow": 70,
            "medium": 80,
            "deep": 50,
        },
        "split": {
            "train": 160,
            "val": 40,
            "train_by_foci": train_counts,
            "val_by_foci": val_counts,
            "train_by_depth": train_by_depth,
            "val_by_depth": val_by_depth,
        },
        "samples": {},
    }

    train_set = set(train_samples)
    for name, info in sorted(sample_metadata.items()):
        split = "train" if name in train_set else "val"
        manifest["samples"][name] = {
            "num_foci": info["num_foci"],
            "depth_tier": info["depth_tier"],
            "depth_mm": info["depth_mm"],
            "organ_constraint_passed": info["organ_constraint_passed"],
            "split": split,
            "foci_details": info["foci_details"],
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
