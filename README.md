# FMT-SimGen

Fluorescence Molecular Tomography (FMT) Simulation Dataset Generator with dual-carrier ground truth sampling.

## Overview

FMT-SimGen generates synthetic FMT datasets for training and evaluation of reconstruction algorithms. It produces:

1. **Surface measurements** (b): Fluence values at detector positions from forward diffusion equation solve
2. **Ground truth at FEM nodes** (gt_nodes): Analytic tumor function sampled at tetrahedral mesh nodes
3. **Ground truth at voxels** (gt_voxels): Same analytic tumor function sampled at regular voxel grid

Key design features:
- Tumors defined as **continuous analytic functions** (Gaussian spheres/ellipsoids) in 3D space
- Dual GT sampling ensures node-level and voxel-level GT are **perfectly aligned**
- Whole-body Digimouse atlas support (not just brain region)
- Adaptive tetrahedral mesh generation via iso2mesh

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
FMT-SimGen/
├── config/
│   └── default.yaml              # Global configuration
├── fmt_simgen/
│   ├── atlas/
│   │   └── digimouse.py          # Digimouse atlas loading
│   ├── mesh/
│   │   └── mesh_generator.py     # Tetrahedral mesh generation
│   ├── physics/
│   │   ├── optical_params.py     # Optical parameter management
│   │   └── fem_solver.py          # DE FEM solver
│   ├── tumor/
│   │   └── tumor_generator.py    # Analytic tumor generation
│   ├── sampling/
│   │   └── dual_sampler.py        # Dual-carrier GT sampling
│   ├── dataset/
│   │   └── builder.py            # Pipeline orchestration
│   └── utils/
│       └── io.py                 # I/O utilities
├── scripts/
│   ├── 01_generate_mesh.py       # Generate mesh + system matrix
│   ├── 02_generate_dataset.py     # Generate samples
│   └── 03_verify_dataset.py       # Verify data correctness
└── requirements.txt
```

## Usage

### 1. Generate Mesh and System Matrix (once)

```bash
python scripts/01_generate_mesh.py
```

This creates:
- `assets/mesh/mesh.npz` - Tetrahedral mesh (nodes, elements, surface faces)
- `assets/mesh/system_matrix.*.npz` - Pre-assembled FEM system matrix

### 2. Generate Dataset Samples

```bash
python scripts/02_generate_dataset.py --num_samples 100
```

This creates `data/sample_XXXX/` directories, each containing:
- `measurement_b.npy` - Surface fluence measurements [N_d]
- `gt_nodes.npy` - Ground truth at FEM nodes [N_n]
- `gt_voxels.npy` - Ground truth at voxel grid [Nx×Ny×Nz]
- `tumor_params.json` - Tumor parameters for reproducibility

### 3. Verify Dataset

```bash
python scripts/03_verify_dataset.py --sample_idx 0
```

## Configuration

All parameters are in `config/default.yaml`. Key sections:

- `atlas`: Digimouse path, tissue merge rules
- `mesh`: Target node count, volume constraints
- `physics`: Optical parameters (μ_a, μ_s', g, n) for 6 tissue types
- `tumor`: Number of foci distribution, shapes, size ranges, depth constraints
- `dataset`: Number of samples, noise settings, output path

## Dependencies

- numpy, scipy: Numerical computing
- nibabel: Atlas file I/O
- iso2mesh: Tetrahedral mesh generation
- pyyaml: Configuration file parsing

## References

This project references two prior implementations:
- **P1 (GISC-FMT MCX)**: `/home/foods/pro/mcx_simulation` - Monte Carlo + DE simulation with Digimouse brain region
- **MS-GDUN**: `/home/foods/pro/mcx_simulation/data_build` - DE finite element data generation with Gaussian sphere tumors

FMT-SimGen combines insights from both: analytic tumor definitions from MS-GDUN with whole-body atlas support from P1.

## Training Code

Training, evaluation, and visualization code has been migrated to [DU2Vox](https://github.com/foodszhang/DU2Vox).

## License

TBD
