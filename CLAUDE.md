# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMT-SimGen generates synthetic Fluorescence Molecular Tomography (FMT) datasets for training reconstruction algorithms. It produces surface measurements (b), ground truth at FEM nodes (gt_nodes), and ground truth at voxels (gt_voxels) from analytic tumor definitions (Gaussian spheres/ellipsoids).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate mesh and system matrix (run once before samples)
python scripts/01_generate_mesh.py

# Generate dataset samples
python scripts/02_generate_dataset.py --num_samples 100

# Verify dataset
python scripts/03_verify_dataset.py --sample_idx 0

# Verify all imports work correctly
cd /home/foods/pro/FMT-SimGen && uv run python -c "
import sys; sys.path.insert(0, '.')
from fmt_simgen import (DigimouseAtlas, MeshGenerator, FEMSolver, OpticalParameterManager,
    TumorGenerator, TumorSample, AnalyticFocus, DualSampler, DatasetBuilder)
print('All imports successful!')
"
```

## Architecture

The pipeline is orchestrated by `DatasetBuilder` in `fmt_simgen/dataset/builder.py`. It coordinates:

1. **Atlas** (`fmt_simgen/atlas/digimouse.py`) - Loads Digimouse whole-body atlas (380×992×208 voxels at 0.1mm)
2. **Mesh** (`fmt_simgen/mesh/mesh_generator.py`) - Generates tetrahedral FEM mesh via iso2mesh
3. **Physics** (`fmt_simgen/physics/fem_solver.py`) - Assembles and solves the diffusion equation FEM system
4. **Tumor** (`fmt_simgen/tumor/tumor_generator.py`) - Generates analytic Gaussian sphere/ellipsoid tumors
5. **Sampling** (`fmt_simgen/sampling/dual_sampler.py`) - Dual-carrier GT sampling at FEM nodes and voxel grid
6. **Dataset** (`fmt_simgen/dataset/builder.py`) - Orchestrates the full pipeline

Output from `01_generate_mesh.py`:
- `assets/mesh/mesh.npz` - Tetrahedral mesh (nodes, elements, surface faces)
- `assets/mesh/system_matrix.*.npz` - Pre-assembled FEM system matrices

Output from `02_generate_dataset.py` (per sample):
- `measurement_b.npy` - Surface fluence measurements
- `gt_nodes.npy` - Ground truth at FEM nodes
- `gt_voxels.npy` - Ground truth at voxel grid
- `tumor_params.json` - Tumor parameters

## Configuration

All parameters are in `config/default.yaml`. Key sections:
- `atlas.path` - Digimouse atlas location (requires external data at `/home/foods/pro/mcx_simulation/ct_data/`)
- `mesh` - Target node count (~10000), volume constraints
- `physics` - Optical parameters (μ_a, μ_s', g, n) per tissue type
- `tumor` - Number of foci distribution, shapes, radius/depth ranges
- `dataset` - Number of samples, voxel spacing, output path

## Code Style

- Classes: PascalCase (e.g., `DigimouseAtlas`, `FEMSolver`)
- Functions/methods/variables: snake_case
- Type annotations required for all function parameters and returns
- Use `logging` module (not print) with appropriate levels
- Google-style docstrings with parameter/return sections
-具体异常类型 for errors (not bare `except Exception`)

## Important Notes

**Python environment**: Use `uv run` or activate the `.venv` via `source .venv/bin/activate`. All scripts must be run with `uv run python` or `python` (after venv activation), NOT system python or miniforge3 python directly. Example:
```bash
uv run python scripts/01_generate_mesh.py
```

**Volume data order**: Digimouse volume is `[X, Y, Z]` where:
- X = Left(-19mm)→Right(+19mm), 380 voxels
- Y = Anterior(-50mm)→Posterior(+50mm), 992 voxels
- Z = Inferior(-10mm)→Superior(+11mm), 208 voxels
- Dorsal(back)=+Z, Ventral(belly)=-Z

**Float division trap**: Use `int(round(x))` not `int(x)` when converting floats to voxel counts.
