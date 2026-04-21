# FMT-SimGen

Fluorescence Molecular Tomography (FMT) Simulation Dataset Generator with **dual-channel** ground truth.

## Overview

FMT-SimGen generates synthetic FMT datasets for training and evaluation of reconstruction algorithms. It produces two independent channels from the same tumor definition:

**DE channel** (Diffusion Equation):
- `measurement_b.npy` — surface fluence at detector positions
- `gt_nodes.npy` — ground truth at tetrahedral FEM mesh nodes
- `gt_voxels.npy` — ground truth at voxel grid (aligned with nodes)

**MCX channel** (Monte Carlo photon simulation):
- `{id}.jnii` — 3D fluence volume from GPU-accelerated MCX
- `proj.npz` — multi-angle 2D detector projections (7 angles)

Key design features:
- Tumors defined as **continuous analytic functions** (Gaussian spheres/ellipsoids) in 3D space
- **Dual GT sampling** ensures node-level and voxel-level GT are perfectly aligned
- Whole-body Digimouse atlas support with adaptive tetrahedral mesh generation
- **Dual-channel pipeline**: DE (physics-based) + MCX (wave-based) from shared `tumor_params.json`

## Quick Start

```bash
# Verify imports
uv run python -c "from fmt_simgen import DatasetBuilder, TurntableCamera; print('OK')"

# 1. Generate shared assets (once per atlas/mesh change)
uv run python scripts/step0b_generate_mesh.py       # Tetrahedral mesh
uv run python scripts/step0c_fem_matrix.py          # FEM system matrix
uv run python scripts/step0d_voxel_grid.py          # Voxel grid
uv run python scripts/step0f_mcx_volume.py          # MCX trunk volume binary
uv run python scripts/step0g_view_config.py         # TurntableCamera config

# 2. Generate DE channel samples
uv run python scripts/02_generate_dataset.py -n 50

# 3. (Optional) Run MCX channel on existing DE samples
uv run python scripts/run_mcx_pipeline.py --samples_dir data/gaussian_1000/samples --projection_only

# 4. Verify dataset
uv run python scripts/validate_dataset.py --data_dir data/gaussian_1000 --shared_dir output/shared
```

## Project Structure

```
FMT-SimGen/
├── config/
│   ├── default.yaml              # Base configuration (all parameters)
│   └── *.yaml                    # Experiment configs (inherit from default)
├── fmt_simgen/
│   ├── atlas/digimouse.py        # Digimouse atlas loading
│   ├── mesh/mesh_generator.py   # Tetrahedral mesh generation
│   ├── physics/
│   │   ├── optical_params.py     # Optical parameter management
│   │   └── fem_solver.py         # DE FEM solver
│   ├── tumor/tumor_generator.py  # Analytic tumor generation
│   ├── sampling/dual_sampler.py  # Dual-carrier GT sampling
│   ├── dataset/builder.py       # DE pipeline orchestration
│   ├── mcx_*.py                  # MCX channel modules
│   └── view_config.py            # TurntableCamera (projection geometry)
├── scripts/
│   ├── step0*.py                  # Shared asset generation (Step 0a–g)
│   ├── 02_generate_dataset.py   # DE channel sample generation
│   ├── run_all.py                # DE+MCX dual-channel entry point
│   ├── run_mcx_pipeline.py      # Standalone MCX channel
│   ├── validate_dataset.py       # Dataset integrity + stats + figures
│   └── verify_dual_channel.py   # Single-sample dual-channel verification
└── output/shared/                # Shared assets (gitignored)
    ├── frame_manifest.json       # Authoritative frame/bbox metadata
    ├── mcx_volume_trunk.bin      # MCX trunk volume binary
    └── view_config.json          # TurntableCamera config
```

## Usage

### 1. Generate Shared Assets (once)

```bash
uv run python scripts/step0b_generate_mesh.py
uv run python scripts/step0c_fem_matrix.py
uv run python scripts/step0d_voxel_grid.py
uv run python scripts/step0f_mcx_volume.py
uv run python scripts/step0g_view_config.py
```

This creates `output/shared/` containing mesh, FEM system matrix, voxel grid, MCX volume, and camera config.

### 2. Generate Dataset Samples

```bash
uv run python scripts/02_generate_dataset.py -n 100
```

Creates `data/{experiment}/samples/sample_XXXX/` directories, each containing:
- `tumor_params.json` — shared tumor parameters
- `measurement_b.npy` — surface fluence [N_d]
- `gt_nodes.npy` — ground truth at FEM nodes [N_n]
- `gt_voxels.npy` — ground truth at voxel grid [Nx×Ny×Nz]

### 3. (Optional) Run MCX Channel

```bash
uv run python scripts/run_mcx_pipeline.py \
  --samples_dir data/{experiment}/samples \
  --simulation_only   # produces *.jnii fluence volumes

uv run python scripts/run_mcx_pipeline.py \
  --samples_dir data/{experiment}/samples \
  --projection_only  # produces proj.npz from existing .jnii
```

### 4. Verify Dataset

```bash
uv run python scripts/validate_dataset.py \
  --data_dir data/{experiment} \
  --shared_dir output/shared
```

## Configuration

All parameters in `config/default.yaml`. Experiment configs (e.g. `config/gaussian_1000.yaml`) inherit from default via `_base_` + recursive deep merge — do not duplicate blocks manually.

Key sections:
- `atlas`: Digimouse path, tissue merge rules
- `mesh`: Target node count, volume constraints
- `physics`: Optical parameters (μ_a, μ_s', g, n) per tissue type
- `tumor`: Number of foci distribution, shapes, size ranges, depth constraints
- `mcx`: trunk offset, voxel size, volume shape (for MCX channel)
- `view_config`: turntable angles and pose (for MCX projections)

## Dependencies

```
numpy >= 1.21
scipy >= 1.7
nibabel >= 4.0
iso2mesh >= 0.1
meshio >= 5.0
pyyaml >= 6.0
```

Install with: `uv run pip install -r requirements.txt` (use `uv run python`, never system Python).

## References

Reference implementations:
- **P1 (GISC-FMT MCX)**: `/home/foods/pro/mcx_simulation` — MCX+DE with Digimouse brain region
- **MS-GDUN**: `/home/foods/pro/mcx_simulation/data_build` — DE with Gaussian sphere tumors

FMT-SimGen combines: analytic tumor definitions from MS-GDUN with whole-body atlas support from P1, plus GPU-accelerated MCX photon simulation.

Training, evaluation, and visualization code has migrated to [DU2Vox](https://github.com/foodszhang/DU2Vox).
