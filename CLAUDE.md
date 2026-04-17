# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMT-SimGen generates synthetic Fluorescence Molecular Tomography (FMT) datasets with **dual-channel** output:

- **DE channel**: Surface fluence measurements (b), ground truth at FEM nodes (gt_nodes), ground truth at voxels (gt_voxels) — from analytic tumor definitions
- **MCX channel**: Monte Carlo photon simulation fluence volumes (.jnii) + multi-angle 2D detector projections (proj.npz) — from MCX GPU simulation

Reference implementations: `/home/foods/pro/mcx_simulation` (P1: MCX+DE with Digimouse brain) and `/home/foods/pro/mcx_simulation/data_build` (MS-GDUN: DE with Gaussian sphere tumors).

## Common Commands

```bash
# Verify imports
uv run python -c "from fmt_simgen import DatasetBuilder, TurntableCamera; print('OK')"

# Asset generation (once per atlas/mesh change)
uv run python scripts/step0b_generate_mesh.py           # Tetrahedral mesh
uv run python scripts/step0c_fem_matrix.py              # FEM system matrix
uv run python scripts/step0d_voxel_grid.py              # Voxel grid
uv run python scripts/step0e_v2_full_graph_laplacian.py  # Graph Laplacian
uv run python scripts/step0f_mcx_volume.py               # MCX trunk volume binary
uv run python scripts/step0g_view_config.py             # TurntableCamera config

# DE channel samples
uv run python scripts/02_generate_dataset.py -n 50

# Full dual-channel (DE + MCX)
uv run python scripts/run_all.py -n 50 --enable_mcx

# MCX channel only (on existing DE samples)
uv run python scripts/run_mcx_pipeline.py \
  --samples_dir data/gaussian_1000/samples --projection_only
```

## Architecture: Dual-Channel Pipeline

### Shared assets (Step 0)
| Step | Script | Output |
|------|---------|--------|
| 0b | `step0b_generate_mesh.py` | `assets/mesh/mesh.npz` |
| 0c | `step0c_fem_matrix.py` | `assets/mesh/system_matrix.*.npz` |
| 0d | `step0d_voxel_grid.py` | `assets/mesh/voxel_grid.npz` |
| 0e | `step0e_v2_full_graph_laplacian.py` | Graph Laplacian for regularization |
| 0f | `step0f_mcx_volume.py` | `output/shared/mcx_volume_trunk.bin` (trunk-cropped atlas) |
| 0g | `step0g_view_config.py` | `output/shared/view_config.json` (camera model) |

### DE channel (Steps 1–4)
- `DatasetBuilder` (`fmt_simgen/dataset/builder.py`) orchestrates per-sample generation
- Per-sample output: `tumor_params.json`, `measurement_b.npy`, `gt_nodes.npy`, `gt_voxels.npy`

### MCX channel (Steps 2m–4m)
| Step | Script | Module | Output |
|------|--------|--------|--------|
| 2m | `step2m_generate_mcx_sources.py` | `mcx_config.py` | `{id}.json` + `source-{id}.bin` |
| 3m | `run_mcx_pipeline.py --simulation_only` | `mcx_runner.py` | `{id}.jnii` (3D fluence) |
| 4m | `run_mcx_pipeline.py --projection_only` | `mcx_projection.py` | `proj.npz` (7-angle projections) |

### Dual-channel entry points
- `scripts/run_all.py` — orchestrates DE + MCX with `--enable_mcx`
- `scripts/run_mcx_pipeline.py` — standalone MCX (2m→3m→4m), supports `--projection_only`

## Key Coordinate Systems

### Digimouse atlas volume `[X=380, Y=992, Z=208]`
- X: Left(−19mm)→Right(+19mm), Y: Anterior(−50mm)→Posterior(+50mm), Z: Inferior(−10mm)→Superior(+11mm)
- Dorsal(back)=+Z, Ventral(belly)=−Z

### MCX trunk volume (after Step 0f crop + downsample)
- Shape: `[Z=104, Y=200, X=190]`, voxel_size=0.2mm
- JNII→XYZ: `nifti.transpose(2, 1, 0)` → `[X=190, Y=200, Z=104]`
- Physical origin (voxel [0,0,0] world position): `trunk_offset_mm = [0, 30, 0]`

### MCX projection camera (TurntableCamera)
- Camera at `[0, 0, D]`, looking toward origin along −Z, rotation around Y axis
- Angles: `[-90, -60, -30, 0, 30, 60, 90]` degrees
- Orthographic: all rays parallel to −Z, keeps shallowest non-zero voxel per pixel

## Per-Sample Output Structure

```
data/{experiment}/samples/sample_XXXX/
├── tumor_params.json        # Shared: tumor parameters
├── measurement_b.npy        # DE: surface fluence [N_d]
├── gt_nodes.npy             # DE: GT at FEM nodes [N_n]
├── gt_voxels.npy            # DE: GT at voxels [Nx×Ny×Nz]
├── sample_XXXX.json         # MCX: simulation config
├── source-sample_XXXX.bin   # MCX: source pattern binary
├── sample_XXXX.jnii         # MCX: fluence volume (JNII format)
└── proj.npz                 # MCX: 7-angle projections {"-90": [H,W], ...}
```

## Configuration

All parameters in `config/default.yaml`. MCX-specific keys:
- `mcx.trunk_offset_mm`: physical offset `[0, 30, 0]` for MCX volume origin
- `mcx.voxel_size_mm`: 0.2mm (2× downsample from 0.1mm atlas)
- `mcx.volume_shape`: `[104, 200, 190]` (Z, Y, X)
- `view_config.angles`: `[-90, -60, -30, 0, 30, 60, 90]`
- `view_config.pose`: `"prone"` or `"supine"`

## MCX Module Reference

| Module | Role |
|--------|------|
| `mcx_volume.py` | Load atlas, crop trunk, downsample, save binary volume |
| `mcx_config.py` | Generate MCX JSON config and source binary from tumor_params |
| `mcx_source.py` | Pattern3D fluence source definition |
| `mcx_runner.py` | MCX CLI invocation, auto-detects `mcx` (GPU) vs `mcxcl` (OpenCL/CPU) |
| `mcx_projection.py` | Orthographic projection from fluence volume to 7-angle proj.npz |
| `view_config.py` | TurntableCamera: pose, occlusion, surface normals, project_volume |

## Pilot Experiments

The `pilot/` directory contains ad-hoc experimental investigations, not part of the main pipeline:
- `pilot/e0_psf_validation/` — PSF vs MCX point-source comparison
- `pilot/e1b_model_mismatch/` — Model-mismatch analysis
- `pilot/e1c_green_function_selection/` — Kernel selection experiments
- `pilot/e1d_finite_source_local_surface/` — Atlas-aware surface rendering (SR-6/UT-7 quadrature)
- `pilot/visualization/` — Plotting scripts for experimental results

## Testing

This project has no formal test suite. Validation is done via:
- `scripts/03_verify_dataset.py` — per-sample data verification
- ad-hoc pilot scripts for experimental validation

## MCX Executable

MCX binary: `/mnt/f/win-pro/bin/mcx.exe`. Invoke via `subprocess.run(["mcx.exe", "-f", config.json"], cwd=work_dir)`. Auto-detected by `mcx_runner.py` — uses `mcx` for GPU, `mcxcl` for CPU fallback.

## Python Environment

**Always** use `uv run python`, never system python or miniforge python:
```bash
uv run python scripts/run_all.py -n 50 --enable_mcx
```

## Code Style

- Classes: PascalCase, functions/variables: snake_case
- Type annotations required on all function parameters and returns
- Use `logging` (not print), module-level `logger = logging.getLogger(__name__)`
- Specific exception types (not bare `except Exception`)
- Float division: `int(round(x))` not `int(x)` for voxel count conversion
- See `AGENTS.md` for full style guide (imports, docstrings, data structures)

## Import Verification

```bash
uv run python -c "
from fmt_simgen import (
    DigimouseAtlas, MeshGenerator, FEMSolver,
    OpticalParameterManager, TumorGenerator, TumorSample,
    AnalyticFocus, DualSampler, DatasetBuilder, TurntableCamera
)
print('OK')
"
