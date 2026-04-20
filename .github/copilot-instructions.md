# Copilot instructions for FMT-SimGen

## Build, test, and lint commands

Use `uv run python` for all project scripts (do not use system/miniforge Python).

```bash
# Install dependencies
pip install -r requirements.txt

# "Lint" equivalent used in this repo: syntax check all Python files
for f in $(find . -name "*.py" -type f); do uv run python -m py_compile "$f"; done

# Import smoke test for core package exports
uv run python -c "from fmt_simgen import DatasetBuilder, TurntableCamera; print('OK')"
```

There is no formal unit-test suite (no pytest/unittest test tree). Validation is script-based:

```bash
# End-to-end dual-channel generation
uv run python scripts/run_all.py -n 50

# Full dataset validation (integrity + stats + figures)
uv run python scripts/validate_dataset.py --data_dir data/<experiment> --shared_dir output/shared

# Single-sample style verification
uv run python scripts/verify_dual_channel.py \
  --shared_dir output/shared \
  --samples_dir data/<experiment>/samples \
  --output_dir output/verification \
  --n_samples 1
```

## High-level architecture

FMT-SimGen is a **dual-channel synthetic data pipeline** sharing one tumor definition:

1. **Shared assets (Step 0)**: mesh/system matrix/voxel grid + MCX trunk volume + view config are generated once and stored under `output/shared/`.
2. **DE channel (Steps 1–4)**: `DatasetBuilder` (`fmt_simgen/dataset/builder.py`) generates each sample by
   - sampling analytic tumors (`tumor_generator.py`)
   - sampling GT on FEM nodes and voxel grid from the same analytic function (`dual_sampler.py`)
   - solving forward DE measurement (`fem_solver.py`)
3. **MCX channel (Steps 2m–4m)**: from each sample’s `tumor_params.json`,
   - `mcx_config.py` writes `{sample}.json` + `source-{sample}.bin`
   - `mcx_runner.py` runs MCX and produces `{session}.jnii`
   - `mcx_projection.py` creates multi-angle `proj.npz`
4. `scripts/run_all.py` orchestrates DE first, then MCX via `scripts/run_mcx_pipeline.py`.

Per-sample directory contract (under `data/<experiment>/samples/sample_XXXX/`) is central to integration: `tumor_params.json`, `measurement_b.npy`, `gt_nodes.npy`, `gt_voxels.npy`, and MCX artifacts (`*.json`, `source-*.bin`, `*.jnii`, `proj.npz`).

## Key conventions (repo-specific)

- **World/frame convention is strict**: use `mcx_trunk_local_mm` as the working world frame. `frame_manifest.json` in `output/shared/` is the authoritative metadata for frame, bbox, voxel grid offset/spacing.
- **Coordinate/order convention**:
  - MCX volume file shape is ZYX (`[104,200,190]`), while most internal processing uses XYZ.
  - `.jnii` is loaded then transposed with `transpose(2, 1, 0)` to XYZ.
  - Tumor centers saved in `tumor_params.json` are trunk-local (`center`); `center_atlas_mm` is retained for debugging.
- **Config pattern**: experiment configs support `_base_` inheritance + recursive deep merge (see `run_all.py` and `02_generate_dataset.py`). Do not duplicate config blocks manually.
- **Output isolation**: all generated datasets are written to `data/{experiment_name}/...` (from `dataset.experiment_name`), not a flat `data/`.
- **Visibility filtering is intentional**: when `view_config.angles` is configured, DE `measurement_b.npy` is filtered to union-visible surface nodes (`visible_mask.npy` in `output/shared/`).
- **MCX path handling**: generated MCX JSON stores `Domain.VolumeFile` relative to each sample directory so runs are portable across experiment folders.
