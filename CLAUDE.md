# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMT-SimGen generates synthetic Fluorescence Molecular Tomography (FMT) datasets with **dual-channel** output:

- **DE channel**: Surface fluence measurements (b), ground truth at FEM nodes (gt_nodes), ground truth at voxels (gt_voxels) — from analytic tumor definitions
- **MCX channel**: Monte Carlo photon simulation fluence volumes (.jnii) + multi-angle 2D detector projections (proj.npz) — from MCX GPU simulation

## Common Commands

```bash
# Verify imports
uv run python -c "from fmt_simgen import DatasetBuilder, TurntableCamera; print('OK')"

# Asset generation (once per atlas/mesh change)
uv run python scripts/step0b_generate_mesh_cgalmesh.py  # Tetrahedral mesh (cgalmesh backend)
uv run python scripts/step0c_fem_matrix.py              # FEM system matrix
uv run python scripts/step0d_voxel_grid.py              # Voxel grid
uv run python scripts/step0e_v2_full_graph_laplacian.py  # Graph Laplacian
uv run python scripts/step0f_mcx_volume.py             # MCX trunk volume binary

# DE channel samples
uv run python scripts/02_generate_dataset.py -n 50

# Full dual-channel (DE + MCX)
uv run python scripts/run_all.py -n 50 --enable_mcx

# MCX channel only (on existing DE samples)
uv run python scripts/run_mcx_pipeline.py \
  --samples_dir data/gaussian_1000/samples --simulation_only
uv run python scripts/run_mcx_pipeline.py \
  --samples_dir data/gaussian_1000/samples --projection_only
```

## Architecture: Dual-Channel Pipeline

### Shared assets (Step 0) — generated once per atlas
| Step | Script | Output |
|------|---------|--------|
| 0b | `step0b_generate_mesh_cgalmesh.py` | `output/shared/mesh.npz` (cgalmesh, ~52k nodes, ~295k tets) |
| 0c | `step0c_fem_matrix.py` | `output/shared/system_matrix.*.npz` |
| 0d | `step0d_voxel_grid.py` | `output/shared/voxel_grid.npz` |
| 0e | `step0e_v2_full_graph_laplacian.py` | Graph Laplacian for regularization |
| 0f | `step0f_mcx_volume.py` | `output/shared/mcx_volume_trunk.bin` (uint8 labels, ZYX order) |

### DE channel (Steps 1–4)
- `DatasetBuilder` (`fmt_simgen/dataset/builder.py`) orchestrates per-sample generation
- Per-sample output: `tumor_params.json`, `measurement_b.npy`, `gt_nodes.npy`, `gt_voxels.npy`

### MCX channel (Steps 2m–4m)
| Step | Script | Module | Output |
|------|--------|--------|--------|
| 2m | `run_mcx_pipeline.py` (source generation) | `mcx_config.py` | `source-{id}.bin` |
| 3m | `run_mcx_pipeline.py --simulation_only` | `mcx_runner.py` | `{id}.jnii` (3D fluence) |
| 4m | `run_mcx_pipeline.py --projection_only` | `mcx_projection.py` | `proj.npz` (7-angle projections) |

### Dual-channel entry points
- `scripts/run_all.py` — orchestrates DE + MCX with `--enable_mcx`
- `scripts/run_mcx_pipeline.py` — standalone MCX (2m→3m→4m), supports `--simulation_only` and `--projection_only`

## Data Flow

```
分割文件 (npz/nii): shape=(190,200,104) @ 0.2mm, origin=(0,0,0)
         ↓
    ┌─────────────────────────────────────────┐
    │  Step 0: 共享资源生成（一次性）           │
    │  step0b: mesh.npz                       │
    │  step0f: mcx_volume_trunk.bin           │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │  DE Channel (per sample)                 │
    │  tumor_params.json → gt_voxels.npy      │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │  MCX Channel (per sample)                │
    │  source-{id}.bin → {id}.jnii → proj.npz │
    └─────────────────────────────────────────┘
```

## Coordinate System (origin = (0, 0, 0))

| Volume | Shape (XYZ) | Voxel size | Origin | 物理范围 |
|--------|------------|------------|--------|---------|
| 分割文件 | (190,200,104) | 0.2mm | (0,0,0) | X=[0,38], Y=[0,40], Z=[0,20.8] |
| MCX Volume | (190,200,104) | 0.2mm | (0,0,0) | 同上 |
| DE gt_voxels | (190,200,104) | 0.2mm | (0,0,0) | 同上 |
| DE FEM mesh | — | — | (0,0,0) | X=[2.8,34.2], Y=[0.1,39.9], Z=[1.6,20.6] |

**体素 [x,y,z] 中心物理坐标:** `(x+0.5)*0.2, (y+0.5)*0.2, (z+0.5)*0.2`

**关键常数 (frame_contract.py):**
```python
VOXEL_SIZE_MM          = 0.2
TRUNK_GRID_SHAPE       = (190, 200, 104)  # MCX/DE 共用，XYZ 顺序
VOLUME_EXTENTS_MM      = [38.0, 40.0, 20.8]  # = TRUNK_GRID_SHAPE * 0.2
VOLUME_CENTER_WORLD    = [19.0, 20.0, 10.4]  # = VOLUME_EXTENTS_MM / 2，投影旋转中心
CAMERA_DISTANCE_MM     = 200.0  # 相机到转台中心距离
FOV_MM                 = 80.0   # 视野范围 (mm)
DETECTOR_RESOLUTION    = (256, 256)  # 探测器分辨率 [宽, 高]
ANGLES                 = [-90, -60, -30, 0, 30, 60, 90]  # 转台角度 (度)
```

**MCX Pattern 坐标映射:**
```python
# Pattern 生成: pattern[x, y, z] (XYZ 顺序，float32)
# 写入文件时转置: pattern.transpose(2,1,0) → ZYX 顺序
# MCX JSON Pos = [z0, y0, x0] (ZYX 顺序)
# MCX 映射: pattern[x,y,z] → volume[Pos_z+x, Pos_y+y, Pos_x+z]
# 即: origin voxel (x0,y0,z0) 在 JSON 中写成 [z0,y0,x0]
```

**投影坐标变换 (mcx_projection.py):**
```python
# 1. 体素中心 → 物理 mm (corner-origin frame)
world_mm = (voxel_indices + 0.5) * VOXEL_SIZE_MM  # [X, Y, Z] mm

# 2. 移至以体积中心为原点
world_mm[:, 0] -= VOLUME_CENTER_WORLD[0]  # 19.0
world_mm[:, 1] -= VOLUME_CENTER_WORLD[1]  # 20.0
world_mm[:, 2] -= VOLUME_CENTER_WORLD[2]  # 10.4

# 3. 绕 Y 轴旋转 angle 度
R = [[ cos_a, 0, sin_a],   # x' = x*cos + z*sin
     [ 0,     1, 0    ],
     [-sin_a, 0, cos_a]]   # z' = -x*sin + z*cos

# 4. 相机深度 = CAMERA_DISTANCE_MM - z_rot
#    同一像素多个体素时，保留最浅（最小depth）
```

## MCX Volume 配置 (config/default.yaml)

```yaml
mcx:
  volume_path: "output/shared/mcx_volume_trunk.bin"
  material_path: "output/shared/mcx_material.yaml"

  # 躯干裁剪（Y 方向，atlas 0.1mm/voxel → 2× 下采样后 0.2mm/voxel）
  trunk_crop:
    y_start: 340   # → 34mm in atlas coordinates
    y_end: 740     # → 74mm in atlas coordinates

  downsample_factor: 2
  voxel_size_mm: 0.2
  trunk_offset_mm: [0, 34, 0]  # Y offset = 34mm (crop start 位置)
  volume_shape: [104, 200, 190]  # MCX Dim 格式 (Z, Y, X)
  num_tissues: 10  # labels 0-9

view_config:
  angles: [-90, -60, -30, 0, 30, 60, 90]
  pose: prone  # 或 supine
  camera_distance_mm: 200
  detector_resolution: [256, 256]
  projection_type: orthographic
  platform_occlusion: true
  fov_mm: 80
  platform_z_center: 4.0
```

**Volume shape 计算:**
- 原始 atlas: (X=380, Y=400, Z=208) @ 0.1mm
- Y 裁剪 [340, 740) → 400 voxels @ 0.1mm → 200 voxels @ 0.2mm
- 2× 下采样后: (X=190, Y=200, Z=104)
- MCX Dim = [104, 200, 190] (ZYX 顺序)

## Per-Sample Output Structure

```
data/{experiment}/samples/sample_XXXX/
├── tumor_params.json        # DE: tumor parameters
├── measurement_b.npy        # DE: surface fluence [N_d]
├── gt_nodes.npy             # DE: GT at FEM nodes [N_n]
├── gt_voxels.npy            # DE: GT at voxels [190×200×104], origin=(0,0,0)
├── source-XXXX.bin          # MCX: source pattern (ZYX order)
├── all_in_one.bin           # MCX: volume with tumor labeled as 15 (ZYX order)
├── sample_XXXX.jnii         # MCX: fluence volume (JNII format)
└── proj.npz                 # MCX: 7-angle projections {"-90": [H,W], ...}
```

## Configuration

All parameters in `config/default.yaml`. Experiment configs support `_base_` inheritance + recursive deep merge (see `run_all.py` and `02_generate_dataset.py`). Do not duplicate config blocks manually.

## MCX Module Reference

### mcx_source.py — Pattern3D 光源生成
**函数:** `tumor_params_to_mcx_pattern(tumor_params, volume_shape, voxel_size_mm)`

**输入:** `tumor_params.json` (物理 mm 单位，trunk-local 坐标)
```json
{
  "source_type": "gaussian" | "uniform",
  "foci": [{"center": [px,py,pz], "shape": "sphere"|"ellipsoid", "radius": float, "rx": float, ...}]
}
```

**输出:** `(pattern, origin)`
- `pattern`: float32 array, shape `(nx, ny, nz)` — **XYZ 顺序**
- `origin`: `(x0, y0, z0)` — pattern 在体积中的起始 voxel 坐标 (XYZ 顺序)

**算法:**
- 多焦点: `np.maximum` blend（取最亮焦点）
- Gaussian: `exp(-0.5*(dist/sigma)^2)`，3σ 截断
- Uniform sphere: 半径内 1.0，超出 0.0
- Uniform ellipsoid: `sum((d/r)^2) <= 1` 区域内 1.0
- 1% 阈值裁剪，小于 `0.01 * pattern_max` 的体素置零
- bbox 扩展 2 voxel padding

**输出文件:** `source-{id}.bin`
```python
# XYZ → ZYX 转置写入 (float32, 无 header)
pattern.transpose(2, 1, 0).tofile(output_dir / f"source-{id}.bin")
```

### mcx_config.py — MCX JSON 配置生成
**函数:** `generate_mcx_config(output_dir, mcx_config, sample_id, pattern_origin)`

**输出文件:** `{sample_id}.json`

**JSON 结构:**
```json
{
  "Domain": {
    "VolumeFile": "../../shared/mcx_volume_trunk.bin",  // 相对路径
    "Dim": [104, 200, 190],          // ZYX 顺序 = TRUNK_GRID_SHAPE[::-1]
    "OriginType": 1,                  // bbox corner 为原点
    "LengthUnit": 0.2,               // mm
    "Media": [...]                    // 来自 mcx_material.yaml
  },
  "Session": {
    "Photons": 10000000,
    "RNGSeed": <hash-derived>,
    "ID": "sample_id"
  },
  "Forward": {
    "T0": 0.0,
    "T1": 5.0e-08,   // 50 ns
    "DT": 5.0e-08
  },
  "Optode": {
    "Source": {
      "Pos": [z0, y0, x0],           // ZYX 顺序（MCX Dim 是 ZYX）
      "Dir": [0, 0, 1, "_NaN_"],      // +Z 方向
      "Type": "pattern3d",
      "Pattern": {
        "Nx": pnx, "Ny": pny, "Nz": pnz,  // pattern 形状 (Z,Y,X)
        "Data": "source-{id}.bin"          // 相对路径
      },
      "Param1": [pnx, pny, pnz]
    }
  }
}
```

**关键映射:**
```python
# origin (x0, y0, z0) 是 XYZ 顺序的 voxel 坐标
# JSON Pos 必须是 ZYX 顺序: [z0, y0, x0]
# MCX 内部: pattern[x,y,z] → volume[Pos_z+x, Pos_y+y, Pos_x+z]
```

### mcx_runner.py — MCX 执行
**函数:** `run_mcx_single(work_dir, json_path, session_id)`

**命令:**
```bash
mcx -f {json_name} -a 1   # -a 1: overwrite 模式
```
**自动检测:** `mcx` (CUDA/GPU) → `mcxcl` (OpenCL/CPU fallback)

**输出:** `{session_id}.jnii` — JNII 格式 (JSON + NIfTI)，**ZYX 顺序**

**读取转换:**
```python
fluence_xyz = jdata.loadjd(jnii_path).transpose(2, 1, 0)  # ZYX → XYZ, shape (190,200,104)
```

**幂等:** 若 `{session_id}.jnii` 已存在则跳过

### mcx_projection.py — 正交投影生成
**函数:** `project_mcx_fluence(fluence_xyz, angles, ...)` → `proj.npz`

**输入:** `fluence_xyz` shape `(190, 200, 104)` float32 (XYZ 顺序)

**相机几何:**
| 参数 | 值 |
|------|-----|
| 相机位置 | `[0, 0, 200]` mm |
| 投影类型 | 正交 (parallel rays) |
| 旋转轴 | Y 轴 |
| 角度 | `[-90, -60, -30, 0, 30, 60, 90]` 度 |
| FOV | 80 mm (square) |
| 像素尺寸 | `80/256 = 0.3125` mm |
| 探测器 | 256×256 pixels |

**投影算法:**
- 每个非零体素填满其物理覆盖的所有像素
- 同一像素多个体素时，保留**最浅**（最小 camera depth）

**输出 `proj.npz`:**
```
keys: "0", "30", "60", "90", "-30", "-60", "-90"
      "depth_0", "depth_30", ...  (可选)
shape per key: (256, 256) float32
```

### view_config.py — TurntableCamera
| 函数 | 作用 |
|------|------|
| `load_volume()` | 加载 jnii 并转置为 XYZ |
| `project_volume()` | 调用 `mcx_projection.project_mcx_fluence()` |
| `compute_surface_normals()` | 计算表面法向量 |
| `apply_pose_occlusion()` | 根据 pose (prone/supine) 遮挡 |

### mcx_volume.py — 躯干体积生成
**输入:** Digimouse atlas 分割文件 (0.1mm/voxel)
**处理:**
1. Y 方向裁剪: voxel [340, 740) → physical [34mm, 74mm]
2. 2×2×2 majority-vote 下采样 → 0.2mm/voxel
3. 原始 22 类 → 10 类组织映射
**输出:** `output/shared/mcx_volume_trunk.bin` — uint8, shape `(104, 200, 190)` ZYX 顺序

## Verification

```bash
# 3D DE vs MCX alignment (requires all_in_one.bin in sample dir)
for i in 0 1 2 3 4; do
  uv run python scripts/verify_3d_de_mcx_alignment.py --sample sample_000$i
done

# 2D projection comparison
for i in 0 1 2 3 4; do
  uv run python scripts/de_surface_to_mcx_projection.py --sample sample_000$i
done
```

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

Float division pitfall:
```python
# Correct — avoids truncation bug
depth_min_vox = int(round(1.0 / 0.1))  # = 10

# Wrong — int() truncates
depth_min_vox = int(1.0 / 0.1)          # = 9
```

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
```
