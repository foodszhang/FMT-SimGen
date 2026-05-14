# FMT-SimGen 代码库代理规范

本文件为在 FMT-SimGen 代码库中工作的 AI 代理提供规范和指南。

---

## 0. MCX 可执行文件

**MCX 路径**: `/mnt/f/win-pro/bin/mcx.exe`

使用方式：
```bash
mcx.exe -f config.json
```

在 Python 中调用：
```python
subprocess.run(["mcx.exe", "-f", "config.json"], cwd=work_dir)
```

---

## 1. 项目概述

**FMT-SimGen** 是一个用于 FMT（荧光分子断层成像）模拟数据集生成的 Python 库。
它从解析肿瘤函数生成合成数据，同时产生 FEM 节点级和体素级两组 ground truth。

**项目路径**: `/home/foods/pro/FMT-SimGen`

### 1.1 当前架构：Subject Manifest 驱动

**重要**：当前主流程已经从固定 Digimouse 几何切换为 **subject manifest 驱动**。后续开发不要再把 `190×200×104`、`0.2mm`、`[19,20,10.4]`、`[104,200,190]`、Digimouse label 数量等当作全局事实写死。

运行时几何唯一来源：
- `fmt_simgen/subject.py`
  - `SubjectManifest`
  - `VolumeSpec`
  - `LabelRoleSpec`
  - `load_subject_manifest(config, shared_dir)`
  - `subject_manifest_from_config(config)`
- 每个 shared output 目录下的 `frame_manifest.json`

兼容规则：
- 如果 config 中显式存在 `subject:`，以 `subject:` 为准，不被旧 `frame_manifest.json` 覆盖。
- 如果没有 `subject:`，legacy Digimouse 配置会从 `mcx.volume_shape`、`mcx.voxel_size_mm`、`mcx.trunk_offset_mm` 自动生成 manifest。
- `fmt_simgen/frame_contract.py` 只保留 legacy 兼容常量。新代码不得新增对这些常量的业务依赖；应从 `SubjectManifest` 读取 shape、voxel size、extent、center、label roles。

**关键依赖**:
- numpy >= 1.21.0
- scipy >= 1.7.0
- nibabel >= 4.0.0
- pyyaml >= 6.0
- iso2mesh >= 0.1.0
- meshio >= 5.0.0

---

## 2. 构建/测试/检查命令

### 2.1 依赖安装

```bash
pip install -r requirements.txt
```

### 2.2 Python 语法检查

检查所有 Python 文件的语法（项目根目录执行）:

```bash
for f in $(find . -name "*.py" -type f); do uv run python -m py_compile "$f" && echo "$f: OK"; done
```

### 2.3 导入验证

验证所有模块可正确导入:

```bash
cd /home/foods/pro/FMT-SimGen
uv run python -c "
import sys
sys.path.insert(0, '.')
from fmt_simgen import (
    DigimouseAtlas, MeshGenerator, FEMSolver,
    OpticalParameterManager, TumorGenerator, TumorSample,
    AnalyticFocus, DualSampler, DatasetBuilder, TurntableCamera,
    SubjectManifest, VolumeSpec, load_subject_manifest
)
print('All imports successful!')
"
```

### 2.4 脚本执行

使用 `uv run python`，**不要**用系统 python 或 miniforge3 python：

```bash
# Step 0a: 加载并分析 Digimouse atlas
uv run python scripts/run_step0a_atlas.py

# Step 0b: 生成 mesh（使用 iso2mesh cgalmesh）
uv run python scripts/step0b_generate_mesh_cgalmesh.py \
    --config config/default.yaml \
    --maxvol 5.0 --radbound 2.8 --distbound 2.5 \
    --output-name digimouse_trunk_mesh

# Step 0c-0e: 生成共享资产
uv run python scripts/step0c_fem_matrix.py
uv run python scripts/step0d_voxel_grid.py
uv run python scripts/step0e_v2_full_graph_laplacian.py

# Step 0f-0g: MCX 体积和相机资源
uv run python scripts/step0f_mcx_volume.py --config config/default.yaml
uv run python scripts/step0g_view_config.py --config config/default.yaml

# Step 1-4: DE 通道数据集生成
uv run python scripts/02_generate_dataset.py --config config/default.yaml -n 50

# MCX 通道（使用 run_mcx_pipeline.py）
uv run python scripts/run_mcx_pipeline.py --samples_dir data/gaussian_1000/samples

# 双通道（推荐）
uv run python scripts/run_all.py --config config/default.yaml -n 50 --phase all
```

### 2.5 多 Mesh 配置支持

**新架构**：支持多个 mesh 配置并存，每个 mesh 有独立的输出目录。

```bash
# 使用 20k mesh（独立目录）
uv run python scripts/step0c_fem_matrix.py \
    --mesh output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz \
    --output-dir output/shared_mesh_20k

uv run python scripts/step0g_view_config.py \
    --mesh output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz \
    --output-dir output/shared_mesh_20k

uv run python scripts/02_generate_dataset.py --config config/mesh_20k.yaml -n 50

uv run python scripts/run_mcx_pipeline.py \
    --samples_dir data/mesh_20k_test/samples \
    --shared-dir output/shared_mesh_20k

# 可视化验证
uv run python scripts/verify_3d_de_mcx_alignment.py \
    --sample sample_0000 \
    --samples_dir data/mesh_20k_test/samples \
    --mesh output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz \
    --output_dir output/visualizations/verification_20k

uv run python scripts/de_surface_to_mcx_projection.py \
    --sample sample_0000 \
    --samples_dir data/mesh_20k_test/samples \
    --mesh output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz \
    --output_dir output/visualizations/verification_20k
```

**支持 --mesh/--output-dir 的脚本**：
- `step0c_fem_matrix.py`
- `step0g_view_config.py`
- `step0d_voxel_grid.py`
- `step0e_v2_full_graph_laplacian.py`
- `run_mcx_pipeline.py` (使用 `--shared-dir`)
- `verify_3d_de_mcx_alignment.py`
- `de_surface_to_mcx_projection.py`

### 2.6 新 CT/分割结果接入方式

新数据推荐通过 `subject:` 配置接入，输入支持 NIfTI (`.nii/.nii.gz`) 和 NPZ。NIfTI 必须已预先配准/重采样到项目约定轴向；本项目当前不自动做刚体/仿射配准。

最小配置示例：

```yaml
_base_: default.yaml

subject:
  id: "mouse_ct_001"
  format: "nifti"  # 或 npz
  segmentation_path: "/path/to/segmentation.nii.gz"
  output_dir: "output/shared_mouse_ct_001"
  target_voxel_size_mm: 0.2
  crop_bbox_mm:
    x: [0.0, 38.0]
    y: [0.0, 40.0]
    z: [0.0, 20.8]
  label_mapping:
    0: 0
    1: 1
    2: 2
  label_roles:
    background_labels: [0]
    allowed_tumor_labels: [1]
    forbidden_tumor_labels: [0, 2]

mesh:
  output_path: "output/shared_mouse_ct_001"
  mesh_file: "output/shared_mouse_ct_001/mouse_ct_001_mesh.npz"

mcx:
  volume_path: "output/shared_mouse_ct_001/mcx_volume_trunk.bin"
  material_path: "output/shared_mouse_ct_001/mcx_material.yaml"
```

主链路必须将 `--config <新配置>` 传给 step0b/0f/0g/02/run_all。每个 subject 的 `output_dir` 会生成独立 `frame_manifest.json`，后续 DE/MCX 都应从该 manifest 读取几何。

### 2.7 运行单个脚本（调试用）

```bash
cd /home/foods/pro/FMT-SimGen
uv run python -c "
import sys
sys.path.insert(0, '.')
# 在此添加调试代码
"
```

---

## 3. 代码风格指南

### 3.1 导入规范

```python
# 标准库放最前面
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 第三方库
import numpy as np
import nibabel as nib
from scipy import ndimage

# 本地导入（相对）
from fmt_simgen.atlas.digimouse import DigimouseAtlas
```

### 3.2 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 模块 | snake_case | `digimouse.py`, `fem_solver.py` |
| 类 | PascalCase | `DigimouseAtlas`, `FEMSolver` |
| 函数/方法 | snake_case | `merge_tissues()`, `assemble_system_matrix()` |
| 变量 | snake_case | `tumor_region`, `depth_min_vox` |
| 常量 | UPPER_SNAKE | `DEFAULT_TAG_MAPPING`, `MAX_ITERATIONS` |
| 类型别名 | PascalCase | `LabelStats`, `AtlasInfo` |

### 3.3 类型注解

**必须**为所有函数参数和返回值添加类型注解:

```python
# 正确
def get_subcutaneous_region(
    self,
    depth_range_mm: Tuple[float, float] = (1.0, 3.0),
    regions: Optional[List[str]] = None
) -> np.ndarray:
    ...

# 错误 - 缺少类型注解
def get_subcutaneous_region(self, depth_range_mm, regions):
    ...
```

### 3.4 Docstring 格式

使用 Google 风格的 docstring:

```python
def assemble_system_matrix(self) -> FEMMatrices:
    """Assemble the FEM system matrix M = K + C + B.

    K (diffusion stiffness):
        K_ij = sum_e D_e * G_e(i,j) / (6 * D0)

    C (absorption mass):
        C_ij = sum_e D0 * mu_a_e * (1/120 * ones(4,4) + diag(1/60))

    Parameters
    ----------
    None

    Returns
    -------
    FEMMatrices
        Named tuple containing M, K, C, B matrices and surface_index.
    """
```

### 3.5 日志记录

使用模块级 logger 并遵循以下级别:

```python
logger = logging.getLogger(__name__)

logger.debug("Detailed debug info for troubleshooting")
logger.info("Normal运行时信息，如 "Loading atlas from: %s", path)
logger.warning("警告信息，如 "Label %d not found, using default", label)
logger.error("错误信息，如 "Failed to load file: %s", str(e))
```

### 3.6 错误处理

```python
# 正确 - 使用具体异常类型
if not self.path.exists():
    raise FileNotFoundError(
        f"Atlas file not found: {self.path}\n"
        f"Please check the path or download the Digimouse atlas."
    )

if self._volume is None:
    raise RuntimeError("Atlas not loaded. Call load() first.")

# 错误 - 过于宽泛
try:
    ...
except Exception:
    pass
```

### 3.7 数据类（Dataclass）

用于简单数据容器:

```python
@dataclass
class LabelStats:
    """Statistics for a single tissue label."""
    label: int
    name: str
    voxel_count: int
    centroid: Tuple[float, float, float]
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
```

### 3.8 NumPy 数组维度

- 3D 体积: `[X, Y, Z]` 或 `[Nx, Ny,Nz]` - 明确注明
- 节点坐标: `[N, 3]` 其中 3 是 (x, y, z)
- 元素索引: `[M, 4]` 其中 4 是四面体的 4 个顶点

### 3.9 坐标系统说明

Digimouse atlas 坐标系（从 Affine 矩阵分析）:
```
X: Left (-19mm at voxel 0) → Right (+19mm at voxel 379)
Y: Anterior (-50mm at voxel 0) → Posterior (+50mm at voxel 991)
Z: Inferior (-10mm at voxel 0) → Superior (+11mm at voxel 207)
→ Dorsal (back) = +Z, Ventral (belly) = -Z
```

### 3.10 代码块组织

每个 .py 文件应按以下顺序组织:

1. 模块 docstring
2. 导入（标准库 → 第三方 → 本地）
3. 常量定义
4. 类型别名（如果有）
5. 数据类（如果有）
6. 主类/函数实现

---

## 4. 目录结构

```
FMT-SimGen/
├── config/
│   ├── default.yaml              # 全局配置（含 MCX/DE 所有参数）
│   └── mesh_20k.yaml             # 20k mesh 专用配置
├── fmt_simgen/
│   ├── __init__.py               # 包导出
│   ├── subject.py                # SubjectManifest/VolumeSpec 几何契约
│   ├── atlas/
│   │   └── digimouse.py          # Digimouse atlas 加载
│   ├── mesh/
│   │   └── mesh_generator.py     # 四面体网格生成
│   ├── physics/
│   │   ├── optical_params.py     # 光学参数管理
│   │   └── fem_solver.py         # DE FEM 求解器
│   ├── tumor/
│   │   └── tumor_generator.py    # 解析肿瘤生成
│   ├── sampling/
│   │   └── dual_sampler.py        # 双载体 GT 采样
│   ├── dataset/
│   │   └── builder.py            # DE Pipeline 编排
│   ├── mcx_volume.py             # MCX: atlas → trunk volume
│   ├── mcx_source.py             # MCX: Pattern3D  source 定义
│   ├── mcx_config.py             # MCX: JSON 配置生成
│   ├── mcx_runner.py             # MCX: CLI 调用（mcx/mcxcl）
│   ├── mcx_projection.py         # MCX: fluence → 7 角度投影
│   └── view_config.py            # TurntableCamera 相机模型
├── scripts/
│   ├── step0b_generate_mesh_cgalmesh.py  # Mesh 生成（支持 --output-name）
│   ├── step0c_fem_matrix.py      # 系统矩阵（支持 --mesh, --output-dir）
│   ├── step0d_voxel_grid.py      # 体素网格（支持 --mesh, --output-dir）
│   ├── step0e_v2_full_graph_laplacian.py  # 图拉普拉斯（支持 --mesh, --output-dir）
│   ├── step0f_mcx_volume.py      # MCX trunk volume 资源
│   ├── step0g_view_config.py     # 相机模型资源（支持 --mesh, --output-dir）
│   ├── 02_generate_dataset.py    # DE 通道样本生成
│   ├── run_mcx_pipeline.py       # MCX 通道独立入口（支持 --shared-dir）
│   ├── verify_3d_de_mcx_alignment.py  # 3D 对齐验证（支持 --mesh）
│   ├── de_surface_to_mcx_projection.py  # DE→MCX 表面对比（支持 --mesh）
│   └── run_all.py                # DE+MCX 双通道入口
├── output/
│   ├── shared/                   # 默认 mesh 资源（52k 节点）
│   │   ├── digimouse_trunk_mesh.npz
│   │   ├── system_matrix.*.npz
│   │   ├── visible_mask.npy
│   │   ├── view_config.json
│   │   ├── mcx_volume_trunk.bin
│   │   └── mcx_material.yaml
│   ├── shared_mesh_20k/          # 20k mesh 资源（独立目录）
│   │   ├── digimouse_trunk_mesh_20k.npz
│   │   ├── system_matrix.*.npz
│   │   ├── visible_mask.npy
│   │   ├── view_config.json
│   │   ├── mcx_volume_trunk.bin
│   │   └── mcx_material.yaml
│   └── visualizations/
│       └── verification_20k/     # 20k mesh 验证可视化
│           ├── sample_0000_3d_de_mcx_alignment.png
│           ├── sample_0000_3d_de_mcx_alignment_slices.png
│           └── sample_0000_de_mcx_surface.png
└── data/
    └── mesh_20k_test/            # 20k mesh 测试数据集
        └── samples/
            └── sample_0000/
                ├── measurement_b.npy
                ├── gt_nodes.npy
                ├── gt_voxels.npy
                ├── tumor_params.json
                ├── sample_0000.json
                ├── source-sample_0000.bin
                ├── sample_0000.jnii
                └── proj.npz
```

---

## 5. 常见任务检查清单

实现新模块时，确保:

- [ ] 函数/方法有完整的类型注解
- [ ] 有 Google 风格的 docstring
- [ ] 使用 `logging` 而非 `print` 进行运行时信息输出
- [ ] 错误使用具体异常类型并提供清晰信息
- [ ] 所有几何参数从 `SubjectManifest` / `frame_manifest.json` / config 读取，不硬编码 shape、voxel size、volume center
- [ ] 新增的函数已在 `__init__.py` 中导出
- [ ] Python 语法检查通过: `python3 -m py_compile newfile.py`

---

## 6. 注意事项

### 6.1 Python 环境

使用 `uv run python` 或激活 `.venv`：

```bash
# 推荐
uv run python scripts/xxx.py

# 或激活 venv
source .venv/bin/activate
python scripts/xxx.py
```

**不要**使用系统 python 或 miniforge3 python。

### 6.2 浮点数除法注意

```python
# 错误 - int() 截断导致不准确
depth_min_vox = int(1.0 / 0.1)  # = 9 而非 10

# 正确
depth_min_vox = int(round(1.0 / 0.1))  # = 10
```

### 6.3 体积数据顺序

Digimouse volume 存储为 `[X, Y, Z]`，其中:
- X = Left-Right (380 voxels, 0=Left)
- Y = Anterior-Posterior (992 voxels, 0=Anterior)
- Z = Inferior-Superior (208 voxels, 0=Inferior)

### 6.4 MCX/JNII 坐标系统

MCX `.bin` 与 JNII 数据按 **ZYX 顺序**存储；项目内部大多数函数使用 **XYZ 顺序**。shape 必须从 `SubjectManifest.shape_zyx` / `SubjectManifest.shape_xyz` 或 `frame_manifest.json` 读取，不要写死 `(104, 200, 190)`。

```python
subject = load_subject_manifest(config, shared_dir)
fluence_xyz = nifti.reshape(subject.shape_zyx).transpose(2, 1, 0)
```

物理原点是 subject 的 trunk-local bbox 角点；atlas/原始 CT 到 trunk-local 的偏移写在 `subject.atlas_to_world_offset_mm` 或 `frame_manifest.json`。

### 6.5 镜像数据位置

Atlas 文件位于:
`/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr`

---

## 7. 数据依赖关系

### 7.1 多 Mesh 配置架构

**设计原则**：每个 mesh 配置有独立的输出目录，避免预计算结果互相冲突。

```
output/
├── shared/                    # 默认 52k mesh
│   ├── digimouse_trunk_mesh.npz
│   ├── system_matrix.*.npz
│   ├── visible_mask.npy
│   ├── view_config.json
│   ├── mcx_volume_trunk.bin
│   └── mcx_material.yaml
│
└── shared_mesh_20k/           # 20k mesh（独立目录）
    ├── digimouse_trunk_mesh_20k.npz
    ├── system_matrix.*.npz
    ├── visible_mask.npy
    ├── view_config.json
    ├── mcx_volume_trunk.bin
    └── mcx_material.yaml
```

**配置文件示例** (`config/mesh_20k.yaml`):
```yaml
_base_: default.yaml

mesh:
  mesh_file: "output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz"
  output_path: "output/shared_mesh_20k/"

dataset:
  num_samples: 3
  experiment_name: "mesh_20k_test"
```

### 7.2 mesh 更新后的级联影响

**当 mesh 文件更新后，必须重新运行以下脚本：**

```bash
# 必需步骤
uv run python scripts/step0c_fem_matrix.py --mesh <mesh_path> --output-dir <output_dir>
uv run python scripts/step0g_view_config.py --mesh <mesh_path> --output-dir <output_dir>

# 可选步骤
uv run python scripts/step0d_voxel_grid.py --mesh <mesh_path> --output-dir <output_dir>
uv run python scripts/step0e_v2_full_graph_laplacian.py --mesh <mesh_path> --output-dir <output_dir>
```

**原因**：
- `step0c` 生成的 M, K, C, B, F 矩阵依赖于 mesh 节点和元素
- `step0g` 生成的 `visible_mask.npy` 依赖于 mesh 表面节点
- 不重新运行会导致节点数不匹配、索引错误

### 7.3 系统矩阵文件说明

```
output/shared/
├── system_matrix.M.npz     # 系统矩阵 M = K + C + B（必需）
├── system_matrix.K.npz     # 扩散刚度矩阵（必需）
├── system_matrix.C.npz     # 吸收质量矩阵（必需）
├── system_matrix.B.npz     # 边界条件矩阵（必需）
├── system_matrix.F.npz     # 源项矩阵（必需）
├── system_matrix.index.npz # 节点索引和表面索引（必需）
└── system_matrix.A.npz     # Forward 矩阵（可选，通常不使用）
```

**重要**：
- **A.npz 不是必需的**，DE 流程使用 M, F + LU 分解实时计算前向测量
- `FEMSolver.forward()` 方法不依赖 A.npz

### 7.4 MCX Volume 存储顺序

**关键**：`mcx_volume_trunk.bin` 以 **ZYX 顺序**存储，但大多数函数期望 **XYZ 顺序**。

加载时必须 transpose：

```python
# 正确加载方式
from fmt_simgen.subject import load_subject_manifest

subject = load_subject_manifest(config, shared_dir)
vol = np.fromfile(shared_dir / "mcx_volume_trunk.bin", dtype=np.uint8)
volume_zyx = vol.reshape(subject.shape_zyx)
volume_xyz = volume_zyx.transpose(2, 1, 0)
```

**已修复的文件**：
- `scripts/step0g_view_config.py` - `load_mcx_volume()` 已正确处理
- `scripts/visualize_visibility_3d.py` - 已正确处理
- `fmt_simgen/dataset/builder.py` - 已正确处理

---

## 8. 坐标系统规范

### 8.1 Trunk-Local 坐标系

所有数据统一使用 **trunk-local** 坐标系（原点在当前 subject 的 volume bbox 角点）。范围不再是全局固定值，必须由 `SubjectManifest.volume_extents_mm` 派生。

| 属性 | 来源 |
|------|------|
| shape XYZ | `SubjectManifest.shape_xyz` |
| shape ZYX | `SubjectManifest.shape_zyx` |
| voxel size | `SubjectManifest.voxel_size_mm` |
| 物理范围 | `SubjectManifest.volume_extents_mm` |
| 投影旋转中心 | `SubjectManifest.volume_center_world_mm` |
| 肿瘤组织约束 | `SubjectManifest.label_roles` |

### 8.2 坐标转换

**Atlas → Trunk-Local**：
```python
subject = load_subject_manifest(config, shared_dir)
trunk_local_coords = atlas_coords - subject.atlas_to_world_offset_mm
```

**Volume center**：
```python
volume_center_world = subject.volume_center_world_mm
```

**用途**：投影时将坐标平移到以旋转中心为原点，然后绕 Y 轴旋转。

### 8.3 投影坐标流程

```
节点坐标 [N, 3] (trunk-local, [0, extent])
    ↓ 减去 subject.volume_center_world_mm
节点坐标 (centered, [-extent/2, +extent/2])
    ↓ 绕 Y 轴旋转
旋转后坐标
    ↓ 正交投影
探测器像素 (u, v) + 深度 depth
```

### 8.4 禁止新增的硬编码

主流程、库代码和新脚本中禁止新增以下几类硬编码：
- `190, 200, 104` / `104, 200, 190`
- `0.2` 作为默认体素大小
- `[19.0, 20.0, 10.4]` 或 `[38.0, 40.0, 20.8]`
- `0=background, 1=soft_tissue, 2=bone` 作为不可配置规则
- Digimouse crop `[340, 740]` 作为通用裁剪规则

如需 legacy Digimouse 默认值，必须通过 `subject_manifest_from_config(config)` 或 `load_subject_manifest(config, shared_dir)` 获取。
