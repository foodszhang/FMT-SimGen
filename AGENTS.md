# FMT-SimGen 代码库代理规范

本文件为在 FMT-SimGen 代码库中工作的 AI 代理提供规范和指南。

---

## 1. 项目概述

**FMT-SimGen** 是一个用于 FMT（荧光分子断层成像）模拟数据集生成的 Python 库。
它从解析肿瘤函数生成合成数据，同时产生 FEM 节点级和体素级两组 ground truth。

**项目路径**: `/home/foods/pro/FMT-SimGen`

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
    AnalyticFocus, DualSampler, DatasetBuilder, TurntableCamera
)
print('All imports successful!')
"
```

### 2.4 脚本执行

使用 `uv run python`，**不要**用系统 python 或 miniforge3 python：

```bash
# Step 0a: 加载并分析 Digimouse atlas
uv run python scripts/run_step0a_atlas.py

# Step 0b-0e: 生成共享资产
uv run python scripts/step0b_generate_mesh.py
uv run python scripts/step0c_fem_matrix.py
uv run python scripts/step0d_voxel_grid.py
uv run python scripts/step0e_v2_full_graph_laplacian.py

# Step 0f: MCX 体积资源
uv run python scripts/step0f_mcx_volume.py
uv run python scripts/step0g_view_config.py

# Step 1-4: DE 通道数据集生成
uv run python scripts/02_generate_dataset.py -n 50

# Step 2m-4m: MCX 通道（独立运行）
uv run python scripts/step2m_generate_mcx_sources.py
uv run python scripts/step3m_mcx_simulate.py --samples_dir data/gaussian_1000/samples
uv run python scripts/step4m_mcx_projection.py --samples_dir data/gaussian_1000/samples

# 双通道（推荐）
uv run python scripts/run_all.py -n 50 --enable_mcx
```

### 2.5 运行单个脚本（调试用）

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
│   └── default.yaml              # 全局配置（含 MCX/DE 所有参数）
├── fmt_simgen/
│   ├── __init__.py               # 包导出
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
│   ├── 01_generate_mesh.py       # 网格 + 系统矩阵
│   ├── 02_generate_dataset.py    # DE 通道样本生成
│   ├── 03_verify_dataset.py      # 数据验证
│   ├── run_step0a_atlas.py       # Atlas 分析
│   ├── step0f_mcx_volume.py      # MCX trunk volume 资源
│   ├── step0g_view_config.py     # MCX 相机模型资源
│   ├── step2m_generate_mcx_sources.py  # MCX 源配置
│   ├── step3m_mcx_simulate.py    # MCX GPU 仿真
│   ├── step4m_mcx_projection.py  # MCX 投影生成
│   ├── run_all.py                # DE+MCX 双通道入口
│   └── run_mcx_pipeline.py      # MCX 通道独立入口
└── output/shared/                # 共享资源（.gitignore）
    ├── mcx_volume_trunk.bin      # MCX trunk volume
    ├── mcx_material.yaml         # MCX 材料参数
    └── view_config.json          # TurntableCamera 配置
```

---

## 5. 常见任务检查清单

实现新模块时，确保:

- [ ] 函数/方法有完整的类型注解
- [ ] 有 Google 风格的 docstring
- [ ] 使用 `logging` 而非 `print` 进行运行时信息输出
- [ ] 错误使用具体异常类型并提供清晰信息
- [ ] 所有配置从 `config/default.yaml` 读取，不硬编码参数
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

### 6.4 MCX JNII 坐标系统

JNII 格式存储为 ZYX 顺序 `(104, 200, 190)`。加载后需 transpose:
```python
fluence_xyz = nifti.transpose(2, 1, 0)  # → (190, 200, 104) = [X, Y, Z]
```
MCX trunk volume 物理原点对应躯干偏移 `trunk_offset_mm = [0, 30, 0]`。

### 6.5 镜像数据位置

Atlas 文件位于:
`/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr`
