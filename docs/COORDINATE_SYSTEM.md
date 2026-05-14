# FMT-SimGen 坐标系规范 v2

## 0. 核心规则

当前主流程是 **Subject Manifest 驱动**。不要把 Digimouse 的固定几何当成全局事实。

权威来源：
- `fmt_simgen.subject.SubjectManifest`
- `fmt_simgen.subject.load_subject_manifest(config, shared_dir)`
- `<shared_dir>/frame_manifest.json`

legacy 兼容：
- 没有 `subject:` 配置时，Digimouse 默认值由 `mcx.volume_shape`、`mcx.voxel_size_mm`、`mcx.trunk_offset_mm` 派生。
- `fmt_simgen/frame_contract.py` 只保留 legacy 兼容常量。新代码不得从这里读取运行时 shape、voxel size、volume center 或 label roles。
- 如果 config 中显式存在 `subject:`，以 `subject:` 为准，不被 shared 目录里的旧 manifest 覆盖。

## 1. 世界坐标系：`mcx_trunk_local_mm`

每个 subject 都有自己的 trunk-local frame：
- 原点：当前 subject volume bbox 的角点，即 voxel `[0,0,0]` 的 corner。
- 单位：毫米。
- 轴向：输入 CT/分割必须在进入主流程前预对齐到项目约定轴向。
- 范围：不是固定值；由 `subject.volume_extents_mm` 决定。

所有以下数据必须使用同一个 subject-local frame：
- `mesh.npz` 的 `nodes`
- `gt_nodes.npy` 对应的隐式坐标
- `gt_voxels.npy` 的 grid offset/spacing/shape
- `tumor_params.json` 中 `foci[i].center`
- MCX source pattern 的 origin，换算到 voxel 后写入 JSON `Pos`
- MCX/DE 投影时使用的旋转中心

## 2. Manifest 字段

典型 `frame_manifest.json`：

```json
{
  "subject_id": "digimouse_legacy",
  "world_frame": "mcx_trunk_local_mm",
  "output_dir": "output/shared",
  "atlas_to_world_offset_mm": [0.0, 34.0, 0.0],
  "mcx_volume": {
    "shape_xyz": [190, 200, 104],
    "shape_zyx": [104, 200, 190],
    "voxel_size_mm": 0.2,
    "origin_world_mm": [0.0, 0.0, 0.0],
    "extent_mm": [38.0, 40.0, 20.8],
    "bbox_world_mm": {
      "min": [0.0, 0.0, 0.0],
      "max": [38.0, 40.0, 20.8]
    }
  },
  "volume_center_world_mm": [19.0, 20.0, 10.4],
  "label_roles": {
    "background_labels": [0],
    "allowed_tumor_labels": [1],
    "forbidden_tumor_labels": [0, 2]
  },
  "voxel_grid_gt": {
    "shape": [190, 200, 104],
    "spacing_mm": 0.2,
    "offset_world_mm": [0.0, 0.0, 0.0],
    "frame": "mcx_trunk_local_mm"
  }
}
```

上面的数值是 legacy Digimouse 示例，不是新代码常量。

## 3. 坐标换算

MCX/GT voxel index 与 world mm：

```python
subject = load_subject_manifest(config, shared_dir)

world_mm = (voxel_xyz + 0.5) * subject.voxel_size_mm
voxel_xyz_float = world_mm / subject.voxel_size_mm
```

Atlas/原始 CT 到 trunk-local：

```python
world_mm = atlas_mm - subject.atlas_to_world_offset_mm
```

投影中心：

```python
centered = world_mm - subject.volume_center_world_mm
```

## 4. 存储顺序

项目内部数组通常使用 XYZ 顺序，MCX `.bin` 和 JNII 使用 ZYX 顺序。

```python
subject = load_subject_manifest(config, shared_dir)

raw = np.fromfile(shared_dir / "mcx_volume_trunk.bin", dtype=np.uint8)
volume_zyx = raw.reshape(subject.shape_zyx)
volume_xyz = volume_zyx.transpose(2, 1, 0)
```

写 source pattern：

```python
# pattern_xyz_oriented 逻辑上覆盖 XYZ bbox，但 MCX pattern3d 按 ZYX 写入
pattern.transpose(2, 1, 0).tofile(source_bin_path)
json_pos = [z0, y0, x0]
```

## 5. 新 CT/分割接入

新 subject 推荐配置：

```yaml
subject:
  id: "mouse_ct_001"
  format: "nifti"
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
```

NIfTI 输入必须已预对齐；本项目当前只消费坐标契约，不自动做配准。

## 6. 禁止新增的硬编码

主流程、库代码和新脚本中不要新增：
- `190, 200, 104` / `104, 200, 190`
- `0.2` 作为体素大小
- `[19.0, 20.0, 10.4]` 或 `[38.0, 40.0, 20.8]`
- `0=background, 1=soft_tissue, 2=bone` 作为不可配置规则
- Digimouse crop `[340, 740]` 作为通用规则

使用：

```python
subject = load_subject_manifest(config, shared_dir)
shape_xyz = subject.shape_xyz
shape_zyx = subject.shape_zyx
voxel_size = subject.voxel_size_mm
center = subject.volume_center_world_mm
roles = subject.label_roles
```

## 7. 验证

基本 smoke：

```bash
uv run python -c "
from fmt_simgen.pipeline.shared import load_config_with_inheritance
from fmt_simgen.subject import load_subject_manifest
cfg = load_config_with_inheritance('config/default.yaml')
m = load_subject_manifest(cfg)
print(m.subject_id, m.shape_xyz, m.shape_zyx, m.voxel_size_mm, m.volume_center_world_mm.tolist())
"
```

mesh 或 volume 更新后，必须重新生成依赖资产：

```bash
uv run python scripts/step0c_fem_matrix.py --mesh <mesh_path> --output-dir <shared_dir>
uv run python scripts/step0g_view_config.py --config <config> --mesh <mesh_path> --output-dir <shared_dir>
```
