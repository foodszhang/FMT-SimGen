# E1c Coordinate Contract

## 1. World Coordinate System

使用 centered world coordinates：

```
x ∈ [-15, 15] mm  (Left → Right)
y ∈ [-15, 15] mm  (Anterior → Posterior)
z ∈ [-10, 10] mm  (Ventral → Dorsal)
```

约定：
- `+X` = right
- `+Y` = posterior / tail
- `+Z` = dorsal / top
- **Dorsal top surface** = `z = +10 mm`
- Ventral bottom surface = `z = -10 mm`

## 2. Config → World Mapping

配置文件中 source 定义为：
```yaml
source_center: [x_mm, y_mm, depth_from_dorsal_mm]
```

第三项 `depth_from_dorsal_mm` 不是 world z，而是从 dorsal 表面到源点的深度。

转换关系：
```python
Lz = 20.0  # Total z extent
z_world = Lz / 2 - depth_from_dorsal_mm
z_world = 10.0 - depth_from_dorsal_mm
```

示例：
| config | x_mm | y_mm | depth | x_world | y_world | z_world |
|--------|------|------|-------|---------|---------|---------|
| M01    | 0    | 0    | 2     | 0       | 0       | 8       |
| M02    | 0    | 0    | 4     | 0       | 0       | 6       |
| M03    | 3    | 3    | 2     | 3       | 3       | 8       |

## 3. World → MCX Mapping

MCX box 坐标（单位 mm）：
```python
x_mcx = x_world + 15.0
y_mcx = y_world + 15.0
z_mcx = depth_from_dorsal_mm
```

MCX volume 尺寸：`[Nx, Ny, Nz]`
- X: 300 voxels → `x_mcx ∈ [0, 30] mm`
- Y: 300 voxels → `y_mcx ∈ [0, 30] mm`
- Z: 200 voxels → `z_mcx ∈ [0, 20] mm`

示例：
| config | x_world | y_world | z_world | depth | x_mcx | y_mcx | z_mcx |
|--------|---------|---------|---------|-------|-------|-------|-------|
| M01    | 0       | 0       | 8       | 2     | 15    | 15    | 2     |
| M02    | 0       | 0       | 6       | 4     | 15    | 15    | 4     |
| M03    | 3       | 3       | 8       | 2     | 18    | 18    | 2     |

## 4. Surface Plane

本任务唯一 surface plane：
```python
z_surface_world = +10.0  # mm
```

**MCX z 轴约定**：
- MCX z=0 是组织**表面**（空气/组织界面）
- MCX z 增加是向组织**内部**
- 源在 z_mcx = depth_from_dorsal_mm（例如 2mm 深度 → z_mcx=2mm）

MCX volume 中：
- Top surface (dorsal) = `z_mcx = 0`（volume 的第一层）
- Bottom (ventral) = `z_mcx = Lz`（volume 的最后一层）

**提取 surface fluence**：
```python
surface_fluence = fluence[:, :, 0]  # z_idx = 0, not z_idx = -1!
```

## 5. Surface Image Pixel Mapping

Surface plane 上的 2D 坐标：
```python
u_mm = x_world  # surface plane 上的 x 坐标
v_mm = y_world  # surface plane 上的 y 坐标
```

像素映射（centered origin）：
```python
# 假设 image_size x image_size 的正方形图像
# pixel_size_mm = fov_mm / image_size
# 图像中心对应 (u_mm, v_mm) = (0, 0)

col = (u_mm + fov_mm / 2) / pixel_size_mm
row = (v_mm + fov_mm / 2) / pixel_size_mm
```

逆映射：
```python
u_mm = (col - image_size / 2) * pixel_size_mm
v_mm = (row - image_size / 2) * pixel_size_mm
```

## 6. Array Index Convention

图像数组索引：
- `image[row, col]` 或 `image[y_idx, x_idx]`
- `col` 对应 x 方向（world X）
- `row` 对应 y 方向（world Y）

**重要**：
- `col` 增加 → `x_world` 增加 → `u_mm` 增加
- `row` 增加 → `y_world` 增加 → `v_mm` 增加
- 源点 `(x_world, y_world) = (3, 3)` 的峰值应出现在图像右下方

## 7. MCX Volume Grid

MCX fluence volume 形状：`[Nx, Ny, Nz]` = `[300, 300, 200]`
- `fluence[i, j, k]` 对应：
  - `x_mcx = i * voxel_size_mm`
  - `y_mcx = j * voxel_size_mm`
  - `z_mcx = k * voxel_size_mm`

Top surface 提取：
```python
# Method 1: 使用最顶层
top_surface_fluence = fluence[:, :, -1]  # shape [Nx, Ny]

# Method 2: 如果有边界层厚度，使用 extrapolated boundary
# top_z_idx = int(z_surface_mcx / voxel_size_mm)
```

转换到 surface image：
```python
# MCX 输出是 [Nx, Ny] 对应 [x_mcx, y_mcx]
# 需要映射到 world coordinates
# x_world = x_mcx - 15.0
# y_world = y_mcx - 15.0
# 然后按 section 5 的 pixel mapping 构建 surface image
```

## 8. Kernel Evaluation Grid

对于解析 kernel（Gaussian, Green infinite, Green half-space）：

```python
# 在 surface plane (z = +10 mm) 上采样
x_grid_mm = (np.arange(image_size) - image_size / 2 + 0.5) * pixel_size_mm
y_grid_mm = (np.arange(image_size) - image_size / 2 + 0.5) * pixel_size_mm

# meshgrid
X_mm, Y_mm = np.meshgrid(x_grid_mm, y_grid_mm)  # shape [image_size, image_size]

# X_mm[row, col] = x coordinate at pixel (row, col)
# Y_mm[row, col] = y coordinate at pixel (row, col)
```

## 9. Sanity Check Example

配置 M01：`source_center = [0, 0, 2]`

| 属性 | 值 |
|------|-----|
| x_world | 0 mm |
| y_world | 0 mm |
| z_world | 8 mm |
| x_mcx | 15 mm |
| y_mcx | 15 mm |
| z_mcx | 2 mm |
| 预期峰值位置 (u_mm, v_mm) | (0, 0) |
| 预期峰值位置 (col, row) | (image_size/2, image_size/2) |

配置 M03：`source_center = [3, 3, 2]`

| 属性 | 值 |
|------|-----|
| x_world | 3 mm |
| y_world | 3 mm |
| z_world | 8 mm |
| x_mcx | 18 mm |
| y_mcx | 18 mm |
| z_mcx | 2 mm |
| 预期峰值位置 (u_mm, v_mm) | (3, 3) |
| 预期峰值位置 (col, row) | (image_size/2 + 3/pixel_size_mm, image_size/2 + 3/pixel_size_mm) |
| 峰值方向 | 图像右下方 |

## 10. Rotation Convention (NOT USED in E1c)

本任务 E1c 不涉及旋转，以下仅为记录：

在 E1b multi-view 设置中：
- `angle` 为转台旋转角度
- `+angle` 表示物体绕 Y 轴正向旋转
- 投影使用 `R_y(+angle) @ source`
