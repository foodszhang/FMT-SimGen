# MCX vs Green's Function 多位置投影验证总结

## 1. 核心结论

### 1.1 投影一致性验证
表面感知 Green 函数投影与 MCX Monte Carlo 仿真在 **5 个不同解剖位置** 均表现出高度一致性：

| 位置 | 最佳角度 | NCC | 信号强度 | 验证状态 |
|------|---------|-----|----------|----------|
| P1-dorsal (背部中心) | 0° | **0.998** | 5.6×10⁵ | ✅ 优秀 |
| P2-left (左侧) | 90° | **0.993** | 3.5×10⁵ | ✅ 优秀 |
| P3-right (右侧) | -90° | **0.993** | 5.6×10⁵ | ✅ 优秀 |
| P4-dorsal-lateral (背侧偏左) | -30° | **0.993** | 2.0×10⁶ | ✅ 优秀 |
| P5-ventral (腹部) | 60° | **0.963** | 7.1×10¹ | ⚠️ 信号弱 |

**结论**: 非 ventral 位置 NCC 均达到 **0.993+**，验证 Green 函数在复杂表面几何下的有效性。

---

## 2. 投影机制对比

### 2.1 MCX 投影流程
```
3D 体素级光通量 (JNII格式)
    ↓
Y轴旋转 (视角变换)
    ↓
Z-buffer 深度投影 (取最近表面)
    ↓
体素覆盖填充 (避免采样间隙)
    ↓
2D 投影图像
```

### 2.2 Green 函数投影流程
```
二值化 atlas 体积
    ↓
Y轴旋转获取可见表面坐标
    ↓
对每个像素计算到源的距离 r
    ↓
G(r) = exp(-μeff·r) / (4π·D·r)
    ↓
2D Green 响应图像
```

### 2.3 关键差异
| 特性 | MCX | Green |
|------|-----|-------|
| 计算方式 | Monte Carlo 随机行走 | 解析解直接计算 |
| 计算时间 | ~30s (百万光子) | <1s |
| 边界处理 | 自然模拟 | 需表面提取 |
| 噪声 | 有 (光子统计) | 无 (解析) |

---

## 3. 关键注意事项

### 3.1 坐标系统 (极易出错!)

**JNII 数据顺序**: MCX 输出的 JNII 文件已经是 **XYZ 顺序**，无需 transpose！
```python
# ❌ 错误 (旧代码)
fluence = data["NIFTIData"].transpose(2, 1, 0)

# ✅ 正确
fluence = data["NIFTIData"]  # 已是 (X, Y, Z)
```

**体积 reshape**: atlas bin 文件是 ZYX 顺序
```python
volume = np.fromfile(atlas_bin, dtype=np.uint8).reshape((104, 200, 190))  # Z,Y,X
volume_xyz = volume.transpose(2, 1, 0)  # → X,Y,Z
```

### 3.2 旋转角度定义

**右手定则围绕 Y 轴**:
- **+90°**: 相机从 **-X 方向** 看 (看向右侧，左侧面可见)
- **-90°**: 相机从 **+X 方向** 看 (看向左侧，右侧面可见)
- **0°**: 相机从 **+Z 方向** 看 ( dorsal/背面)

```python
# 旋转矩阵 (右手定则)
R_y(θ) = [[ cos(θ),  0, sin(θ)],
          [      0,  1,      0],
          [-sin(θ),  0, cos(θ)]]

# 应用: coords_rotated = coords_original @ R.T
```

### 3.3 采样间隙问题

**当 voxel_size (0.2mm) > pixel_size (~0.195mm) 时**，单点采样会出现网格线：

```python
# ❌ 单点采样 (会出现间隙)
px = int((cam_x + half_w) / px_size)
projection[py, px] = value

# ✅ 体素覆盖填充 (推荐)
half_voxel = voxel_size_mm / 2
u_start = int((px - half_voxel + half_w) / px_size_x)
u_end   = int((px + half_voxel + half_w) / px_size_x)
# 填充 u_start 到 u_end 范围内的所有像素
```

### 3.4 归一化策略

**分别归一化**用于可视化对比：
```python
mcx_norm = mcx_proj / mcx_proj.max()    # [0, 1]
green_norm = green_proj / green_proj.max()  # [0, 1]
```

**注意**: 不要对 MCX 和 Green 使用统一 colorbar，因为它们的物理单位不同（光子数 vs 漫射通量）。

### 3.5 深度计算

相机坐标系下深度计算：
```python
# 旋转后坐标: [x, y, z] 其中 +z 指向相机
depth = camera_distance_mm - rotated_z
# depth > 0: 在相机前方 (可见)
# depth 越小: 越靠近相机 (表面)
```

---

## 4. 误差来源分析

### 4.1 P5-ventral 低 NCC (0.963) 原因
- **信号极弱**: max=70 vs 其他位置 10⁵~10⁶ 量级
- **原因**: ventral 表面凹陷，60° 视角下光传播路径长，衰减严重
- **建议**: ventral 位置需要专用处理（如使用 180° 视角或增加光子数）

### 4.2 残差分布特征
从可视化观察：
- **中心区域**: 吻合度极高 (NCC > 0.99)
- **边缘区域**: 轻微差异，可能来自表面提取精度
- **P4-lateral**: 信号最强 (2×10⁶)，Green 函数近似最准确

---

## 5. 最佳实践清单

- [ ] **数据加载**: JNII 无需 transpose，atlas bin 需要 reshape + transpose
- [ ] **角度选择**: 根据源位置选择最佳视角 (+90° for left, -90° for right)
- [ ] **采样方式**: 始终使用 voxel coverage filling
- [ ] **归一化**: 可视化用分别归一化，指标计算用统一归一化
- [ ] **验证指标**: NCC > 0.99 视为优秀，< 0.95 需检查角度/坐标
- [ ] **轮廓线**: 每个角度必须使用对应角度的 `valid_mask` 绘制轮廓

---

## 6. 扩展建议

1. **多角度平均**: 对 ventral 位置可采用多角度平均提升信噪比
2. **半无限介质 Green**: 当前使用无限介质 Green，可测试半无限介质修正
3. **各向异性**: 当前假设各向同性散射 (g=0.9)，可研究各向异性影响
4. **非均匀组织**: 当前使用均匀光学参数，可扩展到多组织类型

---

## 7. 代码参考

核心实现文件：
- `surface_projection.py`: Green 函数投影核心
- `run_multiposition_bestview.py`: 多位置测试框架
- `plot_multiposition_results.py`: 可视化脚本

关键函数：
```python
# 获取表面坐标
surface_coords, valid_mask = project_get_surface_coords(
    volume_mask, angle_deg, camera_dist, fov, resolution, voxel_size
)

# 渲染 Green 投影
green_proj = render_green_surface_projection(
    source_pos, atlas_binary, angle_deg, camera_dist,
    fov, resolution, tissue_params, voxel_size
)
```
