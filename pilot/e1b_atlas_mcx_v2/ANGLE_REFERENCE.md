# 角度定义快速参考

## 坐标系统

```
         Z (dorsal/背部)
         ↑
         |
         |   Y (anterior→posterior)
         |  ↗
         | /
         |/
         +----------→ X (left→right)
        /
       /
      ↙
   (ventral/腹部)
```

- **X**: 左(-) → 右(+)，范围 ~[-12.3, +10.9] mm
- **Y**: 前 → 后，范围 ~[-50, +50] mm（ torso 中部切片）
- **Z**: 腹(-) → 背(+)，范围 ~[-8.1, +10.1] mm

---

## 视角定义

### 0° - 背面视角 (Dorsal View)
```
      相机
        ↓
    [老鼠背部]  ← 直接看 dorsal 表面
        ↑
      (腹部)
```
- **适用**: P1-dorsal (背部中心源)
- **坐标**: 源 Z ≈ +6 mm (dorsal 表面下 4mm)

---

### +90° - 左侧面视角 (Left Side View)
```
相机 → [老鼠左侧]  ← 从 -X 方向看
             ↑
         (右侧)
```
- **适用**: P2-left (左侧源)
- **坐标**: 源 X ≈ -8.3 mm (left 表面右侧 4mm)
- **注意**: +90° 意味着相机在 -X 轴上，看向 +X 方向

---

### -90° - 右侧面视角 (Right Side View)
```
         (左侧)
            ↑
相机 → [老鼠右侧]  ← 从 +X 方向看
```
- **适用**: P3-right (右侧源)
- **坐标**: 源 X ≈ +6.9 mm (right 表面左侧 4mm)
- **注意**: -90° 意味着相机在 +X 轴上，看向 -X 方向

---

### -30° - 背左侧视角 (Dorsal-Left View)
```
      相机
        \
         \  指向 dorsal-left
          \
    [老鼠背部]
        ↑
      (腹部)
```
- **适用**: P4-dorsal-lateral
- **坐标**: X ≈ -6.5 mm, Z ≈ +6.1 mm
- **特点**: 同时看到 dorsal 和左侧表面

---

### 60° - 斜腹视角 (Oblique Ventral)
```
      (背部)
         ↑
    [老鼠腹部]
          \
           \  相机从斜上方看
            ↘
```
- **适用**: P5-ventral (腹侧源)
- **注意**: ventral 表面凹陷，信号较弱

---

## 旋转矩阵验证

```python
import numpy as np

def rotation_matrix_y(angle_deg):
    """右手定则 Y 轴旋转矩阵."""
    θ = np.deg2rad(angle_deg)
    return np.array([
        [np.cos(θ),  0, np.sin(θ)],
        [0,          1,         0],
        [-np.sin(θ), 0, np.cos(θ)]
    ])

# 验证: +90° 应该将 +Z 旋转到 +X
R_90 = rotation_matrix_y(90)
print(R_90 @ [0, 0, 1])  # → [1, 0, 0] ✓

# 验证: -90° 应该将 +Z 旋转到 -X  
R_m90 = rotation_matrix_y(-90)
print(R_m90 @ [0, 0, 1])  # → [-1, 0, 0] ✓
```

---

## 快速对照表

| 源位置 | 最佳角度 | 相机位置 | 观察方向 |
|--------|---------|---------|---------|
| dorsal (Z=+6) | 0° | Z=+∞ | -Z (向腹部) |
| left (X=-8) | +90° | X=-∞ | +X (向右侧) |
| right (X=+7) | -90° | X=+∞ | -X (向左侧) |
| dorsal-left | -30° | 斜上方 | 向腹右下 |
| ventral (Z=-4) | 60° | 斜上方 | 向腹左下 |
