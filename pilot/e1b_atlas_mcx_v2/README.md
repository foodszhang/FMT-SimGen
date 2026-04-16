# E1b-v2: 递进式 MCX vs Green 验证实验

## 关键发现（震惊！）

### ⚠️ Stage 1 vs Stage 1.5 对比结果

| 深度 | Stage 1 (匀质方块) | Stage 1.5 (Atlas 形状) | 退化 |
|------|-------------------|----------------------|------|
| 2mm | **0.9988** | **0.8011** | -0.20 |
| 4mm | **0.9984** | **0.5678** | -0.43 |
| 6mm | **0.9949** | **0.3115** | -0.68 |
| 9mm | **0.9618** | **0.0353** | -0.93 |
| 12mm | **0.8032** | **-0.0009** | -0.80 |

### 核心结论

1. **匀质方块的结果过于乐观** — 不代表真实小鼠场景
2. **小鼠表面边界显著影响 PSF** — 即使是匀质组织，边界效应也很强
3. **纯 Green 函数在真实几何中不准确** — 即使是浅层（2-4mm）也严重退化

### 含义

- **Stage 1（方块）**: 证明 Green 函数在理想条件下理论上可行
- **Stage 1.5（Atlas）**: 证明在真实小鼠上 **Green 函数单独不够用**
- **必须引入残差网络或边界修正** 才能在真实 FMT 中达到可用精度

---

## 实验结构

```
Stage 1   : 匀质方块 + 点源 × 5 深度              ✅ 已完成
Stage 1.5 : Atlas 体积 + 点源 × 5 深度              ✅ 已完成（关键发现！）
Stage 2   : Atlas 体积 + uniform source × 多点拟合  ⏸️ 暂停（需重新设计）
Stage 3   : Atlas 体积 + 多角度                     ⏸️ 暂停
```

---

## 坐标系与配置

### Atlas 原始坐标系
- **文件**: `/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr`
- **形状**: (380, 992, 208) XYZ
- **体素**: 0.1mm
- **X**: Left (-19mm) → Right (+19mm)
- **Y**: Anterior (-50mm) → Posterior (+50mm)
- **Z**: Inferior (-10mm) → Superior (+11mm)，**Dorsal = +Z**

### MCX Trunk 体积
- **形状**: (104, 200, 190) ZYX
- **体素**: 0.2mm
- **Offset**: [0, 30, 0] mm

### 源位置
- **XY**: [17, 48] mm（躯干中心）
- **Dorsal Z**: 15.4 mm
- **深度**: 距背部表面 2-12mm

### 投影配置
- **角度**: [-60, -30, 0, 30, 60]（排除 ±90°）
- **类型**: Orthographic
- **FOV**: 50mm
- **分辨率**: 256×256

---

## 可视化文件

| 文件 | 说明 |
|------|------|
| `stage1_ncc_vs_depth_v2.png` | Stage 1 NCC 曲线 |
| `stage1_peak_attenuation.png` | 信号衰减曲线 |
| `stage1_shape_comparison_v2.png` | 2D 投影形状对比 |
| `stage1_vs_stage1_5_comparison.png` | **关键对比图** |

---

## 执行命令

### Stage 1（匀质方块）
```bash
uv run python pilot/e1b_atlas_mcx_v2/run_stage1.py \
  --mcx /mnt/f/win-pro/bin/mcx.exe \
  --depths 2 4 6 9 12
```

### Stage 1.5（Atlas 形状）
```bash
uv run python pilot/e1b_atlas_mcx_v2/run_stage1_atlas.py \
  --mcx /mnt/f/win-pro/bin/mcx.exe \
  --depths 2 4 6 9 12
```

### 生成对比图
```bash
uv run python pilot/e1b_atlas_mcx_v2/plot_stage1_comparison.py
```

---

## 下一步（重新设计）

鉴于 Stage 1.5 的震惊结果，Stage 2/3 需要重新设计：

**选项 A**: 放弃纯 Green 函数，直接训练残差网络
**选项 B**: 引入边界修正（如半空间 Green + 镜像源）
**选项 C**: 使用局部表面近似（如 E1d 的方法）

待讨论后决定。
