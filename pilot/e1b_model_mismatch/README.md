# E1b: Model Mismatch Validation

量化 MCX GT 与解析 PSF 前向模型的 mismatch 对优化精度的影响。

## 背景

- **E1 结论**：inverse crime 条件下优化完美收敛（误差 ~0），pipeline 正确
- **E1b 问题**：当 GT（MCX）和前向模型（解析 PSF）存在 mismatch 时，优化精度退化多少？
- **关键决策**：E2 阶段是否需要残差网络补偿

## 目录结构

```
e1b_model_mismatch/
├── config.yaml              # 实验配置
├── generate_mcx_gt.py       # MCX 多视角 GT 生成
├── run_e1b.py               # 主脚本：优化 + 对比
├── README.md                # 本文件
└── results/
    ├── gt_data/             # MCX GT 数据
    ├── optimization/        # 优化结果
    └── summary.json         # 汇总
```

## 实验矩阵

| ID | 组织 | 深度 | 位置 | GT 来源 | 前向模型 |
|----|------|------|------|---------|----------|
| M01 | Muscle | 2mm | 中心 | MCX 点源 | 解析 PSF |
| M02 | Muscle | 4mm | 中心 | MCX 点源 | 解析 PSF |
| M03 | Muscle | 2mm | (3,3)mm | MCX 点源 | 解析 PSF |
| M04 | Liver | 2mm | 中心 | MCX 点源 | 解析 PSF |
| M05 | Liver | 4mm | 中心 | MCX 点源 | 解析 PSF |

## 关键区别

### vs E1
- **E1**: GT 和前向模型完全一致（inverse crime）→ 误差 ~0
- **E1b**: GT 来自 MCX，前向模型用解析 PSF → 存在物理 mismatch

### 点源 vs Gaussian 源
- MCX 模拟 isotropic 点源（δ 函数）
- 优化的 Gaussian 有 σ 参数
- 预期：优化后 σ 应趋近小值（~0.1-0.3mm），补偿体素级展宽

## 运行方法

```bash
cd /home/foods/pro/FMT-SimGen

# 方式 1：完整运行（MCX GT 生成 + 优化）
uv run python pilot/e1b_model_mismatch/run_e1b.py

# 方式 2：分步运行
# Step 1: 生成 MCX GT（5 配置 × 7 视角 = 35 次 MCX）
uv run python pilot/e1b_model_mismatch/generate_mcx_gt.py

# Step 2: 运行优化（复用已有 GT）
uv run python pilot/e1b_model_mismatch/run_e1b.py
```

## 判定标准

| 判定 | 条件 | 后续动作 |
|------|------|----------|
| ✅ GO | 所有配置位置误差 < 0.5mm | 进 E2，不需要残差网络 |
| ⚠️ CAUTION | 位置误差 0.5-1.0mm | 进 E2，并行设计残差网络 |
| ❌ NOGO | 位置误差 > 1.0mm | 必须先加残差网络 |

## 预期

E0 验证了 NCC > 0.997（匀质），mismatch 很小。预期位置误差 < 0.3mm，判定 GO。

## 技术细节

### 多视角提取方案

采用**精确方案**：对每个视角旋转源位置，跑 7 次 MCX。

```
对于角度 θ：
1. 将源位置绕 y 轴旋转 -θ
2. 构建匀质体积（z=0 为表面）
3. MCX 仿真 → 提取 z=0 表面
4. 裁剪到 256×256
```

这样每个视角都有正确的 tissue-air 边界条件。

### 图像归一化

MCX fluence 绝对值与解析 PSF 不同，对比前归一化到峰值=1。

## 复用代码

- `e1_single_source/psf_splatting.py` → 前向模型
- `e1_single_source/optimize.py` → 优化逻辑
- `e0_psf_validation/mcx_point_source.py` → MCX 仿真

## 预期耗时

| 步骤 | 耗时 |
|------|------|
| MCX GT 生成（35 次仿真） | 2-3 小时 |
| 优化（5 配置） | 5 分钟 |
| 总计 | ~3 小时 |
