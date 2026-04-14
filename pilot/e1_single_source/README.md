# E1: Single-Source 3DGS Optimization Validation

验证单源情况下，从多视角 2D 图像恢复 3D Gaussian 源参数的可行性。

## 背景

- E0 已验证 Gaussian PSF 与 MCX 高度一致（NCC > 0.997），物理基础成立
- E1 的问题：**给定物理正确的前向模型，反问题能不能解？**
- 最简设置：单灶、单 Gaussian、匀质介质、已知光参 → 纯粹验证优化可行性
- 不涉及：多灶、adaptive control、CT 残差网络、噪声鲁棒性（这些留给 E2）

## 目录结构

```
e1_single_source/
├── config.yaml              # 实验配置
├── psf_splatting.py         # PyTorch 可微 PSF splatting 前向模型
├── generate_data.py         # 生成多视角 GT 图像
├── optimize.py              # Per-sample 优化主循环
├── calibrate_psf.py         # 从 E0 校准 σ_PSF(d) 参数
├── README.md                # 本文件
└── results/
    ├── gt_data/             # GT 数据
    ├── optimization/        # 优化结果
    ├── convergence/         # 收敛曲线
    └── figures/             # 可视化
```

## 实验矩阵

| ID | 组织 | 深度 | σ | 位置 | 目的 |
|----|------|------|---|------|------|
| S01 | Muscle | 2.0mm | 1.0mm | 中心 | 基础 case |
| S02 | Muscle | 2.0mm | 0.5mm | 中心 | 小源 |
| S03 | Muscle | 4.0mm | 1.0mm | 中心 | 较深 |
| S04 | Muscle | 2.0mm | 1.0mm | (3,3)mm | 偏心位置 |
| S05 | Liver | 2.0mm | 1.0mm | 中心 | 高吸收 |
| S06 | Muscle | 2.0mm | 1.0mm | 中心 | 5 个随机种子 |

## 运行方法

```bash
# 1. 校准 PSF 参数（从 E0 结果）
cd /home/foods/pro/FMT-SimGen
uv run python pilot/e1_single_source/calibrate_psf.py

# 2. 生成 GT 数据
uv run python pilot/e1_single_source/generate_data.py

# 3. 运行优化
uv run python pilot/e1_single_source/optimize.py

# 完整流程（推荐）
uv run python pilot/e1_single_source/generate_data.py && \
uv run python pilot/e1_single_source/optimize.py
```

## 判定标准

| 判定 | 条件 | 后续动作 |
|------|------|----------|
| ✅ GO | S01-S05 全部：位置误差 < 0.5mm，σ 误差 < 20%，全部收敛 | 进入 E2：多灶 + adaptive control |
| ⚠️ CAUTION | S01-S03 GO 但 S04/S05 部分失败 | 分析失败原因，调整初始化或学习率 |
| ❌ NOGO | S01 基础 case 无法收敛 | 检查前向模型 + 梯度流 + 优化器配置 |

## 关键设计

### PSF 校准

从 E0 结果拟合 σ_PSF(d) 幂律关系：

```python
# Muscle
σ_PSF(d) = 0.8015 * d^0.5768

# Liver  
σ_PSF(d) = 0.5612 * d^0.5227
```

### Inverse Crime

E1 故意用相同模型做 GT 和前向：
- GT 和前向模型完全一致
- 排除前向模型误差干扰
- 纯粹测试优化能力
- E2 再引入 model mismatch

### 参数化

- σ 和 α 用 log-space 参数化（保证正值，梯度稳定）
- center 直接优化（3 个独立参数）
- 共 5 个可优化标量

### 初始化

- 在 GT 附近随机偏移（50% 范围内）
- S06 测试 5 个随机种子的鲁棒性

## 依赖

- Python 3.10+
- PyTorch 2.x (CUDA 推荐)
- NumPy, SciPy, PyYAML
- E0 的校准结果
