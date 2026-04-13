# E0: PSF Validation Experiment

验证解析 Green's function 与 MCX 蒙特卡洛的一致性，为 GS-FMT 主线方案提供 gate experiment。

## 理论基础

**半无限介质 CW Green's function** (García et al. JBO 2026, Eq.14):

$$G_{semi}(\rho, d) = \frac{1}{4\pi D} \left( \frac{\exp(-\mu_{eff} r_1)}{r_1} - \frac{\exp(-\mu_{eff} r_2)}{r_2} \right)$$

其中：
- $r_1 = \sqrt{\rho^2 + d^2}$
- $r_2 = \sqrt{\rho^2 + (d + 2z_b)^2}$
- $z_b = 2AD$，$A = (1 + R_{eff}) / (1 - R_{eff})$

## 实验矩阵

| ID | 深度 (mm) | 组织类型 | μ_a (mm⁻¹) | μ_s' (mm⁻¹) |
|----|-----------|----------|------------|-------------|
| C01 | 1.5 | Muscle | 0.087 | 4.291 |
| C02 | 3.0 | Muscle | 0.087 | 4.291 |
| C03 | 5.0 | Muscle | 0.087 | 4.291 |
| C04 | 1.5 | Liver | 0.352 | 6.781 |
| C05 | 3.0 | Liver | 0.352 | 6.781 |
| C06 | 5.0 | Liver | 0.352 | 6.781 |
| C07 | 1.5 | Bilayer | - | - |
| C08 | 3.0 | Bilayer | - | - |
| C09 | 5.0 | Bilayer | - | - |

## 执行流程

```bash
# 1. 解析 PSF (秒级)
uv run python pilot/e0_psf_validation/analytic_psf.py

# 2. MCX 仿真 (每个配置 ~3-5 分钟)
uv run python pilot/e0_psf_validation/mcx_point_source.py

# 3. 三路对比 + go/no-go
uv run python pilot/e0_psf_validation/compare.py

# 4. 查看结果
cat pilot/e0_psf_validation/results/summary.json
```

## Go/No-Go 判定

| 判定 | 条件 | 后续 |
|------|------|------|
| GO | NCC > 0.90, FWHM ratio ∈ [0.8, 1.2] | 使用 Gaussian PSF |
| CAUTION | NCC ∈ [0.70, 0.90] | PSF + 残差网络 |
| NOGO | NCC < 0.70 | MCX 预计算查找表 |

## 输出文件

```
results/
├── profiles/
│   ├── C01_analytic.npz
│   ├── C01_mcx.npz
│   └── ...
├── figures/
│   ├── C01_comparison.png
│   └── summary_table.png
└── summary.json
```

## 关键注意事项

1. MCX 光源类型必须是 `isotropic`（各向同性点源）
2. 双层配置不参与 go/no-go 判定（解析公式假设匀质）
3. 体积大小 30×30×20 mm，体素 0.1mm，光子数 1e8