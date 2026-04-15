# E1b-Atlas: MCX vs Analytic Green on Real Atlas Surface

## Purpose

填补证据链的最后一个缺口：**在真实 Digimouse atlas 曲面上，验证解析 Green's function 与 MCX 蒙特卡洛结果的一致性**。

当前证据链：
- E0: 平面 ✓ (解析 ≈ MCX)
- E1d: atlas 曲面，但 GT 也是解析的 ✓ (自己比自己)
- **E1b: atlas 曲面 + MCX GT** ← **本实验，关键缺失环节**

## Quick Start

```bash
cd /home/foods/pro/FMT-SimGen/pilot/e1b_atlas_mcx

# 1. Build MCX volume from atlas
uv run python build_atlas_mcx_volume.py

# 2. Run MCX simulations (3 configs)
uv run python run_mcx_atlas.py --config E1b-A1
uv run python run_mcx_atlas.py --config E1b-A2
uv run python run_mcx_atlas.py --config E1b-A3

# 3. Compare and visualize
uv run python compare_mcx_vs_green.py

# 4. Generate paper figure
uv run python ../visualization/plot_e1b_atlas_mcx_vs_green.py
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Volume builder | 🟡 Partial | Needs FMT-SimGen M1 integration |
| MCX runner | 🟡 Partial | Needs surface sampling implementation |
| Comparison | 🟡 Partial | Needs completion |
| Visualization | 🟡 Partial | Framework ready, needs data |

## Key Technical Challenges

1. **Coordinate system alignment**: MCX volume coords ↔ atlas mesh coords
2. **Volume resolution**: 0.2mm recommended (balance speed/accuracy)
3. **Surface sampling**: Trilinear interpolation from 3D fluence to surface nodes
4. **Normalization**: MCX (photon count) vs Green (analytical) need normalization

## Expected Results

| Config | Source Position | Expected NCC | Go/No-Go |
|--------|----------------|--------------|----------|
| E1b-A1 | [17, 48, 10] (shallow) | >0.90 | GO |
| E1b-A2 | [17, 48, 6] (deep) | >0.85 | GO |
| E1b-A3 | [25, 48, 10] (lateral) | >0.85 | GO |

## Output

- `results/mcx_surface_E1b-*.npz`: MCX surface responses
- `results/green_surface_E1b-*.npz`: Analytic Green responses
- `results/comparison_E1b-*.json`: Metrics (NCC, RMSE, etc.)
- `../visualization/figures/fig_e1b_atlas_mcx_vs_green.pdf`: Paper figure

## Integration with Paper

This experiment provides Figure X: "MCX vs Analytic Green on Atlas Surface" - the final piece of evidence showing the analytic forward model is accurate even on realistic mouse torso geometry.
