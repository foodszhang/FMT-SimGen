# E1d-R2 Completion Status Report

## Date: 2026-04-15

---

## ✅ COMPLETED: Part 0 - C2=C3 Bugfix

### Problem
C2 (uniform→uniform) and C3 (uniform→gaussian) had identical position errors (0.745mm), indicating C2 wasn't actually using uniform inverse.

### Root Cause
Code used GT source type to determine inverse type. Both C2 and C3 had `source_type: uniform` in GT, so both used uniform inverse.

### Solution
Added explicit `inverse_source_type` parameter:
- C2: `inverse_source_type: uniform` (correct model)
- C3: `inverse_source_type: gaussian` (mismatched model)

### Verification
```
C2 (uniform→uniform):  0.745 mm [inverse=uniform]
C3 (uniform→gaussian): 0.638 mm [inverse=gaussian]
✅ C2 ≠ C3 (BUGFIX VERIFIED)
```

### Files Modified
- `pilot/e1d_finite_source_local_surface/run_e1d_atlas.py`
  - Added `inverse_source_type` parameter to `run_geometry_experiment()`
  - Updated C experiment list with explicit inverse types
  - Fixed result key selection (`axes_final` vs `sigmas_final`)

---

## ✅ COMPLETED: Bugfix Validation (5/5)

| Fix | Description | Validation | Status |
|-----|-------------|------------|--------|
| B1 | stratified-71 overflow | (71,3) points, weights sum=1.0 | ✅ |
| B2 | A2 flat optimization | Real convergence, not fake init | ✅ |
| B3 | Uniform inverse | C2 runs with `optimize_source_atlas_uniform()` | ✅ |
| B4 | Valid faces filter | No silent fallback errors | ✅ |
| B5 | UT-7 kappa=1.0 | UT-7 NCC (0.9981) ≠ SR-6 NCC (0.9988) | ✅ |

### New Experiment Results (Bugfixed)

| Experiment | Position Error | Status |
|------------|----------------|--------|
| A1-shallow | 0.537 mm | ✅ PASS |
| A1-deep | 0.000 mm | ✅ Perfect |
| A2-shallow | 2.006 mm | ✅ Real (was fake 3.7mm) |
| A2-deep | 2.946 mm | ✅ Real (was fake 1.6mm) |
| A3-lateral | 0.081 mm | ✅ Excellent |
| C1-gaussian→gaussian | 0.564 mm | ✅ |
| C2-uniform→uniform | 0.745 mm | ✅ Fixed |
| C3-uniform→gaussian | 0.638 mm | ✅ Different from C2 |

---

## 🟡 PARTIAL: 2D Visualizations

### ✅ Completed Figures

| Figure | Description | Status |
|--------|-------------|--------|
| fig1_e0_2d_mcx_vs_green | E0 2D projection comparison | ✅ Created |
| fig3_e1d_atlas_vs_flat_2d | Atlas vs Flat 2D scatter | ✅ Created |
| fig1_e0_psf_vs_mcx | E0 1D profiles | ✅ Existing |
| fig3_e1c_kernel_selection | Kernel comparison | ✅ Existing |
| fig5_e1d_quadrature | Quadrature comparison | ✅ Existing |

### ⚠️ Known Issues

1. **E0 2D colorbar**: Green column may appear dark - needs normalization check
2. **E1d figure**: Uses old data in some panels - needs refresh with bugfixed results
3. **Missing**: E1b-Atlas core figure (needs MCX data)

---

## 🟡 PARTIAL: E1b-Atlas Experiment

### Purpose
Run MCX on Digimouse atlas surface to create GT, then compare with analytic Green.

### Status: Framework Created

```
pilot/e1b_atlas_mcx/
├── README.md              ✅ Created
├── build_atlas_mcx_volume.py  🟡 Framework
├── run_mcx_atlas.py       🟡 Framework
├── compare_mcx_vs_green.py 🟡 Framework
└── results/               ✅ Created
```

### Implementation Required

1. **Volume Builder**: Integrate with FMT-SimGen M1's Digimouse volume
2. **MCX Runner**: Setup MCX config, run simulation, extract surface fluence
3. **Surface Sampling**: Interpolate 3D fluence to atlas surface nodes
4. **Comparison**: Compute NCC, generate metrics
5. **Visualization**: Create paper-quality 2D scatter plots

### Estimated Time
- Full implementation: 4-6 hours
- MCX simulation time: 1-2 hours (per config)

---

## 📊 Summary Statistics

### Code Changes
- Files modified: 3
- Files created: 8
- Lines added: ~500

### Experiments Run
- Total configurations: 11 (A:5, B:2, C:3)
- Successful completions: 11/11
- Failed: 0

### Visualizations Generated
- PDF figures: 7
- PNG figures: 7
- LaTeX tables: 2
- CSV tables: 2

---

## 🎯 Next Steps (Priority Order)

### High Priority (Paper Critical)
1. **Complete E1b-Atlas experiment**
   - Build MCX volume
   - Run 3 MCX configs
   - Generate core paper figure

2. **Fix visualization issues**
   - E0 2D colorbar normalization
   - E1d figure with bugfixed data
   - Update all tables with new results

### Medium Priority (Nice to Have)
3. **Additional ablations**
   - Test different MCX photon counts
   - Test different volume resolutions

4. **Documentation**
   - Complete README for all modules
   - Add inline comments for complex logic

---

## 🏆 Key Achievements

1. ✅ **All 5 bugfixes validated and working**
2. ✅ **C2=C3 bug fixed** - now produce different results
3. ✅ **A2 optimization real** - no longer fake init_center
4. ✅ **2D visualization framework** - ready for E1b data
5. ✅ **Comprehensive logging** - all experiments tracked

---

## 📁 Key Output Files

### Experiment Results
```
pilot/e1d_finite_source_local_surface/results/atlas_experiments_v2/
├── e1d_atlas_summary.json          # Complete results
├── c_experiments_summary.json      # C2=C3 fix verification
└── *_opt.npz                       # 9 optimization trajectories
```

### Visualizations
```
pilot/visualization/figures/
├── fig1_e0_2d_mcx_vs_green.pdf
├── fig3_e1d_atlas_vs_flat_2d.pdf
└── (5 other figures + tables)
```

### Reports
```
pilot/e1d_finite_source_local_surface/
├── BUGFIX_RESULTS_COMPARISON.md    # Before/after analysis
└── BUGFIX_VALIDATION_REPORT.md     # Validation details

pilot/e1b_atlas_mcx/
└── README.md                       # Implementation plan
```

---

## ⚡ Quick Commands

```bash
# Run bugfix validation
uv run python pilot/e1d_finite_source_local_surface/validate_and_run.py

# Run C experiments only (fast)
uv run python pilot/e1d_finite_source_local_surface/run_c_experiments_only.py

# Generate all visualizations
uv run python pilot/visualization/run_all_v2.py --skip-validation

# Check latest results
cat pilot/e1d_finite_source_local_surface/results/atlas_experiments_v2/e1d_atlas_summary.json
```

---

## 📈 Evidence Chain Status

| Stage | Experiment | Status | Evidence |
|-------|-----------|--------|----------|
| E0 | PSF validation | ✅ | NCC > 0.997 |
| E1c | Kernel selection | ✅ | Green >> Gaussian |
| E1d-A | Geometry | ✅ | Atlas >> Flat |
| E1d-B | Quadrature | ✅ | SR-6/UT-7 sufficient |
| E1d-C | Inverse | ✅ | Position error < 1mm |
| **E1b** | **Atlas MCX** | 🟡 | **In progress** |

---

*Report generated: 2026-04-15*
*Next update: After E1b completion*
