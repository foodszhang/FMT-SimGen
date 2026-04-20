# MVP Pipeline Validation Summary

## Status (2026-04-19, Updated after D1-D3 Diagnosis)

### M0: Three-source closed-form validation ✅
- Total flux ratio validation passed

### M1: Single-source single-view ✅  
- NCC (log space) = 0.976
- k = 1.21e7
- Comparison plot: `results/m1_full_comparison.png`
- **VALID**: Uses atlas binary mask surface (no circular validation)

### M2: Three-source single-view ❌ (INVALID - Circular Validation)
- **Ball source**: NCC=0.9809 - INVALID (used MCX fluence mask)
- **Gaussian source**: NCC=0.9753 - INVALID (used MCX fluence mask)
- **Point source**: NCC=0.9127 - INVALID (used MCX fluence mask)
- **Must re-run with atlas binary mask surface**

### M3: Multi-view validation ❌
- **Issue identified**: Camera-space vs Surface-space mismatch
- See D2 report for correct physics explanation

---

## D1-D3 Diagnosis Results

### D1: valid_mask Audit

| Script | Surface Source | Circular? | Verdict |
|--------|----------------|-----------|---------|
| M1 | Atlas binary mask | NO | Valid |
| M2 | MCX fluence mask | YES | Invalid |
| M3 | (inherited from M2) | YES | Invalid |

### D2: Surface-Space NCC (Independent Validation)

Measured on **mesh vertices** (view-independent):

| Filter | N | NCC | Status |
|--------|---|-----|--------|
| All vertices | 290,808 | 0.860 | ⚠ Boundary |
| Distance ≤ 6mm | 17,758 | 0.866 | ⚠ Boundary |
| Distance ≤ 9mm | 50,548 | 0.850 | ⚠ Boundary |
| **Top 10% fluence** | **51,176** | **0.901** | **✓ Pass** |

**Conclusion**: Physics layer valid in **direct-path, high-fluence regime**.

### D3: Physics Explanation Correction

**WRONG** (previous):
> Green function ignores attenuation path

**CORRECT**:
> Closed-form forward uses infinite-medium Green's function $G_\infty(r) = \frac{\exp(-r/\delta)}{4\pi D r}$, which **includes** tissue attenuation via $\delta = \sqrt{D/\mu_a} \approx 0.95$ mm. The validity regime is:
> 1. Superficial source depth ≤ 9 mm
> 2. Direct-path view geometry (line from source to detector doesn't traverse organs)
> 
> Mismatch with MCX at non-0° views is due to:
> - Infinite-medium approximation (no Robin BC at tissue-air interface)
> - Homogeneous assumption (ignores organ boundaries with different μ_a, μ_s')
> - Camera projection vs surface fluence (different physical quantities)

---

## Root Cause Analysis: M3 Failure (Corrected)

### Camera-Space vs Surface-Space Mismatch

| Quantity | Definition | View-dependent? |
|----------|------------|-----------------|
| **Surface fluence** $\phi(\mathbf{r}_s)$ | Fluence at surface point | NO |
| **MCX projection** | Line integral through tissue + visibility | YES |

These are **different physical quantities**:
- Surface fluence: What a contact detector at that point would measure
- MCX projection: What a remote camera would see (attenuated by path length)

They coincide only when the surface point is directly visible from the source.

### Why 0° Works (Dorsal source, dorsal camera)

- Source is on dorsal surface
- Camera views dorsal surface directly
- Surface fluence ≈ MCX projection

### Why Other Angles Fail

For ventral camera viewing dorsal source:
- MCX projection: Integrates through entire body (liver, intestine, spine)
- Green surface fluence: Computed at ventral surface, but photons must traverse heterogeneous tissue

The NCC drop is **not** due to "ignoring attenuation" - it's due to:
1. **Heterogeneity**: Liver (μ_a=0.35) absorbs 4× more than soft tissue
2. **Robin BC missing**: Real tissue-air interface has ~50% internal reflection
3. **Path geometry**: Green uses Euclidean distance, real photons follow tortuous paths

---

## Valid Regime Definition

The closed-form forward is valid for:

1. **Superficial depth**: Source ≤ 6-9mm from surface
2. **Direct-path geometry**: Line from source to detector doesn't traverse organs
3. **Homogeneous tissue**: Not near organ boundaries (liver, bone, lung)

This is the **direct-path scope** for Paper-04b.

---

## Key Finding: Surface Selection (Corrected)

**Previous claim (WRONG)**:
> Green function must use fluence mask surface, not atlas binary mask surface

**Correct interpretation**:
> Using fluence mask surface creates **circular validation** - the evaluation domain is biased toward where MCX found photons. Use atlas binary mask surface for independent validation.

| Surface Source | NCC | Valid? |
|----------------|-----|--------|
| Atlas binary mask | 0.34-0.86 | ✓ Independent |
| Fluence mask | 0.97 | ✗ Circular |

The 0.97 NCC from fluence masking is **inflated** because it only evaluates where both methods succeed.

---

## Recommendations (Updated)

1. **Accept direct-path scope**: Through-organ views are out of scope for closed-form validation
2. **Re-run M2**: Use atlas binary mask surface (not fluence mask)
3. **Implement preflight**: Ray-marching to check direct-path condition
4. **Multi-view inversion**: Use MCX as forward model for through-organ views, OR restrict to direct-path views only

---

## Files (Updated)

- `m1_single_view.py` - Valid, uses atlas binary mask
- `m2_three_sources.py` - **DEPRECATED**, uses circular validation
- `m3_multi_view.py` - **DEPRECATED**, physics explanation incorrect
- `d2_surface_space_ncc.py` - Independent surface-space validation
- `d2b_analyze_distance.py` - Distance-based NCC analysis
- `d2c_superficial_regime.py` - Superficial regime analysis

## Results

- `results/d1_mask_audit.md` - D1 diagnosis report
- `results/d2_report.md` - D2 full report
- `results/d2/` - D2 data files and plots
