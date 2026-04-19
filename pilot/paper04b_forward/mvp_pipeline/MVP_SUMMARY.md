# MVP Pipeline Validation Summary

## Status (2026-04-19)

### M0: Three-source closed-form validation ✅
- Total flux ratio validation passed

### M1: Single-source single-view ✅  
- NCC (log space) = 0.976
- k = 1.21e7
- Comparison plot: `results/m1_full_comparison.png`

### M2: Three-source single-view ✅ (partial)
- **Ball source**: NCC=0.9809, k=6.0e6 ✓
- **Gaussian source**: NCC=0.9753, k=6.8e6 ✓
- **Point source**: NCC=0.9127, k=2.3e6 (slightly below 0.95 threshold)

### M3: Multi-view validation ❌
Only angle 0° passes. Other angles (90°, -90°, 180°) fail.

## Root Cause Analysis: M3 Failure

### Physical Mismatch
The Green function and MCX projection compute different physical quantities:

| Method | What it computes |
|--------|-----------------|
| **Green function** | Fluence at a surface point (assumes infinite homogeneous medium) |
| **MCX projection** | Integrated fluence along view direction through tissue geometry |

### Why 0° works
When the source is on the visible surface (dorsal source viewed from dorsal camera):
- The surface fluence from Green function ≈ MCX projection
- Light doesn't need to travel through much tissue

### Why other angles fail
When viewing from a different angle (e.g., 180° = ventral view):
- MCX projection: Integrates through tissue, sees attenuated fluence
- Green function: Still gives surface fluence, ignores attenuation path

The k factor mismatch is dramatic:
- 0°: k ~ 10^6 (reasonable)
- 180°: k ~ 1.0 (MCX projection much weaker due to tissue attenuation)

## Key Finding: Green Function Surface Selection

**Critical**: Green function must use **fluence mask surface**, not atlas binary mask surface.

| Surface source | NCC |
|---------------|-----|
| Atlas binary mask surface | 0.34 |
| **Fluence mask surface** | **0.97** |

This is because the Green function needs surface points where photons actually reach, not just anatomical surface.

## Recommendations

1. **Accept limitation**: Green function validation is only meaningful for the "best angle" where source is on visible surface
2. **Use archived approach**: Validate each source position at its specific best angle:
   - P1-dorsal: 0°
   - P2-left: 90°  
   - P3-right: -90°
   - P4-dorsal-lat: -30°
   - P5-ventral: 60°

3. **Multi-view inversion**: For actual FMT reconstruction with multi-view data, use MCX-generated projections as forward model, not Green function.

## Files

- `m2_three_sources.py` - Three source types at single view (angle 0°)
- `m3_multi_view.py` - Multi-view validation (showcases the limitation)

## Archived Data Reference

Located at: `pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/`

- 5 source positions (P1-P5), each validated at its "best angle"
- Uses archived volume: `mcx_volume_downsampled_2x.bin` (shape 95×100×52, voxel 0.4mm)
- Source position (P1-dorsal): [-0.6, 2.4, 5.8] mm in XYZ physical coordinates
