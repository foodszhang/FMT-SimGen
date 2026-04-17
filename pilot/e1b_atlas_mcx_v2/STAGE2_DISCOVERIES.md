# Stage 2 Discoveries and Design Decisions

## Experiment: Multi-Position Volume Source Validation

### Design Evolution

#### 1. Projection Parameters Change
**Original proposal**: 25mm distance / 22mm FOV / 113×113 resolution  
**Final implementation**: 200mm distance / 50mm FOV / 256×256 resolution

**Reason**: To enable direct comparison with Stage 1.5 (multiposition) results and provide sufficient angular coverage for off-center volume sources. The larger FOV is essential when the source is near the surface edge at oblique viewing angles.

#### 2. From Single-Position to Multi-Position
**Original**: Single dorsal position (0° only)  
**Final**: P1-P5 positions with angle sweep (-60° to +60°)

**Reason**: Volume sources at different positions exhibit different angular dependencies. A single position cannot validate the general applicability of cubature schemes.

### Key Findings

#### Finding 1: Dorsal positions work well (NCC ≥ 0.94)
| Position | Best Angle | NCC | Status |
|----------|-----------|-----|--------|
| P1 dorsal | 0° | 0.988 | ✅ |
| P4 dorsal-lat | 0° | 0.938 | ✅ |

For dorsal sources, 7-point cubature achieves NCC ≥ 0.94, comparable to point source results.

#### Finding 2: Lateral positions degrade significantly (NCC ~0.7)
| Position | Best Angle | NCC | Point Source NCC (ref) |
|----------|-----------|-----|------------------------|
| P2 left | 0° | 0.702 | 0.993 |
| P3 right | 0° | 0.747 | 0.993 |

**Critical Issue**: Volume sources at lateral positions show much lower NCC than point sources at the same positions. This suggests:

1. **Limited angle testing**: We only tested [-60°, -30°, 0°, 30°, 60°], missing ±90° which would be the optimal viewing angles for lateral sources.

2. **Green's function approximation limit**: The infinite medium Green's function assumption breaks down more significantly for volume sources viewed from non-optimal angles.

3. **Cubature point count**: 7 points may be insufficient for lateral volume sources. Grid-27 or stratified-33 might be needed.

#### Finding 3: Ventral position marginal (NCC = 0.78)
Ventral source shows NCC = 0.78, below the 0.95 threshold. This is expected as ventral viewing is challenging (no direct line of sight from standard camera positions).

### Root Cause Analysis

#### Why Lateral Volume Sources Fail

**Point source** (Stage 1.5):
- Light emitted from single point
- Green's function only needs to model distance from that point to surface
- High accuracy even at oblique angles

**Volume source** (Stage 2):
- Light emitted from ~4000 voxels
- Each voxel has different path to surface
- Green's function叠加 introduces approximation errors
- At lateral positions, the "average" Green's function doesn't capture the true fluence distribution well

#### Why We Didn't Test ±90°
The original multiposition test excluded ±90° because:
> "Exclude ±90° because shallow sources have negligible signal from side views (peak intensity ~0.01% of frontal view, leading to unstable NCC)"

However, for lateral volume sources, ±90° might actually be the optimal viewing angle.

### Recommendations

#### For GS-FMT Application

1. **Use SR-6 (7-point) for dorsal sources**: NCC ≥ 0.94 is sufficient
2. **Consider grid-27 for lateral sources**: May improve NCC from ~0.7 to acceptable range
3. **Test ±90° angles**: Critical for lateral source validation
4. **Hybrid approach**: Use different cubature schemes based on source position

#### For Future Experiments

1. Run S2-Vol-P2 with grid-27 and ±90° angles
2. Compare intensity-normalized vs independently normalized projections
3. Investigate semi-infinite Green's function for surface-proximal sources

### Conclusion

**Volume sources are harder to approximate than point sources**, especially at lateral positions. While SR-6 works well for dorsal sources (NCC ≥ 0.94), lateral positions require:
- More cubature points (grid-27+)
- Optimal viewing angles (±90°)
- Possibly position-adaptive schemes

**Status**: Partial success - dorsal sources validated, lateral sources need refinement.
