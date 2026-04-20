# Projection NCC Report — P5-ventral

## Protocol

This is the **original §4.C protocol**: 2D camera projection images with **linear NCC**.

NOT the vertex-based log-NCC that was accidentally substituted later.

## Configuration

- GT position: [-0.6, 2.4, -3.8] mm
- Voxel size: 0.4 mm
- Camera: distance=200.0mm, fov=50.0mm, resolution=(256, 256)
- Tissue: mua=0.0870/mm, mus'=4.3000/mm

## Results

| Angle | N_valid | Linear NCC | Scale |
|-------|---------|------------|-------|
| -90° | 18434 | 0.8584 | 7.3161e+05 |
| -60° | 21773 | 0.2989 | 4.3652e+04 |
| -30° | 24854 | 0.3150 | 5.8957e+04 |
| +0° | 26390 | 0.3891 | 6.2243e+04 |
| +30° | 25791 | 0.1562 | 2.5802e+04 |
| +60° | 22745 | 0.1794 | 1.3003e+04 |
| +90° | 18434 | 0.8896 | 3.1537e+05 |
| +120° | 21773 | 0.8985 | 9.0365e+05 |
| +150° | 24854 | 0.9212 | 1.0615e+06 |
| +180° | 26390 | 0.9269 | 1.1579e+06 |

## Summary

- **Best angle**: +180° with **NCC = 0.9269**
- Worst angle: +30° with NCC = 0.1562
- Angles with NCC ≥ 0.9: **2**

## Key Answers

### 1. Which angles have linear NCC ≥ 0.9?

- +150°: 0.9212
- +180°: 0.9269

### 2. Highest NCC vs historical §4.C

- This run: **0.9269** @ +180°
- Historical §4.C: **0.9578** (P5-ventral)

**Match?** YES

### 3. If highest NCC < 0.85

Highest NCC = 0.9269 ≥ 0.85. Protocol appears correct.

## Figure

- `projection_comparison.png`: MCX vs Green projections for best/worst angles

## Protocol Change Audit

Git history shows `ec_y10.py` was created in commit:
```
dbb93b7 feat(paper04b): complete MVP pipeline D2.1-M6' + E-series experiments
```

This commit introduced the **vertex-based log-NCC** protocol, replacing the original **projection-based linear NCC** protocol.

The CSV column names remained the same, causing false consistency between incompatible protocols.
