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
| -90° | 18434 | 0.0494 | 7.4221e+10 |
| -60° | 21773 | -0.0560 | 1.6010e+06 |
| -30° | 24854 | -0.0441 | 6.9125e+05 |
| +0° | 26042 | -0.0024 | 2.4859e+01 |
| +30° | 25789 | 0.0784 | 1.2641e+05 |
| +60° | 22745 | 0.3120 | 6.1340e+04 |
| +90° | 18434 | 0.0600 | 2.0099e+08 |
| +120° | 21773 | 0.2938 | 1.2646e+07 |
| +150° | 24854 | 0.6461 | 1.2779e+07 |
| +180° | 26042 | 0.1467 | 1.1255e+06 |

## Summary

- **Best angle**: +150° with **NCC = 0.6461**
- Worst angle: -60° with NCC = -0.0560
- Angles with NCC ≥ 0.9: **0**

## Key Answers

### 1. Which angles have linear NCC ≥ 0.9?

**NONE**

### 2. Highest NCC vs historical §4.C

- This run: **0.6461** @ +150°
- Historical §4.C: **0.9578** (P5-ventral)

**Match?** NO

### 3. If highest NCC < 0.85

**WARNING**: Highest NCC < 0.85. Historical numbers may have issues.
**STOP**: Do not proceed to downstream experiments.

## Figure

- `projection_comparison.png`: MCX vs Green projections for best/worst angles

## Protocol Change Audit

Git history shows `ec_y10.py` was created in commit:
```
dbb93b7 feat(paper04b): complete MVP pipeline D2.1-M6' + E-series experiments
```

This commit introduced the **vertex-based log-NCC** protocol, replacing the original **projection-based linear NCC** protocol.

The CSV column names remained the same, causing false consistency between incompatible protocols.
