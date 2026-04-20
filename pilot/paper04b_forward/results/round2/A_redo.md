# P5-ventral Forward Metrics (Round 2)

## Configuration
- GT position: [-0.6, 2.4, -3.8] mm
- Optical: mua=0.0870/mm, mus'=4.3000/mm, delta=0.9439mm
- Voxel size: 0.4 mm

## Results

### Primary Metric (Linear NCC)
- **NCC (linear) = 0.6493**

### Auxiliary Metric (Log NCC)
- NCC (log) = 0.9578

### Scale Factor
- Scale (geomean) = 1.7219e+04
- Scale (old sum/sum) = 4.9048e+06
- Ratio = 0.0035

## Vertex Statistics
- Total vertices: 511771
- Direct vertices: 10468 (2.0%)
- Valid vertices: 8382

## Per-Distance-Bin Analysis

| Distance (mm) | N | NCC (linear) | NCC (log) |
|---------------|---|--------------|-----------|
| 0.0-0.5 | 17 | 0.4120 | 0.4116 |
| 0.5-1.0 | 44 | 0.4848 | 0.4994 |
| 1.0-1.5 | 123 | 0.5916 | 0.6047 |
| 1.5-2.0 | 164 | 0.6177 | 0.5832 |
| 2.0-2.5 | 234 | 0.6759 | 0.6275 |
| 2.5-3.0 | 280 | 0.6382 | 0.6202 |
| 3.0-3.5 | 393 | 0.5577 | 0.5984 |
| 3.5-4.0 | 410 | 0.2880 | 0.4116 |
| 4.0-4.5 | 436 | 0.1930 | 0.3384 |
| 4.5-5.0 | 392 | 0.1659 | 0.2883 |
| 5.0-5.5 | 391 | 0.1436 | 0.2645 |
| 5.5-6.0 | 348 | 0.1441 | 0.3202 |
| 6.0-6.5 | 330 | 0.0559 | 0.2336 |
| 6.5-7.0 | 324 | 0.0756 | 0.2111 |
| 7.0-7.5 | 327 | 0.0071 | 0.1579 |
| 7.5-8.0 | 304 | 0.0448 | 0.1344 |
| 8.0-8.5 | 310 | 0.0963 | 0.1605 |
| 8.5-9.0 | 293 | 0.0095 | 0.0972 |
| 9.0-9.5 | 336 | 0.0594 | 0.0942 |
| 9.5-10.0 | 305 | 0.0562 | 0.1510 |
| 10.0-10.5 | 298 | -0.0315 | 0.1204 |
| 10.5-11.0 | 255 | 0.0054 | 0.1549 |
| 11.0-11.5 | 276 | 0.0188 | 0.0604 |
| 11.5-12.0 | 243 | 0.0248 | 0.0838 |
| 12.0-12.5 | 236 | -0.0072 | 0.1198 |
| 12.5-13.0 | 223 | 0.0849 | 0.1262 |
| 13.0-13.5 | 262 | 0.0309 | 0.1664 |
| 13.5-14.0 | 206 | 0.0311 | 0.1138 |
| 14.0-14.5 | 134 | -0.0296 | 0.0035 |
| 14.5-15.0 | 66 | 0.0652 | 0.1501 |

## Distance-Bin Summary

| Metric | Value |
|--------|-------|
| NCC (linear) min | -0.0315 |
| NCC (linear) max | 0.6759 |
| NCC (linear) mean | 0.1837 |

## Comparison with §4.C Historical

| Metric | This run | Historical §4.C |
|--------|----------|-----------------|
| NCC (log) | 0.9578 | 0.9578 |
| NCC (linear) | 0.6493 | (not computed) |

**Note**: Historical §4.C used log-space NCC. Linear NCC was not computed.

## Figure

- `A7_ncc_vs_distance.png`: NCC vs distance curve

## Conclusion

- Linear NCC = **0.6493** (primary metric for paper)
- Log NCC = 0.9578 (matches historical 0.9578, confirms consistency)
- Scale geomean differs from sum/sum by factor 0.00
