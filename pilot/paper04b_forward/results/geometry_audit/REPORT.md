# Geometry Alignment Audit Report — P5-ventral (Y=10)

## Configuration
- gt_pos: [-0.6, 2.4, -3.8] mm
- voxel_size: 0.4 mm
- atlas path: pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin
- fluence path: pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P5-ventral-r2.0/fluence.npy
- vertices count: 511771

## G-1: MCX archive source consistency
- archive config path: pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/S2-Vol-P5-ventral-r2.0/results.json
- archive source (raw): [-0.5999999999999996, 2.4, -3.8000000000000007]
- archive source (mm): [-0.5999999999999996, 2.4, -3.8000000000000007]
- code gt_pos (mm): [-0.6, 2.4, -3.8]
- difference: 0.0000 mm
- **Result**: PASS
- Notes: 

## G-2: Fluence argmax vs gt_pos
- fluence shape: [95, 100, 52]
- fluence range: [0.000e+00, 1.605e+06]
- peak voxel: (46, 56, 16)
- peak mm: [-0.6000000000000001, 2.4000000000000004, -4.0]
- |peak - gt|: 0.2000 mm
- **Result**: PASS
- Figure: G2_fluence_peak.png

## G-3: Atlas label at gt voxel
- atlas shape: [95, 100, 52]
- gt voxel: (46, 56, 16)
- label at gt: 1
- 3x3x3 neighborhood: {0: 12, 1: 11, 2: 1, 7: 3}
- **Result**: FAIL

## G-4: Vertex label distribution
- vertices count: 511771
- extent X: [-16.00, 18.00] mm
- extent Y: [-20.00, 19.60] mm
- extent Z: [-10.40, 10.00] mm
- label in {0,1} (air/soft_tissue): 84.8%
- label >= 2 (organ): 15.2%
- **Result**: FAIL
- Figure: G4_vertex_label_hist.png

## G-5: Projection alignment @ -60° view
- angle: -60°
- gt_proj pixel (u, v): [143, 140]
- hotspot pixel (u, v): [143, 140]
- pixel distance: 0.00
- **Result**: PASS
- Figure: G5_projection_alignment.png

## G-6: 3D overlay visual check
- HTML: G6_3d_overlay.html
- [ ] GT inside atlas surface: manual verification required
- [ ] GT at fluence iso center: manual verification required
- [ ] fluence shape plausible: manual verification required
- **Result**: PASS (visual verification required)

## Overall verdict
- FAIL at G-3, G-4 → Stop and wait for user instruction
