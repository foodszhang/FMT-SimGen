# Frame Literal Audit (H1)

**Generated**: 2026-04-21  
**Total raw hits**: 954  
**Unique (deduped) hits**: 859  

## Verdict Legend
- `replace_with_import`: literal must be replaced with import from `frame_contract`  
- `legitimate_non_frame`: not a frame literal (add to whitelist in `check_frame_literals.py`)  
- `TODO`: needs review/fix before H1 pass  

| File | Line | Value | Pattern | Verdict | Context |
|------|------|-------|---------|---------|--------|
| `FMT-SimGen/fmt_simgen/atlas/digimouse.py` | 492 | `30` | TRUNK_OFFSET legacy Y | TODO | `y_head_threshold = int(30 / voxel_size)` |
| `FMT-SimGen/fmt_simgen/atlas/digimouse.py` | 493 | `30` | TRUNK_OFFSET legacy Y | TODO | `y_tail_threshold = y_dim - int(30 / voxel_size)` |
| `FMT-SimGen/fmt_simgen/config/view_contract.py` | 9 | `30` | TRUNK_OFFSET legacy Y | TODO | `VIEW_ANGLES: Final[list[int]] = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/fmt_simgen/config/view_contract.py` | 15 | `200` | TRUNK_GRID_SHAPE Y | TODO | `VIEW_CAMERA_DISTANCE_MM: Final[float] = 200.0` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 300 | `190` | TRUNK_GRID_SHAPE X | TODO | `mcx_zyx = mcx_raw.reshape(TRUNK_GRID_SHAPE[::-1])  # ZYX = (104, 200, 190)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 300 | `200` | TRUNK_GRID_SHAPE Y | TODO | `mcx_zyx = mcx_raw.reshape(TRUNK_GRID_SHAPE[::-1])  # ZYX = (104, 200, 190)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 300 | `104` | TRUNK_GRID_SHAPE Z | TODO | `mcx_zyx = mcx_raw.reshape(TRUNK_GRID_SHAPE[::-1])  # ZYX = (104, 200, 190)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 345 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size: float        = VOXEL_SIZE_MM                  # 0.2` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 346 | `190` | TRUNK_GRID_SHAPE X | TODO | `grid_shape_xyz: tuple    = TRUNK_GRID_SHAPE               # (190, 200, 104)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 346 | `200` | TRUNK_GRID_SHAPE Y | TODO | `grid_shape_xyz: tuple    = TRUNK_GRID_SHAPE               # (190, 200, 104)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 346 | `104` | TRUNK_GRID_SHAPE Z | TODO | `grid_shape_xyz: tuple    = TRUNK_GRID_SHAPE               # (190, 200, 104)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 368 | `30` | TRUNK_OFFSET legacy Y | TODO | `roi_size: float = self.dataset_config.get("voxel_grid_roi_size_mm", 30.0)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 384 | `20` | VOLUME_CENTER_WORLD Y | TODO | `"frame_contract_version": "2026-04-20-U2-locked",` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 441 | `30` | TRUNK_OFFSET legacy Y | TODO | `roi_size = self.dataset_config.get("voxel_grid_roi_size_mm", 30.0)` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 457 | `0.2` | VOXEL_SIZE_MM | TODO | `trunk_voxel_size = float(mcx_cfg["voxel_size_mm"])  # 0.2` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 465 | `20` | VOLUME_CENTER_WORLD Y | TODO | `)  # ≈ [38.0, 40.0, 20.8]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 465 | `38` | TRUNK_SIZE_MM X | TODO | `)  # ≈ [38.0, 40.0, 20.8]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 465 | `40` | TRUNK_SIZE_MM Y | TODO | `)  # ≈ [38.0, 40.0, 20.8]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 465 | `20.8` | TRUNK_SIZE_MM Z | TODO | `)  # ≈ [38.0, 40.0, 20.8]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 500 | `19` | VOLUME_CENTER_WORLD X | TODO | `roi_center = trunk_size_mm / 2.0  # ≈ [19, 20, 10.4]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 500 | `20` | VOLUME_CENTER_WORLD Y | TODO | `roi_center = trunk_size_mm / 2.0  # ≈ [19, 20, 10.4]` |
| `FMT-SimGen/fmt_simgen/dataset/builder.py` | 500 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `roi_center = trunk_size_mm / 2.0  # ≈ [19, 20, 10.4]` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 13 | `19` | VOLUME_CENTER_WORLD X | TODO | `(derived from TRUNK_SIZE_MM / 2 = (19.0, 20.0, 10.4) mm).` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 13 | `20` | VOLUME_CENTER_WORLD Y | TODO | `(derived from TRUNK_SIZE_MM / 2 = (19.0, 20.0, 10.4) mm).` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 13 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `(derived from TRUNK_SIZE_MM / 2 = (19.0, 20.0, 10.4) mm).` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 55 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 207 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 336 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 359 | `30` | TRUNK_OFFSET legacy Y | TODO | `Projection keys: "-90", "-60", "-30", "0", "30", "60", "90" → [H×W] float32` |
| `FMT-SimGen/fmt_simgen/mcx_projection.py` | 387 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/fmt_simgen/mcx_source.py` | 35 | `0.2` | VOXEL_SIZE_MM | TODO | `- voxel_size_mm: float (e.g., 0.2)` |
| `FMT-SimGen/fmt_simgen/mcx_volume.py` | 27 | `20` | VOLUME_CENTER_WORLD Y | TODO | `{"label": 2, "name": "bone",        "mua": 0.04,     "mus_prime": 20.0,       "g": 0.90, "n": 1.37},` |
| `FMT-SimGen/fmt_simgen/mcx_volume.py` | 70 | `19` | VOLUME_CENTER_WORLD X | TODO | `19: 8,  # kidney` |
| `FMT-SimGen/fmt_simgen/mcx_volume.py` | 71 | `20` | VOLUME_CENTER_WORLD Y | TODO | `20: 8,  # adrenal -> kidney` |
| `FMT-SimGen/fmt_simgen/mesh/mesh_generator.py` | 76 | `0.1` | ATLAS voxel size | TODO | `voxel_size: float = 0.1,` |
| `FMT-SimGen/fmt_simgen/mesh/mesh_generator.py` | 87 | `0.1` | ATLAS voxel size | TODO | `voxel_size : float, default 0.1` |
| `FMT-SimGen/fmt_simgen/mesh/mesh_generator.py` | 177 | `30` | TRUNK_OFFSET legacy Y | TODO | `"y": (30.0, 70.0),    # trunk Y extent: 30..70mm atlas (matches MCX crop)` |
| `FMT-SimGen/fmt_simgen/mesh/mesh_generator.py` | 178 | `20` | VOLUME_CENTER_WORLD Y | TODO | `"z": (-1.0, 21.8),    # trunk Z extent: 0..20.8mm` |
| `FMT-SimGen/fmt_simgen/physics/fem_solver.py` | 61 | `19` | VOLUME_CENTER_WORLD X | TODO | `19: 10,  # kidney` |
| `FMT-SimGen/fmt_simgen/physics/fem_solver.py` | 62 | `20` | VOLUME_CENTER_WORLD Y | TODO | `20: 10,  # adrenal` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 227 | `30` | TRUNK_OFFSET legacy Y | TODO | `- num_foci_distribution: Dict[int, float] - {1: 0.30, 2: 0.35, 3: 0.35}` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 251 | `30` | TRUNK_OFFSET legacy Y | TODO | `[FIX v3] Offset from atlas-corner to trunk-local world frame [0,30,0].` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 264 | `0.2` | VOXEL_SIZE_MM | TODO | `(defaults to 0.2 if not provided).` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 285 | `0.2` | VOXEL_SIZE_MM | TODO | `self.gt_spacing_mm = gt_spacing_mm if gt_spacing_mm is not None else 0.2` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 289 | `30` | TRUNK_OFFSET legacy Y | TODO | `"num_foci_distribution", {1: 0.30, 2: 0.35, 3: 0.35}` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 685 | `20` | VOLUME_CENTER_WORLD Y | TODO | `X: [2.4, 34.4] mm, Y: [4.8, 92.8] mm, Z: [1.6, 20.0] mm` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 685 | `34` | TRUNK_OFFSET_ATLAS_MM Y | TODO | `X: [2.4, 34.4] mm, Y: [4.8, 92.8] mm, Z: [1.6, 20.0] mm` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 689 | `20` | VOLUME_CENTER_WORLD Y | TODO | `Trunk region = Y in [20, 70] mm (exclude head and tail)` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 703 | `19` | VOLUME_CENTER_WORLD X | TODO | `z_center = max(10.5, min(19.0, z_center))` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 705 | `19` | VOLUME_CENTER_WORLD X | TODO | `z_center = self._rng.uniform(15.0, 19.0)` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 709 | `20` | VOLUME_CENTER_WORLD Y | TODO | `self._rng.uniform(20.0, 70.0),` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 729 | `20` | VOLUME_CENTER_WORLD Y | TODO | `self._rng.uniform(20.0, 70.0),` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 737 | `19` | VOLUME_CENTER_WORLD X | TODO | `z_center = max(10.5, min(19.0, z_center))` |
| `FMT-SimGen/fmt_simgen/tumor/tumor_generator.py` | 743 | `20` | VOLUME_CENTER_WORLD Y | TODO | `self._rng.uniform(20.0, 70.0),` |
| `FMT-SimGen/fmt_simgen/view_config.py` | 9 | `38` | TRUNK_SIZE_MM X | TODO | `- X: left(-19mm) → right(+19mm), range [0, 38] mm` |
| `FMT-SimGen/fmt_simgen/view_config.py` | 43 | `30` | TRUNK_OFFSET legacy Y | TODO | `self.angles: list[int] = config.get("angles", [-90, -60, -30, 0, 30, 60, 90])` |
| `FMT-SimGen/fmt_simgen/view_config.py` | 45 | `200` | TRUNK_GRID_SHAPE Y | TODO | `self.camera_distance_mm: float = config.get("camera_distance_mm", 200.0)` |
| `FMT-SimGen/fmt_simgen/view_config.py` | 132 | `0.2` | VOXEL_SIZE_MM | TODO | `depth_tolerance_mm: float = 0.2,` |
| `FMT-SimGen/fmt_simgen/view_config.py` | 414 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm = 200.0` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_magnitude.py` | 76 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Metric':<20} {'MCX':<25} {'Green':<25} {'Ratio':<15}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_magnitude.py` | 83 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{key:<20} {mcx_val:<25.6e} {green_val:<25.6e} {ratio:<15.6e}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_magnitude.py` | 86 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{'n_nonzero':<20} {mcx_stats['n_nonzero']:<25d} {green_stats['n_nonzero']:<25d}"` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_magnitude.py` | 420 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"{'Position':<30} {'k_sum':<15} {'R²':<10} {'Go/No-Go':<10}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_magnitude.py` | 426 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"{d['position_id']:<30} {k_sum:<15.4e} {r_sq:<10.4f} {go:<10}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_normalization_summary.py` | 141 | `30` | TRUNK_OFFSET legacy Y | TODO | `/ f"mcx_a{int({'Dorsal': 0, 'Left': 90, 'Right': -90, 'Dorsal-Lateral': -30, 'Ventral': 60}[name])}.npy"` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_normalization_summary.py` | 146 | `30` | TRUNK_OFFSET legacy Y | TODO | `/ f"green_a{int({'Dorsal': 0, 'Left': 90, 'Right': -90, 'Dorsal-Lateral': -30, 'Ventral': 60}[name])}.npy"` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 28 | `190` | TRUNK_GRID_SHAPE X | TODO | `original = original.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 28 | `200` | TRUNK_GRID_SHAPE Y | TODO | `original = original.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 28 | `104` | TRUNK_GRID_SHAPE Z | TODO | `original = original.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 44 | `200` | TRUNK_GRID_SHAPE Y | TODO | `y_vox = int(round((y_mm + 50) / 100 * 200))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 58 | `0.1` | ATLAS voxel size | TODO | `left_x_mm = (left_x_vox - 95) * 0.1` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 59 | `0.1` | ATLAS voxel size | TODO | `right_x_mm = (right_x_vox - 95) * 0.1` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 77 | `19` | VOLUME_CENTER_WORLD X | TODO | `ix = int(round((source_pos[0] + 19) / 38 * 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 77 | `38` | TRUNK_SIZE_MM X | TODO | `ix = int(round((source_pos[0] + 19) / 38 * 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 77 | `190` | TRUNK_GRID_SHAPE X | TODO | `ix = int(round((source_pos[0] + 19) / 38 * 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 79 | `20` | VOLUME_CENTER_WORLD Y | TODO | `iz = int(round((source_pos[2] + 10) / 20 * 104))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 79 | `104` | TRUNK_GRID_SHAPE Z | TODO | `iz = int(round((source_pos[2] + 10) / 20 * 104))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 83 | `190` | TRUNK_GRID_SHAPE X | TODO | `if 0 <= iz < 104 and 0 <= iy < 200 and 0 <= ix < 190:` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 83 | `200` | TRUNK_GRID_SHAPE Y | TODO | `if 0 <= iz < 104 and 0 <= iy < 200 and 0 <= ix < 190:` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p2_geometry.py` | 83 | `104` | TRUNK_GRID_SHAPE Z | TODO | `if 0 <= iz < 104 and 0 <= iy < 200 and 0 <= ix < 190:` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 35 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 35 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 35 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 96 | `200` | TRUNK_GRID_SHAPE Y | TODO | `n_samples: int = 200,` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 183 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_geometry.py` | 226 | `200` | TRUNK_GRID_SHAPE Y | TODO | `labels = trace_ray_labels(source_vox, detector_vox, volume_xyz, n_samples=200)` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 38 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 38 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 38 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 234 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 313 | `30` | TRUNK_OFFSET legacy Y | TODO | `if np.mean(liver_percentages) >= 30:` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_p5_tissue_path.py` | 314 | `30` | TRUNK_OFFSET legacy Y | TODO | `print("  ✓ Liver occupies ≥30% of P5 paths → HETEROGENEITY CONFIRMED")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_scale_theory.py` | 169 | `0.1` | ATLAS voxel size | TODO | `elif 0.1 < ratio < 10.0:` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_units.py` | 218 | `30` | TRUNK_OFFSET legacy Y | TODO | `ratio = mcx_v / (green_v + 1e-30)` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_units.py` | 289 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"{'Position':<30} {'k_raw':<15} {'k_absolute':<15} {'k_expected':<15}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diag_units.py` | 295 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"{d['position_id']:<30} {k_raw:<15.4e} {k_abs:<15.4e} {k_exp:<15.4e}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diagnostic_summary.py` | 42 | `30` | TRUNK_OFFSET legacy Y | TODO | `("S2-Vol-P4-dorsal-lat-r2.0", "P4-dorsal-lat", -30),` |
| `FMT-SimGen/pilot/_archive/diag_v2/diagnostic_summary.py` | 48 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Position':<20} {'k_sum':<15} {'k_max':<15} {'NCC':<10} {'Status':<15}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/diagnostic_summary.py` | 72 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{name:<20} {k_sum:<15.4e} {k_max:<15.4e} {ncc:<10.4f} {status:<15}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/final_geometry_summary.py` | 101 | `30` | TRUNK_OFFSET legacy Y | TODO | `("S2-Vol-P4-dorsal-lat-r2.0", "P4-dorsal-lat", -30),` |
| `FMT-SimGen/pilot/_archive/diag_v2/final_geometry_summary.py` | 106 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Position':<20} {'NCC':<10} {'k_sum':<15} {'Status':<20}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/final_geometry_summary.py` | 124 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{name:<20} {results['ncc']:<10.4f} {k:<15.4e} {status:<20}")` |
| `FMT-SimGen/pilot/_archive/diag_v2/final_geometry_summary.py` | 134 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{'P5-ventral (orig)':<20} {p5_orig['ncc']:<10.4f} {k_p5_orig:<15.4e} {'✗ FAIL (in liver)':<20}"` |
| `FMT-SimGen/pilot/_archive/diag_v2/final_geometry_summary.py` | 142 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{'P5-ventral (fixed)':<20} {p5_corr['ncc']:<10.4f} {p5_corr['k_sum']:<15.4e} {'~ ACCEPTABLE':<20}"` |
| `FMT-SimGen/pilot/_archive/diag_v2/plot_p5_comparison.py` | 56 | `30` | TRUNK_OFFSET legacy Y | TODO | `ratio = mcx / (green + 1e-30)` |
| `FMT-SimGen/pilot/_archive/diag_v2/plot_p5_comparison.py` | 158 | `30` | TRUNK_OFFSET legacy Y | TODO | `ratio_orig = mcx_orig / (green_orig + 1e-30)` |
| `FMT-SimGen/pilot/_archive/diag_v2/plot_p5_comparison.py` | 188 | `0.1` | ATLAS voxel size | TODO | `0.1,` |
| `FMT-SimGen/pilot/_archive/diag_v2/plot_summary.py` | 25 | `30` | TRUNK_OFFSET legacy Y | TODO | `("S2-Vol-P4-dorsal-lat-r2.0", "P4-dorsal-lat", -30),` |
| `FMT-SimGen/pilot/_archive/diag_v2/rerun_p5_corrected.py` | 43 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/_archive/diag_v2/rerun_p5_corrected.py` | 53 | `30` | TRUNK_OFFSET legacy Y | TODO | `"P4-dorsal-lat": -30.0,` |
| `FMT-SimGen/pilot/e0_psf_validation/analytic_psf.py` | 125 | `0.1` | ATLAS voxel size | TODO | `sigma0 = max(sigma0, 0.1)  # 保底` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 34 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 34 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 35 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm: float = 0.1,` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 88 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 88 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 89 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm: float = 0.1,` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 341 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm: float = 0.1,` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 390 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm: float = 0.1,` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 391 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 391 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e0_psf_validation/mcx_point_source.py` | 525 | `0.1` | ATLAS voxel size | TODO | `voxel_size = sim_params.get("voxel_size_mm", 0.1)` |
| `FMT-SimGen/pilot/e1_single_source/optimize.py` | 65 | `0.1` | ATLAS voxel size | TODO | `init_sigma = max(init_sigma, 0.1)` |
| `FMT-SimGen/pilot/e1_single_source/optimize.py` | 67 | `0.1` | ATLAS voxel size | TODO | `init_alpha = max(init_alpha, 0.1)` |
| `FMT-SimGen/pilot/e1_single_source/optimize.py` | 87 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: tuple = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1_single_source/optimize.py` | 87 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: tuple = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1_single_source/optimize.py` | 175 | `20` | VOLUME_CENTER_WORLD Y | TODO | `converged = loss_range / (max(recent_loss) + 1e-20) < 0.01` |
| `FMT-SimGen/pilot/e1_single_source/psf_splatting.py` | 98 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1_single_source/psf_splatting.py` | 98 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: Tuple[float, float, float] = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1_single_source/psf_splatting.py` | 135 | `20` | VOLUME_CENTER_WORLD Y | TODO | `T_peak = torch.clamp(T_peak, min=1e-20)` |
| `FMT-SimGen/pilot/e1_single_source/psf_splatting.py` | 167 | `0.1` | ATLAS voxel size | TODO | `d_clamped = torch.clamp(d, min=0.1, max=15.0)` |
| `FMT-SimGen/pilot/e1_single_source/psf_splatting.py` | 223 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles_deg = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx/run_e1b_experiment.py` | 29 | `30` | TRUNK_OFFSET legacy Y | TODO | `def load_atlas_surface(mesh_path: Path, roi_center=None, roi_radius=30.0):` |
| `FMT-SimGen/pilot/e1b_atlas_mcx/run_e1b_experiment.py` | 164 | `30` | TRUNK_OFFSET legacy Y | TODO | `surface_coords = load_atlas_surface(mesh_path, roi_center, roi_radius=30.0)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_coord_ranges.py` | 6 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_coord_ranges.py` | 6 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_coord_ranges.py` | 6 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_coord_ranges.py` | 15 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_coord_ranges.py` | 39 | `30` | TRUNK_OFFSET legacy Y | TODO | `mcx_offset = [0, 30, 0]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_projection_peaks.py` | 32 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_projection_peaks.py` | 35 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_dist = 200` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_projection_peaks.py` | 36 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_projection_peaks.py` | 106 | `0.1` | ATLAS voxel size | TODO | `elif ratio_90 < 0.1 or ratio_m90 < 0.1:` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_stage1_5_projection.py` | 26 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_stage1_5_projection.py` | 26 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_stage1_5_projection.py` | 26 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_stage1_5_projection.py` | 67 | `20` | VOLUME_CENTER_WORLD Y | TODO | `green_log = np.log10(green_proj + 1e-20)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_check_stage1_5_projection.py` | 68 | `20` | VOLUME_CENTER_WORLD Y | TODO | `im4 = axes[1, 1].imshow(green_log, cmap="hot", vmin=-20, vmax=-15)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 8 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 8 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 8 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 9 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 9 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 9 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 17 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2  # mm` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 87 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"\nMCX config offset: [0, 30, 0] mm")` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_coordinates.py` | 89 | `30` | TRUNK_OFFSET legacy Y | TODO | `mcx_y_vox = (source_xy[1] - 30) / voxel_size` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_current_run.py` | 18 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_current_run.py` | 18 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_current_run.py` | 18 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_current_run.py` | 23 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_current_run.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 22 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 22 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 22 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 27 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 28 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_green_peak.py` | 112 | `20` | VOLUME_CENTER_WORLD Y | TODO | `green_log = np.log10(green_proj + 1e-20)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_surface_coords.py` | 18 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_surface_coords.py` | 18 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_surface_coords.py` | 18 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_surface_coords.py` | 25 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_debug_surface_coords.py` | 26 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 18 | `0.2` | VOXEL_SIZE_MM | TODO | `VOXEL_SIZE_MM = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 19 | `190` | TRUNK_GRID_SHAPE X | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 19 | `200` | TRUNK_GRID_SHAPE Y | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 19 | `104` | TRUNK_GRID_SHAPE Z | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 32 | `30` | TRUNK_OFFSET legacy Y | TODO | `center = np.array([0.0, 30.0, 4.0])` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 72 | `30` | TRUNK_OFFSET legacy Y | TODO | `atlas = ((xx / 18.0) ** 2 + ((yy - 30) / 22.0) ** 2 + (zz / 9.0) ** 2) <= 1.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 110 | `30` | TRUNK_OFFSET legacy Y | TODO | `source_pos = np.array([0.0, 30.0, 4.0])` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 140 | `30` | TRUNK_OFFSET legacy Y | TODO | `atlas = ((xx / 18.0) ** 2 + ((yy - 30) / 22.0) ** 2 + (zz / 9.0) ** 2) <= 1.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_stage2_components.py` | 144 | `30` | TRUNK_OFFSET legacy Y | TODO | `source_center = np.array([0.0, 30.0, 4.0])` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 29 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 29 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 29 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 30 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 30 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 30 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_test_surface_projection.py` | 110 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [-60, -30, 30, 60]:` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_verify_source_position.py` | 8 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_verify_source_position.py` | 8 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_verify_source_position.py` | 8 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_verify_source_position.py` | 12 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/_verify_source_position.py` | 22 | `30` | TRUNK_OFFSET legacy Y | TODO | `mcx_offset = np.array([0, 30, 0])` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 17 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 17 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 17 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 22 | `200` | TRUNK_GRID_SHAPE Y | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 22 | `0.2` | VOXEL_SIZE_MM | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_multi_position.py` | 27 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig, axes = plt.subplots(5, 3, figsize=(12, 20))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 18 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 18 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 18 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 24 | `0.2` | VOXEL_SIZE_MM | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/check_projection_shapes.py` | 88 | `20` | VOLUME_CENTER_WORLD Y | TODO | `im4 = axes[1, 1].imshow(green_log, cmap="hot", vmin=-20, vmax=0)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/debug_p2_projection.py` | 25 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/debug_p2_projection.py` | 30 | `200` | TRUNK_GRID_SHAPE Y | TODO | `fluence, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/debug_p2_projection.py` | 30 | `0.2` | VOXEL_SIZE_MM | TODO | `fluence, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/debug_projection.py` | 98 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/debug_projection.py` | 101 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_before_after_comparison.py` | 19 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_before_after_comparison.py` | 19 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_before_after_comparison.py` | 19 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_before_after_comparison.py` | 25 | `200` | TRUNK_GRID_SHAPE Y | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_before_after_comparison.py` | 25 | `0.2` | VOXEL_SIZE_MM | TODO | `atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 24 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 24 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 36 | `200` | TRUNK_GRID_SHAPE Y | TODO | `atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 36 | `0.2` | VOXEL_SIZE_MM | TODO | `atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 41 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig, axes = plt.subplots(5, 4, figsize=(16, 20))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 102 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Position':<20} {'Best Angle':<12} {'NCC':<8} {'RMSE':<8}")` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_multiposition_results.py` | 106 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{r['config_id']:<20} {r['best_angle']:>6.0f}°      {r['ncc_best']:>6.3f}   {r['rmse_best']:>6.4f}"` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_projection_comparison.py` | 22 | `200` | TRUNK_GRID_SHAPE Y | TODO | `fluence_mcx, angle, 200, 50, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_projection_comparison.py` | 22 | `0.2` | VOXEL_SIZE_MM | TODO | `fluence_mcx, angle, 200, 50, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_projection_comparison.py` | 25 | `200` | TRUNK_GRID_SHAPE Y | TODO | `fluence_green, angle, 200, 50, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_projection_comparison.py` | 25 | `0.2` | VOXEL_SIZE_MM | TODO | `fluence_green, angle, 200, 50, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_projection_comparison.py` | 89 | `20` | VOLUME_CENTER_WORLD Y | TODO | `cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1.py` | 80 | `0.1` | ATLAS voxel size | TODO | `ax.axhspan(ncc_go, 1.0, alpha=0.1, color="#06A77D")` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1.py` | 81 | `0.1` | ATLAS voxel size | TODO | `ax.axhspan(ncc_caution, ncc_go, alpha=0.1, color="#F4A261")` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1.py` | 82 | `0.1` | ATLAS voxel size | TODO | `ax.axhspan(0, ncc_caution, alpha=0.1, color="#E63946")` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1.py` | 124 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_dist = 200` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1.py` | 125 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_projections.py` | 110 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig2, axes2 = plt.subplots(3, 5, figsize=(20, 12))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_projections.py` | 111 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-60, -30, 0, 30, 60]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_with_outline.py` | 19 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_with_outline.py` | 19 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_with_outline.py` | 19 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_with_outline.py` | 24 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_5_with_outline.py` | 25 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_comparison.py` | 49 | `0.1` | ATLAS voxel size | TODO | `ax.fill_between(depths, 0.95, 1.0, alpha=0.1, color='#06A77D')` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_comparison.py` | 50 | `0.1` | ATLAS voxel size | TODO | `ax.fill_between(depths, 0.85, 0.95, alpha=0.1, color='#F4A261')` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_comparison.py` | 51 | `0.1` | ATLAS voxel size | TODO | `ax.fill_between(depths, 0, 0.85, alpha=0.1, color='#E63946')` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_comparison.py` | 58 | `0.1` | ATLAS voxel size | TODO | `ax.set_ylim(-0.1, 1.05)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_fixed.py` | 135 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_dist = 200` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_fixed.py` | 136 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_fixed.py` | 263 | `20` | VOLUME_CENTER_WORLD Y | TODO | `"Normalized Intensity\n(to shallowest MCX peak)", rotation=270, labelpad=20` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_v2.py` | 129 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_dist = 200` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_v2.py` | 130 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_v2.py` | 242 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_dist = 200` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_v2.py` | 243 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage1_v2.py` | 330 | `20` | VOLUME_CENTER_WORLD Y | TODO | `cbar.set_label("Normalized Intensity\n(row-wise)", rotation=270, labelpad=20)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_multiposition.py` | 24 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_multiposition.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_multiposition.py` | 24 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 27 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 27 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 27 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 52 | `200` | TRUNK_GRID_SHAPE Y | TODO | `atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 52 | `0.2` | VOXEL_SIZE_MM | TODO | `atlas_binary, float(angle), 200.0, 50.0, (256, 256), 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_projection_comparison.py` | 57 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig, axes = plt.subplots(5, 4, figsize=(16, 20))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 40 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 40 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 40 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 74 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,  # Match multiposition` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 77 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm=0.2,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_results.py` | 81 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig, axes = plt.subplots(5, 4, figsize=(16, 20))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/plot_stage2_v2_comparison.py` | 74 | `30` | TRUNK_OFFSET legacy Y | TODO | `("S2-Vol-P4-dorsal-lat-r2.0", -30, "P4-dorsal-lat"),` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/recompute_projection.py` | 29 | `200` | TRUNK_GRID_SHAPE Y | TODO | `200.0,  # camera_distance` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/recompute_projection.py` | 32 | `0.2` | VOXEL_SIZE_MM | TODO | `0.2,  # voxel_size` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 43 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 43 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 43 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 44 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 44 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multi_position_test.py` | 44 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multiposition_bestview.py` | 47 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multiposition_bestview.py` | 47 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multiposition_bestview.py` | 47 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multiposition_bestview.py` | 154 | `30` | TRUNK_OFFSET legacy Y | TODO | `"best_angle": -30.0,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_multiposition_bestview.py` | 426 | `30` | TRUNK_OFFSET legacy Y | TODO | `angle_offsets = [0, 15, 30, 45, 60]` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 50 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 50 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 50 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 52 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 52 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_5_surface_aware.py` | 52 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume_xyz = volume.transpose(2, 1, 0)  # (190, 200, 104) XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_atlas.py` | 46 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_atlas.py` | 46 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage1_atlas.py` | 46 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition.py` | 52 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition.py` | 62 | `30` | TRUNK_OFFSET legacy Y | TODO | `"P4-dorsal-lat": -30.0,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition.py` | 293 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition.py` | 293 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition.py` | 293 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 42 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 52 | `30` | TRUNK_OFFSET legacy Y | TODO | `"P4-dorsal-lat": -30.0,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 73 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 73 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 73 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 149 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 593 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 593 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_multiposition_v2.py` | 593 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = volume.reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_uniform_source.py` | 54 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_uniform_source.py` | 57 | `0.2` | VOXEL_SIZE_MM | TODO | `VOXEL_SIZE_MM = 0.2` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_uniform_source.py` | 61 | `190` | TRUNK_GRID_SHAPE X | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)  # XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_uniform_source.py` | 61 | `200` | TRUNK_GRID_SHAPE Y | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)  # XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/run_stage2_uniform_source.py` | 61 | `104` | TRUNK_GRID_SHAPE Z | TODO | `ATLAS_VOLUME_SHAPE = (190, 200, 104)  # XYZ` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 24 | `19` | VOLUME_CENTER_WORLD X | TODO | `"19-point": {"n_points": 19, "description": "Center + 6 face + 12 edge centers"},` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 264 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[7:19] = 0.025` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 265 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[19:27] = 0.018` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 334 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[1:19] = 0.018` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 335 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[19:55] = 0.012` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 441 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif scheme == "19-point":` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 528 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif scheme == "19-point":` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 556 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights = np.ones(19, dtype=np.float32) / 19.0 * alpha` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/source_quadrature.py` | 609 | `0.1` | ATLAS voxel size | TODO | `extent_ratio = source_extent_mm / max(depth_mm, 0.1)` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/surface_projection.py` | 332 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/visualize_source_positions.py` | 21 | `190` | TRUNK_GRID_SHAPE X | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/visualize_source_positions.py` | 21 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/visualize_source_positions.py` | 21 | `104` | TRUNK_GRID_SHAPE Z | TODO | `volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/pilot/e1b_atlas_mcx_v2/visualize_source_positions.py` | 33 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = 0.2` |
| `FMT-SimGen/pilot/e1b_model_mismatch/analyze_mcx_peaks.py` | 30 | `30` | TRUNK_OFFSET legacy Y | TODO | `view_angles = [0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_model_mismatch/analyze_mcx_peaks.py` | 84 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/analyze_mcx_peaks.py` | 84 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_mcx_coords.py` | 24 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_mcx_coords.py` | 24 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_mcx_coords.py` | 25 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm = 0.1` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_projection.py` | 26 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_projection.py` | 26 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_projection.py` | 27 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm = 0.1` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_projection.py` | 71 | `30` | TRUNK_OFFSET legacy Y | TODO | `"angles": [-90, -60, -30, 0, 30, 60, 90],` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_projection.py` | 73 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"camera_distance_mm": 200.0,` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 24 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 35 | `0.1` | ATLAS voxel size | TODO | `sigma_init=0.1,` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 44 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"\n{'Angle':>8} | {'center_cam':>30} | {'depth (dot)':>12} | {'normal':>20}")` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 44 | `30` | TRUNK_OFFSET legacy Y | TODO | `print(f"\n{'Angle':>8} | {'center_cam':>30} | {'depth (dot)':>12} | {'normal':>20}")` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 66 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{angle:>8}° | {cam_str:>30} | {d.item():>12.3f} | {normal_str:>20} {status}"` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_psf_render.py` | 66 | `30` | TRUNK_OFFSET legacy Y | TODO | `f"{angle:>8}° | {cam_str:>30} | {d.item():>12.3f} | {normal_str:>20} {status}"` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 32 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [-90, -60, -30, 0, 30, 60, 90]:` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 41 | `30` | TRUNK_OFFSET legacy Y | TODO | `vms, norms = build_turntable_views(angles_deg=[-90, -60, -30, 0, 30, 60, 90])` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 43 | `30` | TRUNK_OFFSET legacy Y | TODO | `for i, angle in enumerate([-90, -60, -30, 0, 30, 60, 90]):` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 51 | `30` | TRUNK_OFFSET legacy Y | TODO | `print("\n旋转矩阵对比 (30°):")` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 52 | `30` | TRUNK_OFFSET legacy Y | TODO | `R_mcx = rotation_matrix_y(30)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 53 | `30` | TRUNK_OFFSET legacy Y | TODO | `vms, _ = build_turntable_views(angles_deg=[30])` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 72 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [-90, -60, -30, 0, 30, 60, 90]:` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 88 | `30` | TRUNK_OFFSET legacy Y | TODO | `vms, _ = build_turntable_views(angles_deg=[-90, -60, -30, 0, 30, 60, 90])` |
| `FMT-SimGen/pilot/e1b_model_mismatch/debug_rotation.py` | 89 | `30` | TRUNK_OFFSET legacy Y | TODO | `for i, angle in enumerate([-90, -60, -30, 0, 30, 60, 90]):` |
| `FMT-SimGen/pilot/e1b_model_mismatch/diagnose_coords.py` | 36 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/diagnose_coords.py` | 36 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/diagnose_coords.py` | 40 | `30` | TRUNK_OFFSET legacy Y | TODO | `view_angles = [0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 46 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 46 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 47 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm = 0.1` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 56 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 103 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 128 | `0.1` | ATLAS voxel size | TODO | `sigma_init=0.1,` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 142 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"\n{'Angle':>8} | {'MCX Peak (mm)':>20} | {'PSF Peak (mm)':>20} | {'Diff (mm)':>12}"` |
| `FMT-SimGen/pilot/e1b_model_mismatch/final_verification.py` | 165 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{angle:>8}° | {mcx_str:>20} | {psf_str:>20} | {diff:>10.2f} {status}")` |
| `FMT-SimGen/pilot/e1b_model_mismatch/generate_mcx_gt.py` | 123 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"camera_distance_mm": 200.0,` |
| `FMT-SimGen/pilot/e1b_model_mismatch/run_e1b.py` | 66 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm: tuple = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1b_model_mismatch/run_e1b.py` | 66 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm: tuple = (30.0, 30.0, 20.0),` |
| `FMT-SimGen/pilot/e1b_model_mismatch/run_e1b.py` | 152 | `20` | VOLUME_CENTER_WORLD Y | TODO | `max(recent_loss) + 1e-20` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_projection_match.py` | 32 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_projection_match.py` | 91 | `30` | TRUNK_OFFSET legacy Y | TODO | `angle = 30` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_psf_mcx_alignment.py` | 107 | `0.1` | ATLAS voxel size | TODO | `sigma_init=0.1,  # 小 sigma 近似点源` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_psf_mcx_alignment.py` | 123 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"\n{'Angle':>8} | {'MCX Peak (px)':>20} | {'PSF Peak (px)':>20} | {'Diff (px)':>15}"` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_psf_mcx_alignment.py` | 147 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{angle:>8}° | {mcx_str:>20} | {psf_str:>20} | {diff_str:>15} {status}")` |
| `FMT-SimGen/pilot/e1b_model_mismatch/verify_psf_mcx_alignment.py` | 160 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Angle':>8} | {'MCX (mm)':>20} | {'PSF (mm)':>20}")` |
| `FMT-SimGen/pilot/e1c_green_function_selection/compare_kernels.py` | 543 | `0.2` | VOXEL_SIZE_MM | TODO | `and hs_metrics.get("mean_peak_error_mm", 1) < 0.2` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 137 | `30` | TRUNK_OFFSET legacy Y | TODO | `if r["x_mcx_mm"] < 0 or r["x_mcx_mm"] > 30:` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 138 | `30` | TRUNK_OFFSET legacy Y | TODO | `errors.append(f"{config_id}: x_mcx_mm={r['x_mcx_mm']} out of [0, 30]")` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 139 | `30` | TRUNK_OFFSET legacy Y | TODO | `if r["y_mcx_mm"] < 0 or r["y_mcx_mm"] > 30:` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 140 | `30` | TRUNK_OFFSET legacy Y | TODO | `errors.append(f"{config_id}: y_mcx_mm={r['y_mcx_mm']} out of [0, 30]")` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 141 | `20` | VOLUME_CENTER_WORLD Y | TODO | `if r["z_mcx_mm"] < 0 or r["z_mcx_mm"] > 20:` |
| `FMT-SimGen/pilot/e1c_green_function_selection/coordinate_sanity.py` | 142 | `20` | VOLUME_CENTER_WORLD Y | TODO | `errors.append(f"{config_id}: z_mcx_mm={r['z_mcx_mm']} out of [0, 20]")` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 47 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm: float = 0.1,` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 106 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 106 | `30` | TRUNK_OFFSET legacy Y | TODO | `vol_size_mm = (30.0, 30.0, 20.0)` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 107 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm = mcx_config.get("voxel_size_mm", 0.1)` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 108 | `20` | VOLUME_CENTER_WORLD Y | TODO | `n_photons = mcx_config.get("n_sim", 20) * 1_000_000` |
| `FMT-SimGen/pilot/e1c_green_function_selection/generate_surface_gt.py` | 153 | `0.1` | ATLAS voxel size | TODO | `pixel_size_mm = mcx_config.get("pixel_size_mm", 0.1)` |
| `FMT-SimGen/pilot/e1c_green_function_selection/kernels.py` | 33 | `0.1` | ATLAS voxel size | TODO | `R_eff = 0.493 if abs(n - 1.37) < 0.1 else 0.493` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/atlas_surface_renderer_torch.py` | 219 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[7:19] = 0.025` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/atlas_surface_renderer_torch.py` | 220 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[19:27] = 0.018` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/compare_ablation.py` | 5 | `19` | VOLUME_CENTER_WORLD X | TODO | `- Sampling levels: 1-point, 7-point, 19-point, 27-point` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/generate_gt_atlas.py` | 153 | `30` | TRUNK_OFFSET legacy Y | TODO | `roi_radius = gt_config.get("roi_radius_mm", 30.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/generate_summary.py` | 46 | `0.1` | ATLAS voxel size | TODO | `mean_ic_pos / 0.1` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/generate_summary.py` | 64 | `0.1` | ATLAS voxel size | TODO | `alpha_ok = mean_ic_alpha < 0.1` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/generate_summary.py` | 69 | `0.2` | VOXEL_SIZE_MM | TODO | `elif mean_ic_pos < 0.2:` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/real_surface_ablation.py` | 71 | `30` | TRUNK_OFFSET legacy Y | TODO | `roi_radius = rs_config.get("roi_radius_mm", 30.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d.py` | 358 | `0.1` | ATLAS voxel size | TODO | `init_sigmas = tuple(max(s, 0.1) for s in init_sigmas)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d.py` | 360 | `0.1` | ATLAS voxel size | TODO | `init_alpha = max(init_alpha, 0.1)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 159 | `30` | TRUNK_OFFSET legacy Y | TODO | `source.center.data[0].clamp_(5.0, 30.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 160 | `20` | VOLUME_CENTER_WORLD Y | TODO | `source.center.data[1].clamp_(20.0, 80.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 162 | `0.1` | ATLAS voxel size | TODO | `source.log_sigmas.data.clamp_(np.log(0.1), np.log(5.0))` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 163 | `0.1` | ATLAS voxel size | TODO | `source.log_alpha.data.clamp_(np.log(0.1), np.log(10.0))` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 282 | `30` | TRUNK_OFFSET legacy Y | TODO | `source.center.data[0].clamp_(5.0, 30.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 283 | `20` | VOLUME_CENTER_WORLD Y | TODO | `source.center.data[1].clamp_(20.0, 80.0)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 285 | `0.1` | ATLAS voxel size | TODO | `source.log_axes.data.clamp_(np.log(0.1), np.log(5.0))` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 286 | `0.1` | ATLAS voxel size | TODO | `source.log_alpha.data.clamp_(np.log(0.1), np.log(10.0))` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 376 | `0.1` | ATLAS voxel size | TODO | `offset_scale = 0.1` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 383 | `0.1` | ATLAS voxel size | TODO | `init_sigmas = tuple(max(s, 0.1) for s in init_sigmas)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 388 | `0.1` | ATLAS voxel size | TODO | `init_axes = tuple(max(a, 0.1) for a in init_axes)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/run_e1d_atlas.py` | 391 | `0.1` | ATLAS voxel size | TODO | `init_alpha = max(init_alpha, 0.1)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_models.py` | 21 | `19` | VOLUME_CENTER_WORLD X | TODO | `"19-point": {"n_points": 19, "description": "Center + 6 face + 12 edge centers"},` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_models.py` | 50 | `19` | VOLUME_CENTER_WORLD X | TODO | `sampling_level: "1-point", "7-point", "19-point", or "27-point"` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_models.py` | 145 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif n_points == 19:` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_models.py` | 266 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif n_points == 19:` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 24 | `19` | VOLUME_CENTER_WORLD X | TODO | `"19-point": {"n_points": 19, "description": "Center + 6 face + 12 edge centers"},` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 264 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[7:19] = 0.025` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 265 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[19:27] = 0.018` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 334 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[1:19] = 0.018` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 335 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights[19:55] = 0.012` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 441 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif scheme == "19-point":` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 528 | `19` | VOLUME_CENTER_WORLD X | TODO | `elif scheme == "19-point":` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 556 | `19` | VOLUME_CENTER_WORLD X | TODO | `weights = np.ones(19, dtype=np.float32) / 19.0 * alpha` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/source_quadrature.py` | 609 | `0.1` | ATLAS voxel size | TODO | `extent_ratio = source_extent_mm / max(depth_mm, 0.1)` |
| `FMT-SimGen/pilot/e1d_finite_source_local_surface/surface_data.py` | 260 | `20` | VOLUME_CENTER_WORLD Y | TODO | `k_neighbors: int = 20,` |
| `FMT-SimGen/pilot/paper04b_forward/cubature_conv/cubature_schemes.py` | 100 | `200` | TRUNK_GRID_SHAPE Y | TODO | `points : (200, 3) array` |
| `FMT-SimGen/pilot/paper04b_forward/cubature_conv/cubature_schemes.py` | 101 | `200` | TRUNK_GRID_SHAPE Y | TODO | `weights : (200,) array` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_angle_sweep.py` | 45 | `200` | TRUNK_GRID_SHAPE Y | TODO | `cam = dict(camera_distance_mm=200.0, fov_mm=50.0, detector_resolution=(256, 256))` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_angle_sweep.py` | 165 | `20` | VOLUME_CENTER_WORLD Y | TODO | `np.log10(np.array(peaks) + 1e-20),` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_m4prime_grad_check.py` | 79 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_m4prime_grad_check.py` | 80 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_m4prime_grad_check.py` | 98 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_m4prime_grad_check.py` | 99 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 53 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 62 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 146 | `20` | VOLUME_CENTER_WORLD Y | TODO | `eps = 1e-20` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 178 | `0.1` | ATLAS voxel size | TODO | `if rank_corr > linear_ncc + 0.1:` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 180 | `0.1` | ATLAS voxel size | TODO | `if log_ncc > linear_ncc + 0.1:` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 265 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"camera_distance": 200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 282 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 291 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 334 | `200` | TRUNK_GRID_SHAPE Y | TODO | `shape = (200, 200, 200)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 345 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"camera_distance": 200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 359 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_fix.py` | 368 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_ncc.py` | 46 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"camera_distance_mm": 200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_projection_ncc.py` | 59 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = [-90, -60, -30, 0, 30, 60, 90, 120, 150, 180]` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold.py` | 58 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold.py` | 67 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold.py` | 85 | `40` | TRUNK_SIZE_MM Y | TODO | `print("-" * 40)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold_full.py` | 102 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold_full.py` | 111 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold_full.py` | 131 | `40` | TRUNK_SIZE_MM Y | TODO | `print("-" * 40)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold_full.py` | 135 | `40` | TRUNK_SIZE_MM Y | TODO | `print("-" * 40)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r1_threshold_full.py` | 181 | `0.1` | ATLAS voxel size | TODO | `if abs(ncc_diff) > 0.1:` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diag_r2_photon_audit.py` | 171 | `40` | TRUNK_SIZE_MM Y | TODO | `print("-" * 40)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_forward_round2.py` | 48 | `0.1` | ATLAS voxel size | TODO | `source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm, step_mm=0.1` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 113 | `0.2` | VOXEL_SIZE_MM | TODO | `passed = diff < 0.2` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 174 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fluence = np.log10(fluence + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 405 | `200` | TRUNK_GRID_SHAPE Y | TODO | `pos_3d, angle_deg, distance=200.0, fov=50.0, resolution=(256, 256)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 423 | `200` | TRUNK_GRID_SHAPE Y | TODO | `volume, angle_deg, distance=200.0, fov=50.0, resolution=(256, 256)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 464 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_params = {"distance": 200.0, "fov": 50.0, "resolution": (256, 256)}` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 475 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_proj = np.log10(proj_2d + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 500 | `20` | VOLUME_CENTER_WORLD Y | TODO | `vmin=log_proj[log_proj > -20].min(),` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 510 | `200` | TRUNK_GRID_SHAPE Y | TODO | `s=200,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 519 | `200` | TRUNK_GRID_SHAPE Y | TODO | `s=200,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 580 | `0.1` | ATLAS voxel size | TODO | `iso_val = 0.1 * fluence.max()` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_geometry.py` | 597 | `0.2` | VOXEL_SIZE_MM | TODO | `opacity=0.2,` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_surface_mesh.py` | 55 | `20` | VOLUME_CENTER_WORLD Y | TODO | `for ln, content in relevant_lines[:20]:` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/diagnose_surface_mesh.py` | 59 | `20` | VOLUME_CENTER_WORLD Y | TODO | `[f"{ln:4d}: {line.rstrip()}\n" for ln, line in relevant_lines[:20]]` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/forward_audit.py` | 76 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/forward_audit.py` | 77 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/ncc_investigation.py` | 46 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_phi = np.log10(phi_valid + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/ncc_investigation.py` | 47 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(fwd_scaled + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/ncc_investigation.py` | 107 | `20` | VOLUME_CENTER_WORLD Y | TODO | `bins = [0, 2, 5, 10, 15, 20, 30]` |
| `FMT-SimGen/pilot/paper04b_forward/diagnostics/ncc_investigation.py` | 107 | `30` | TRUNK_OFFSET legacy Y | TODO | `bins = [0, 2, 5, 10, 15, 20, 30]` |
| `FMT-SimGen/pilot/paper04b_forward/ec_atlas_surface/config.py` | 64 | `0.1` | ATLAS voxel size | TODO | `return abs(self.y_mm - self.out_of_scope_y) < 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 78 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 147 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 148 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 260 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'Metric':<25} {'All vertices':<20} {'Direct-path only':<20}")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 262 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'N':<25} {metrics_all['n_valid']:<20} {metrics_direct['n_valid']:<20}")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 263 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'NCC':<25} {metrics_all['ncc']:<20.4f} {metrics_direct['ncc']:<20.4f}")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 264 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'k':<25} {metrics_all['k']:<20.2e} {metrics_direct['k']:<20.2e}")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_1_direct_path_vertex_ncc.py` | 265 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"{'RMSE':<25} {metrics_all['rmse']:<20.4f} {metrics_direct['rmse']:<20.4f}")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_surface_space_ncc.py` | 152 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2_surface_space_ncc.py` | 153 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 39 | `20` | VOLUME_CENTER_WORLD Y | TODO | `bins = [0, 3, 6, 9, 12, 15, 20, 30, 50]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 39 | `30` | TRUNK_OFFSET legacy Y | TODO | `bins = [0, 3, 6, 9, 12, 15, 20, 30, 50]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 40 | `20` | VOLUME_CENTER_WORLD Y | TODO | `bin_labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-20", "20-30", "30-50"]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 40 | `30` | TRUNK_OFFSET legacy Y | TODO | `bin_labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-20", "20-30", "30-50"]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 52 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 53 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 123 | `20` | VOLUME_CENTER_WORLD Y | TODO | `mask = valid & (phi_mcx > 0) & (phi_closed > 0) & (dist <= 20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 129 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2b_analyze_distance.py` | 130 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2c_superficial_regime.py` | 38 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2c_superficial_regime.py` | 39 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2c_superficial_regime.py` | 98 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/d2c_superficial_regime.py` | 99 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/debug_archived.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 68 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 69 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 126 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 143 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_valid + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 144 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale_gt * forward_gt_valid + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 155 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd_init = np.log10(scale_init * forward_init_valid + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_ef_vs_m4.py` | 166 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd_rec = np.log10(scale_rec * forward_rec_valid + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 43 | `0.1` | ATLAS voxel size | TODO | `source_pos_mm, vertices, volume_labels, voxel_size_mm, step_mm=0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 87 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 88 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 166 | `0.1` | ATLAS voxel size | TODO | `elif loss_near_gt < 0.1:` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 167 | `0.1` | ATLAS voxel size | TODO | `print(f"  → Shallow near GT (variation < 0.1)")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 177 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"\n{'seed':>4} {'init_err_mm':>12} {'loss(init)':>12} {'|grad|':>12} {'status':>20}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 199 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{seed:>4} {init_err:>12.3f} {loss_val:>12.4f} {grad_norm:>12.6f} {status:>20}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 209 | `0.2` | VOXEL_SIZE_MM | TODO | `for x in np.arange(gt_pos[0] - 3, gt_pos[0] + 3.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 210 | `0.2` | VOXEL_SIZE_MM | TODO | `for y in np.arange(gt_pos[1] - 3, gt_pos[1] + 3.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc.py` | 211 | `0.2` | VOXEL_SIZE_MM | TODO | `for z in np.arange(gt_pos[2] - 3, gt_pos[2] + 3.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 30 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 31 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 94 | `0.1` | ATLAS voxel size | TODO | `elif loss_near_gt < 0.1:` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 95 | `0.1` | ATLAS voxel size | TODO | `print(f"  → Shallow near GT (variation < 0.1)")` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 105 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"\n{'seed':>4} {'init_err_mm':>12} {'loss(init)':>12} {'|grad|':>12} {'status':>20}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 127 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{seed:>4} {init_err:>12.3f} {loss_val:>12.4f} {grad_norm:>12.6f} {status:>20}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 158 | `0.2` | VOXEL_SIZE_MM | TODO | `for x in np.arange(coarse_pos[0] - 1, coarse_pos[0] + 1.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 159 | `0.2` | VOXEL_SIZE_MM | TODO | `for y in np.arange(coarse_pos[1] - 1, coarse_pos[1] + 1.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/diag_p01_abc_fast.py` | 160 | `0.2` | VOXEL_SIZE_MM | TODO | `for z in np.arange(coarse_pos[2] - 1, coarse_pos[2] + 1.1, 0.2):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ec_y10.py` | 53 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ec_y10.py` | 77 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ec_y10.py` | 78 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(phi_closed[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ed_equivariance.py` | 52 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ed_equivariance.py` | 76 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ed_equivariance.py` | 77 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(phi_closed[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ed_equivariance.py` | 114 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [0, 30, 60, 90, -30, -60, -90]:` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ef_multi_source.py` | 53 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ef_multi_source.py` | 94 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ef_multi_source.py` | 95 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/ef_multi_source.py` | 111 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/eg_optical_prior.py` | 53 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/eg_optical_prior.py` | 93 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/eg_optical_prior.py` | 94 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/eg_optical_prior.py` | 119 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m0_validate_closed_form.py` | 38 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m0_validate_closed_form.py` | 39 | `40` | TRUNK_SIZE_MM Y | TODO | `size_mm: float = 40.0,` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m0_validate_closed_form.py` | 121 | `0.1` | ATLAS voxel size | TODO | `if ratio < 0.1 or ratio > 10:` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m0_validate_closed_form.py` | 123 | `0.1` | ATLAS voxel size | TODO | `f"  FAIL: {k1}/{k2} ratio {ratio:.2f} out of range [0.1, 10]"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 44 | `0.2` | VOXEL_SIZE_MM | TODO | `VOXEL_SIZE_MM = 0.2  # MUST match MCX JSON LengthUnit` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 45 | `190` | TRUNK_GRID_SHAPE X | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 45 | `200` | TRUNK_GRID_SHAPE Y | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 45 | `104` | TRUNK_GRID_SHAPE Z | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 46 | `190` | TRUNK_GRID_SHAPE X | TODO | `VOLUME_SHAPE_XYZ = (190, 200, 104)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 46 | `200` | TRUNK_GRID_SHAPE Y | TODO | `VOLUME_SHAPE_XYZ = (190, 200, 104)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 46 | `104` | TRUNK_GRID_SHAPE Z | TODO | `VOLUME_SHAPE_XYZ = (190, 200, 104)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m1_single_view.py` | 117 | `20` | VOLUME_CENTER_WORLD Y | TODO | `for dz in range(1, 20):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m2_prime_validation.py` | 77 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m2_prime_validation.py` | 141 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m2_prime_validation.py` | 142 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m2_three_sources.py` | 38 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m2_three_sources.py` | 58 | `200` | TRUNK_GRID_SHAPE Y | TODO | `{"mua": 0.04, "mus": 200.0, "g": 0.9, "n": 1.37, "tag": 2, "name": "bone"},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_multi_view.py` | 37 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_multi_view.py` | 59 | `200` | TRUNK_GRID_SHAPE Y | TODO | `{"mua": 0.04, "mus": 200.0, "g": 0.9, "n": 1.37, "tag": 2, "name": "bone"},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_prime_multi_view.py` | 89 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_prime_multi_view.py` | 174 | `30` | TRUNK_OFFSET legacy Y | TODO | `all_angles = [0, 30, 60, 90, -90, -30, -60, 180]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_prime_multi_view.py` | 245 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{'Position':<15} {'Direct Views':<20} {'N_direct':<12} {'NCC_direct':<12} {'Pass'}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m3_prime_multi_view.py` | 257 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"{pos_name:<15} {views_str:<20} {data['n_direct_vertices']:<12} {ncc:<12.4f} {passed}"` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 71 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 155 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 156 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 183 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 184 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 215 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 225 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_inversion.py` | 226 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview.py` | 56 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview.py` | 125 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[vertex_valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview.py` | 126 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[vertex_valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview.py` | 136 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_params = {"distance": 200.0, "fov": 50.0, "resolution": 256}` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview.py` | 197 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview_fixed.py` | 37 | `30` | TRUNK_OFFSET legacy Y | TODO | `CANDIDATE_ANGLES = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview_fixed.py` | 72 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview_fixed.py` | 73 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale_i * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_multiview_fixed.py` | 123 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_surface.py` | 36 | `30` | TRUNK_OFFSET legacy Y | TODO | `CANDIDATE_ANGLES = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_surface.py` | 37 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_PARAMS = {"distance": 200.0, "fov": 50.0, "resolution": (256, 256)}` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_surface.py` | 101 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_surface.py` | 102 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m4_prime_surface.py` | 113 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 51 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 99 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(measurement[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 100 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 123 | `20` | VOLUME_CENTER_WORLD Y | TODO | `bounds=[(-20, 20), (-20, 20), (-20, 20), (0.01, 1.0), (0.5, 20.0)],` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 124 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 176 | `0.2` | VOXEL_SIZE_MM | TODO | `init_mua = gt_mua * (1 + np.random.randn() * 0.2)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m5_prime_joint.py` | 177 | `0.2` | VOXEL_SIZE_MM | TODO | `init_musp = gt_musp * (1 + np.random.randn() * 0.2)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m6_teaser.py` | 51 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m6_teaser.py` | 75 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/m6_teaser.py` | 76 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(phi_closed[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/mcx_runner.py` | 27 | `0.2` | VOXEL_SIZE_MM | TODO | `VOXEL_SIZE_MM = 0.2` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/mcx_runner.py` | 29 | `190` | TRUNK_GRID_SHAPE X | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/mcx_runner.py` | 29 | `200` | TRUNK_GRID_SHAPE Y | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/mcx_runner.py` | 29 | `104` | TRUNK_GRID_SHAPE Z | TODO | `VOLUME_SHAPE_ZYX = (104, 200, 190)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/plot_comparison.py` | 24 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DISTANCE_MM = 200.0` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/plot_comparison.py` | 95 | `0.1` | ATLAS voxel size | TODO | `ax.scatter(my_green[valid][::5], archived_mcx[valid][::5], alpha=0.1, s=1)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/precompute_p5.py` | 36 | `0.1` | ATLAS voxel size | TODO | `source_pos_mm, vertices, volume_labels, voxel_size_mm, step_mm=0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/run_d2_1_all_positions.py` | 69 | `0.1` | ATLAS voxel size | TODO | `step_mm = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/run_d2_1_all_positions.py` | 117 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_mcx = np.log10(mcx_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/run_d2_1_all_positions.py` | 118 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_closed = np.log10(closed_vals + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/source_placement_preflight.py` | 85 | `20` | VOLUME_CENTER_WORLD Y | TODO | `for dz in range(-20, 21):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/source_placement_preflight.py` | 86 | `20` | VOLUME_CENTER_WORLD Y | TODO | `for dy in range(-20, 21):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/source_placement_preflight.py` | 87 | `20` | VOLUME_CENTER_WORLD Y | TODO | `for dx in range(-20, 21):` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/source_placement_preflight.py` | 176 | `30` | TRUNK_OFFSET legacy Y | TODO | `"P4-dorsal-lat": {"pos_mm": [-6.3, 2.4, 5.8], "best_angle": -30},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/verify_init_effect.py` | 61 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_meas = np.log10(phi_mcx[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/verify_init_effect.py` | 62 | `20` | VOLUME_CENTER_WORLD Y | TODO | `log_fwd = np.log10(scale * forward[valid] + 1e-20)` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/verify_init_effect.py` | 97 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/mvp_pipeline/verify_init_effect.py` | 129 | `200` | TRUNK_GRID_SHAPE Y | TODO | `options={"maxiter": 200},` |
| `FMT-SimGen/pilot/paper04b_forward/shared/config.py` | 40 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2` |
| `FMT-SimGen/pilot/paper04b_forward/shared/config.py` | 42 | `40` | TRUNK_SIZE_MM Y | TODO | `fov_mm: float = 40.0` |
| `FMT-SimGen/pilot/paper04b_forward/shared/direct_path.py` | 36 | `0.1` | ATLAS voxel size | TODO | `DEFAULT_STEP_MM = 0.1` |
| `FMT-SimGen/pilot/paper04b_forward/shared/direct_path.py` | 465 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_params.get("distance", 200.0),` |
| `FMT-SimGen/pilot/paper04b_forward/shared/direct_path.py` | 500 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [0, 30, 60, 90, -90, 180]:` |
| `FMT-SimGen/pilot/paper04b_forward/shared/direct_path.py` | 511 | `30` | TRUNK_OFFSET legacy Y | TODO | `for angle in [0, 30, 60, 90, -90, 180]:` |
| `FMT-SimGen/pilot/paper04b_forward/shared/green_surface_projection.py` | 245 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: float = 0.2,` |
| `FMT-SimGen/pilot/paper04b_forward/shared/metrics.py` | 23 | `20` | VOLUME_CENTER_WORLD Y | TODO | `def ncc_log(a: np.ndarray, b: np.ndarray, eps: float = 1e-20) -> float:` |
| `FMT-SimGen/pilot/paper04b_forward/shared/metrics.py` | 45 | `20` | VOLUME_CENTER_WORLD Y | TODO | `meas: np.ndarray, forward: np.ndarray, eps: float = 1e-20` |
| `FMT-SimGen/pilot/paper04b_forward/shared/preflight.py` | 74 | `0.1` | ATLAS voxel size | TODO | `tolerance_mm: float = 0.1,` |
| `FMT-SimGen/pilot/paper04b_forward/shared/surface_coords.py` | 65 | `30` | TRUNK_OFFSET legacy Y | TODO | `best_angle=30,` |
| `FMT-SimGen/pilot/visualization/plot_e0_2d_comparison.py` | 65 | `200` | TRUNK_GRID_SHAPE Y | TODO | `image_size = 200` |
| `FMT-SimGen/pilot/visualization/plot_e0_2d_comparison.py` | 66 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fov_mm = 20.0` |
| `FMT-SimGen/pilot/visualization/plot_e1c_kernel_selection.py` | 172 | `0.1` | ATLAS voxel size | TODO | `if ncc > 0.99 and abs(fwhm - 1.0) < 0.1:` |
| `FMT-SimGen/pilot/visualization/plot_e1c_kernel_selection.py` | 192 | `0.2` | VOXEL_SIZE_MM | TODO | `colWidths=[0.4, 0.2, 0.2, 0.2],` |
| `FMT-SimGen/pilot/visualization/plot_e1c_kernel_selection.py` | 203 | `20` | VOLUME_CENTER_WORLD Y | TODO | `ax.set_title("Panel C: Summary Metrics", pad=20)` |
| `FMT-SimGen/pilot/visualization/plot_e1d_atlas_vs_flat_2d.py` | 215 | `0.1` | ATLAS voxel size | TODO | `0.1,` |
| `FMT-SimGen/pilot/visualization/plot_e1d_quadrature.py` | 85 | `30` | TRUNK_OFFSET legacy Y | TODO | `label=scheme if info["n"] <= 30 else None,` |
| `FMT-SimGen/pilot/visualization/plot_e1d_quadrature.py` | 170 | `0.2` | VOXEL_SIZE_MM | TODO | `colWidths=[0.25, 0.15, 0.2, 0.2, 0.2],` |
| `FMT-SimGen/pilot/visualization/plot_e1d_quadrature.py` | 185 | `20` | VOLUME_CENTER_WORLD Y | TODO | `ax.set_title("Panel B: Quadrature Scheme Comparison", pad=20)` |
| `FMT-SimGen/pilot/visualization/plot_e1d_quadrature.py` | 192 | `0.1` | ATLAS voxel size | TODO | `-0.1,` |
| `FMT-SimGen/pilot/visualization/run_all_v2.py` | 79 | `200` | TRUNK_GRID_SHAPE Y | TODO | `print(f"    Error: {result.stderr[:200]}")` |
| `FMT-SimGen/pilot/visualization/run_all_v2.py` | 154 | `30` | TRUNK_OFFSET legacy Y | TODO | `print("This may take 30+ minutes...")` |
| `FMT-SimGen/scripts/04_stratified_split.py` | 6 | `20` | VOLUME_CENTER_WORLD Y | TODO | `- Total: 1000 samples, 80% train / 20% val` |
| `FMT-SimGen/scripts/04_stratified_split.py` | 13 | `200` | TRUNK_GRID_SHAPE Y | TODO | `- train/splits/val.txt (200 lines, sample_XXXX only)` |
| `FMT-SimGen/scripts/04_stratified_split.py` | 138 | `200` | TRUNK_GRID_SHAPE Y | TODO | `"val": 200,` |
| `FMT-SimGen/scripts/05_verify_frame_consistency.py` | 43 | `30` | TRUNK_OFFSET legacy Y | TODO | `assert nodes.min() >= -30.0, (` |
| `FMT-SimGen/scripts/_archive/02b_generate_more_samples.py` | 3 | `200` | TRUNK_GRID_SHAPE Y | TODO | `Generate additional dataset samples (up to 200 total).` |
| `FMT-SimGen/scripts/_archive/02b_generate_more_samples.py` | 7 | `200` | TRUNK_GRID_SHAPE Y | TODO | `python scripts/02b_generate_more_samples.py --num_samples 200` |
| `FMT-SimGen/scripts/_archive/02b_generate_more_samples.py` | 32 | `200` | TRUNK_GRID_SHAPE Y | TODO | `target_total = 200` |
| `FMT-SimGen/scripts/_archive/02b_generate_more_samples.py` | 52 | `0.2` | VOXEL_SIZE_MM | TODO | `spacing = 0.2` |
| `FMT-SimGen/scripts/_archive/02b_generate_more_samples.py` | 96 | `20` | VOLUME_CENTER_WORLD Y | TODO | `if (i - start_idx + 1) % 20 == 0:` |
| `FMT-SimGen/scripts/_archive/step4m_mcx_projection.py` | 105 | `30` | TRUNK_OFFSET legacy Y | TODO | `origin = np.array([0.0, 30.0, 0.0])` |
| `FMT-SimGen/scripts/_archive/step4m_mcx_projection.py` | 109 | `0.2` | VOXEL_SIZE_MM | TODO | `proj = camera.project_volume(fluence, angle, 0.2, origin)` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 27 | `34` | TRUNK_OFFSET_ATLAS_MM Y | TODO | `34: ("TRUNK_OFFSET_ATLAS_MM[1]", "Y offset"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 28 | `34` | TRUNK_OFFSET_ATLAS_MM Y | TODO | `30: ("TRUNK_OFFSET legacy (was 30, now 34)", "Y offset"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 28 | `30` | TRUNK_OFFSET legacy Y | TODO | `30: ("TRUNK_OFFSET legacy (was 30, now 34)", "Y offset"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 29 | `19` | VOLUME_CENTER_WORLD X | TODO | `19: ("VOLUME_CENTER_WORLD[0] or TRUNK_SIZE_MM[0]/2", "X half-size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 30 | `20` | VOLUME_CENTER_WORLD Y | TODO | `20: ("VOLUME_CENTER_WORLD[1] or TRUNK_SIZE_MM[1]/2", "Y half-size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 31 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `10.4: ("VOLUME_CENTER_WORLD[2] or TRUNK_SIZE_MM[2]/2", "Z half-size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 32 | `38` | TRUNK_SIZE_MM X | TODO | `38: ("TRUNK_SIZE_MM[0]", "X size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 33 | `40` | TRUNK_SIZE_MM Y | TODO | `40: ("TRUNK_SIZE_MM[1]", "Y size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 34 | `20` | VOLUME_CENTER_WORLD Y | TODO | `20.8: ("TRUNK_SIZE_MM[2]", "Z size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 34 | `20.8` | TRUNK_SIZE_MM Z | TODO | `20.8: ("TRUNK_SIZE_MM[2]", "Z size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 35 | `190` | TRUNK_GRID_SHAPE X | TODO | `190: ("TRUNK_GRID_SHAPE[0]", "X voxels"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 36 | `200` | TRUNK_GRID_SHAPE Y | TODO | `200: ("TRUNK_GRID_SHAPE[1]", "Y voxels"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 37 | `104` | TRUNK_GRID_SHAPE Z | TODO | `104: ("TRUNK_GRID_SHAPE[2]", "Z voxels"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 38 | `0.2` | VOXEL_SIZE_MM | TODO | `0.2: ("VOXEL_SIZE_MM", "voxel size"),` |
| `FMT-SimGen/scripts/check_doc_consistency.py` | 39 | `0.1` | ATLAS voxel size | TODO | `0.1: ("ATLAS voxel size", "atlas voxel size"),` |
| `FMT-SimGen/scripts/check_frame_literals.py` | 172 | `30` | TRUNK_OFFSET legacy Y | TODO | `print("\nFirst 30 hits:")` |
| `FMT-SimGen/scripts/check_frame_literals.py` | 173 | `30` | TRUNK_OFFSET legacy Y | TODO | `for hit in unique_hits[:30]:` |
| `FMT-SimGen/scripts/continue_uniform_1000_v2.py` | 85 | `20` | VOLUME_CENTER_WORLD Y | TODO | `min_gt_nonzero_count = quality_filter.get("min_gt_nonzero_count", 20)` |
| `FMT-SimGen/scripts/diagnose_dataset.py` | 77 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"Pairs with distance < 20mm: {np.sum(all_distances < 20)}")` |
| `FMT-SimGen/scripts/diagnose_dataset.py` | 252 | `30` | TRUNK_OFFSET legacy Y | TODO | `print("  Trunk region: Y in [30, 70], Dorsal Z>15 or Lateral X<8 or X>28")` |
| `FMT-SimGen/scripts/diagnose_dataset.py` | 254 | `30` | TRUNK_OFFSET legacy Y | TODO | `trunk_mask = (mesh_nodes[:, 1] >= 30) & (mesh_nodes[:, 1] <= 70)` |
| `FMT-SimGen/scripts/export_paraview.py` | 76 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_spacing = 0.2` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 28 | `190` | TRUNK_GRID_SHAPE X | TODO | `vol_zyx = raw.reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 28 | `200` | TRUNK_GRID_SHAPE Y | TODO | `vol_zyx = raw.reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 28 | `104` | TRUNK_GRID_SHAPE Z | TODO | `vol_zyx = raw.reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 29 | `190` | TRUNK_GRID_SHAPE X | TODO | `vol_xyz = vol_zyx.transpose(2, 1, 0)  # [X=190, Y=200, Z=104]` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 29 | `200` | TRUNK_GRID_SHAPE Y | TODO | `vol_xyz = vol_zyx.transpose(2, 1, 0)  # [X=190, Y=200, Z=104]` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 29 | `104` | TRUNK_GRID_SHAPE Z | TODO | `vol_xyz = vol_zyx.transpose(2, 1, 0)  # [X=190, Y=200, Z=104]` |
| `FMT-SimGen/scripts/generate_surface_depth.py` | 42 | `30` | TRUNK_OFFSET legacy Y | TODO | `parser.add_argument("--angles", default="-90,-60,-30,0,30,60,90")` |
| `FMT-SimGen/scripts/generate_surface_depth_body.py` | 40 | `0.1` | ATLAS voxel size | TODO | `voxel_size_mm=0.1,   # atlas is 0.1mm` |
| `FMT-SimGen/scripts/geometry_health_check.py` | 56 | `30` | TRUNK_OFFSET legacy Y | TODO | `parser.add_argument("--trunk_offset", type=float, default=30.0, help="Trunk offset Y (mm)")` |
| `FMT-SimGen/scripts/geometry_health_check.py` | 71 | `38` | TRUNK_SIZE_MM X | TODO | `query_y = (46.38, 67.90)` |
| `FMT-SimGen/scripts/mesh_outer_hull_diagnostic.py` | 72 | `38` | TRUNK_SIZE_MM X | TODO | `ax.set_xlim(0, 38)` |
| `FMT-SimGen/scripts/mesh_outer_hull_diagnostic.py` | 73 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.set_ylim(0, 40)` |
| `FMT-SimGen/scripts/mesh_outer_hull_diagnostic.py` | 83 | `19` | VOLUME_CENTER_WORLD X | TODO | `circle = plt.Circle((19, 20), 17, fill=False, color="black", linewidth=2, linestyle="--")` |
| `FMT-SimGen/scripts/mesh_outer_hull_diagnostic.py` | 83 | `20` | VOLUME_CENTER_WORLD Y | TODO | `circle = plt.Circle((19, 20), 17, fill=False, color="black", linewidth=2, linestyle="--")` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 84 | `190` | TRUNK_GRID_SHAPE X | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 84 | `200` | TRUNK_GRID_SHAPE Y | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 84 | `104` | TRUNK_GRID_SHAPE Z | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 136 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 141 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = camera.angles  # [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 194 | `40` | TRUNK_SIZE_MM Y | TODO | `u_mm = (u_px[in_fov] / 256.0) * 80.0 - 40.0` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 195 | `40` | TRUNK_SIZE_MM Y | TODO | `v_mm = (v_px[in_fov] / 256.0) * 80.0 - 40.0` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 208 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 218 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 228 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 238 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 248 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 292 | `40` | TRUNK_SIZE_MM Y | TODO | `u_mm = (u_px[in_fov] / 256.0) * 80.0 - 40.0` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 293 | `40` | TRUNK_SIZE_MM Y | TODO | `v_mm = (v_px[in_fov] / 256.0) * 80.0 - 40.0` |
| `FMT-SimGen/scripts/plot_visibility_fluence_overlay.py` | 323 | `40` | TRUNK_SIZE_MM Y | TODO | `extent=[-40, 40, -40, 40])` |
| `FMT-SimGen/scripts/plot_visibility_per_angle.py` | 30 | `190` | TRUNK_GRID_SHAPE X | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_per_angle.py` | 30 | `200` | TRUNK_GRID_SHAPE Y | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_per_angle.py` | 30 | `104` | TRUNK_GRID_SHAPE Z | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)` |
| `FMT-SimGen/scripts/plot_visibility_per_angle.py` | 64 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/scripts/plot_visibility_per_angle.py` | 87 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig = plt.figure(figsize=(20, 10))` |
| `FMT-SimGen/scripts/preflight_regen.py` | 219 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = ["-90", "-60", "-30", "0", "30", "60", "90"]` |
| `FMT-SimGen/scripts/preflight_regen.py` | 233 | `0.2` | VOXEL_SIZE_MM | TODO | `vs = 0.2` |
| `FMT-SimGen/scripts/preflight_regen.py` | 258 | `200` | TRUNK_GRID_SHAPE Y | TODO | `camera_distance_mm=200.0,` |
| `FMT-SimGen/scripts/preflight_regen.py` | 280 | `0.2` | VOXEL_SIZE_MM | TODO | `depth_tolerance_mm=0.2,` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 60 | `190` | TRUNK_GRID_SHAPE X | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)  # XYZ uint8` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 60 | `200` | TRUNK_GRID_SHAPE Y | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)  # XYZ uint8` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 60 | `104` | TRUNK_GRID_SHAPE Z | TODO | `mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)  # XYZ uint8` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 118 | `20` | VOLUME_CENTER_WORLD Y | TODO | `y_lim = (0, 40); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 118 | `40` | TRUNK_SIZE_MM Y | TODO | `y_lim = (0, 40); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 118 | `20.8` | TRUNK_SIZE_MM Z | TODO | `y_lim = (0, 40); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 120 | `40` | TRUNK_SIZE_MM Y | TODO | `y_coords = np.arange(0, 40, VOXEL_SIZE_MM)[:vol.shape[0]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 121 | `20` | VOLUME_CENTER_WORLD Y | TODO | `z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 121 | `20.8` | TRUNK_SIZE_MM Z | TODO | `z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 131 | `20` | VOLUME_CENTER_WORLD Y | TODO | `x_lim = (0, 38); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 131 | `38` | TRUNK_SIZE_MM X | TODO | `x_lim = (0, 38); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 131 | `20.8` | TRUNK_SIZE_MM Z | TODO | `x_lim = (0, 38); z_lim = (0, 20.8)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 133 | `38` | TRUNK_SIZE_MM X | TODO | `x_coords = np.arange(0, 38, VOXEL_SIZE_MM)[:vol.shape[0]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 134 | `20` | VOLUME_CENTER_WORLD Y | TODO | `z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 134 | `20.8` | TRUNK_SIZE_MM Z | TODO | `z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 143 | `38` | TRUNK_SIZE_MM X | TODO | `x_lim = (0, 38); y_lim = (0, 40)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 143 | `40` | TRUNK_SIZE_MM Y | TODO | `x_lim = (0, 38); y_lim = (0, 40)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 145 | `38` | TRUNK_SIZE_MM X | TODO | `x_coords = np.arange(0, 38, VOXEL_SIZE_MM)[:vol.shape[0]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 146 | `40` | TRUNK_SIZE_MM Y | TODO | `y_coords = np.arange(0, 40, VOXEL_SIZE_MM)[:vol.shape[1]]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 164 | `20` | VOLUME_CENTER_WORLD Y | TODO | `("Y", 20.0, "Coronal (Y=20mm)"),` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 165 | `19` | VOLUME_CENTER_WORLD X | TODO | `("X", 19.0, "Sagittal (X=19mm)"),` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 174 | `40` | TRUNK_SIZE_MM Y | TODO | `axes[i].set_xlim(0, 40)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 186 | `40` | TRUNK_SIZE_MM Y | TODO | `ax2.set_xlim(0, 40); ax2.set_ylim(0, 21)` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 207 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = VOXEL_SIZE_MM  # 0.2 mm` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 315 | `30` | TRUNK_OFFSET legacy Y | TODO | `angles = camera.angles  # [-90, -60, -30, 0, 30, 60, 90]` |
| `FMT-SimGen/scripts/qa_pilot_sample.py` | 393 | `0.1` | ATLAS voxel size | TODO | `if source_val < max_val * 0.1:` |
| `FMT-SimGen/scripts/run_step0a_atlas.py` | 121 | `30` | TRUNK_OFFSET legacy Y | TODO | `max_val = int(volume.max()) + 1 if volume.max() < 100 else 30` |
| `FMT-SimGen/scripts/run_step0a_atlas.py` | 220 | `19` | VOLUME_CENTER_WORLD X | TODO | `exclude_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21],` |
| `FMT-SimGen/scripts/run_step0a_atlas.py` | 220 | `20` | VOLUME_CENTER_WORLD Y | TODO | `exclude_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21],` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 6 | `190` | TRUNK_GRID_SHAPE X | TODO | `trunk_volume: [X=190, Y=200, Z=104] uint8 labels` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 6 | `200` | TRUNK_GRID_SHAPE Y | TODO | `trunk_volume: [X=190, Y=200, Z=104] uint8 labels` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 6 | `104` | TRUNK_GRID_SHAPE Z | TODO | `trunk_volume: [X=190, Y=200, Z=104] uint8 labels` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 7 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size_mm: 0.2` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 8 | `34` | TRUNK_OFFSET_ATLAS_MM Y | TODO | `trunk_offset_atlas_mm: [0, 34, 0]` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 32 | `0.1` | ATLAS voxel size | TODO | `VS_ATLAS = 0.1   # mm, atlas voxel size` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 33 | `0.2` | VOXEL_SIZE_MM | TODO | `VS_TRUNK = 0.2   # mm, trunk/MCX voxel size (2× downsample)` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 74 | `34` | TRUNK_OFFSET_ATLAS_MM Y | TODO | `offset = TRUNK_OFFSET_ATLAS_MM  # [0, 34, 0] in atlas mm` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 82 | `190` | TRUNK_GRID_SHAPE X | TODO | `nx_out = int(np.round(TRUNK_SIZE_MM[0] / VS_TRUNK))   # 190` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 83 | `200` | TRUNK_GRID_SHAPE Y | TODO | `ny_out = int(np.round(TRUNK_SIZE_MM[1] / VS_TRUNK))   # 200` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 84 | `104` | TRUNK_GRID_SHAPE Z | TODO | `nz_out = int(np.round(TRUNK_SIZE_MM[2] / VS_TRUNK))   # 104` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 140 | `190` | TRUNK_GRID_SHAPE X | TODO | `assert trunk_zyx.shape == (104, 200, 190), f"ZYX shape mismatch: {trunk_zyx.shape}"` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 140 | `200` | TRUNK_GRID_SHAPE Y | TODO | `assert trunk_zyx.shape == (104, 200, 190), f"ZYX shape mismatch: {trunk_zyx.shape}"` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 140 | `104` | TRUNK_GRID_SHAPE Z | TODO | `assert trunk_zyx.shape == (104, 200, 190), f"ZYX shape mismatch: {trunk_zyx.shape}"` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 156 | `190` | TRUNK_GRID_SHAPE X | TODO | `old = np.fromfile(old_bin, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 156 | `200` | TRUNK_GRID_SHAPE Y | TODO | `old = np.fromfile(old_bin, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/step0a_build_trunk_canonical.py` | 156 | `104` | TRUNK_GRID_SHAPE Z | TODO | `old = np.fromfile(old_bin, dtype=np.uint8).reshape((104, 200, 190))` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 199 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size = float(trunk_data["voxel_size_mm"])  # 0.2` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 203 | `190` | TRUNK_GRID_SHAPE X | TODO | `logger.info(f"Expected shape: (190, 200, 104)")` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 203 | `200` | TRUNK_GRID_SHAPE Y | TODO | `logger.info(f"Expected shape: (190, 200, 104)")` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 203 | `104` | TRUNK_GRID_SHAPE Z | TODO | `logger.info(f"Expected shape: (190, 200, 104)")` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 204 | `190` | TRUNK_GRID_SHAPE X | TODO | `assert trunk_volume.shape == (190, 200, 104), (` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 204 | `200` | TRUNK_GRID_SHAPE Y | TODO | `assert trunk_volume.shape == (190, 200, 104), (` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 204 | `104` | TRUNK_GRID_SHAPE Z | TODO | `assert trunk_volume.shape == (190, 200, 104), (` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 205 | `190` | TRUNK_GRID_SHAPE X | TODO | `f"Unexpected trunk_volume shape {trunk_volume.shape}, expected (190, 200, 104)"` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 205 | `200` | TRUNK_GRID_SHAPE Y | TODO | `f"Unexpected trunk_volume shape {trunk_volume.shape}, expected (190, 200, 104)"` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 205 | `104` | TRUNK_GRID_SHAPE Z | TODO | `f"Unexpected trunk_volume shape {trunk_volume.shape}, expected (190, 200, 104)"` |
| `FMT-SimGen/scripts/step0b_generate_mesh.py` | 223 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_size=voxel_size,  # 0.2 mm (trunk voxel size)` |
| `FMT-SimGen/scripts/step0d_voxel_grid.py` | 32 | `0.1` | ATLAS voxel size | TODO | `spacing: float = 0.1,` |
| `FMT-SimGen/scripts/step0d_voxel_grid.py` | 112 | `0.1` | ATLAS voxel size | TODO | `spacing = 0.1` |
| `FMT-SimGen/scripts/step0f_mcx_volume.py` | 131 | `40` | TRUNK_SIZE_MM Y | TODO | `y_end = y_start + int(np.round(40.0 / voxel_size))  # TRUNK_SIZE_MM[1] = 40mm` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 18 | `20` | VOLUME_CENTER_WORLD Y | TODO | `assert_focus_in_trunk([20.0, 26.0, 10.0], tol_mm=1.0)` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 26 | `20` | VOLUME_CENTER_WORLD Y | TODO | `assert_focus_in_trunk([20.0, -5.0, 18.0], tol_mm=1.0)` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 41 | `19` | VOLUME_CENTER_WORLD X | TODO | `assert_vcw([19.0, 0.0, 10.4], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 41 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `assert_vcw([19.0, 0.0, 10.4], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 49 | `19` | VOLUME_CENTER_WORLD X | TODO | `assert_vcw([19.0, 20.0, 10.4], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 49 | `20` | VOLUME_CENTER_WORLD Y | TODO | `assert_vcw([19.0, 20.0, 10.4], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 49 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `assert_vcw([19.0, 20.0, 10.4], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 64 | `190` | TRUNK_GRID_SHAPE X | TODO | `assert_mcx_volume_shape([190, 200, 100], "test")  # Z=100 instead of 104` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 64 | `200` | TRUNK_GRID_SHAPE Y | TODO | `assert_mcx_volume_shape([190, 200, 100], "test")  # Z=100 instead of 104` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 64 | `104` | TRUNK_GRID_SHAPE Z | TODO | `assert_mcx_volume_shape([190, 200, 100], "test")  # Z=100 instead of 104` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 72 | `190` | TRUNK_GRID_SHAPE X | TODO | `assert_mcx_volume_shape([190, 200, 104], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 72 | `200` | TRUNK_GRID_SHAPE Y | TODO | `assert_mcx_volume_shape([190, 200, 104], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 72 | `104` | TRUNK_GRID_SHAPE Z | TODO | `assert_mcx_volume_shape([190, 200, 104], "test")` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 88 | `190` | TRUNK_GRID_SHAPE X | TODO | `fake_volume = np.zeros((190, 200, 104), dtype=np.float32)` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 88 | `200` | TRUNK_GRID_SHAPE Y | TODO | `fake_volume = np.zeros((190, 200, 104), dtype=np.float32)` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 88 | `104` | TRUNK_GRID_SHAPE Z | TODO | `fake_volume = np.zeros((190, 200, 104), dtype=np.float32)` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 92 | `19` | VOLUME_CENTER_WORLD X | TODO | `camera.project_volume(fake_volume, volume_center_world=[19.0, 0.0, 10.4])` |
| `FMT-SimGen/scripts/test_h3_assertions.py` | 92 | `10.4` | VOLUME_CENTER_WORLD Z | TODO | `camera.project_volume(fake_volume, volume_center_world=[19.0, 0.0, 10.4])` |
| `FMT-SimGen/scripts/validate_dataset.py` | 263 | `0.1` | ATLAS voxel size | TODO | `"WEAK": [],      # gt_max < 0.1` |
| `FMT-SimGen/scripts/validate_dataset.py` | 331 | `0.1` | ATLAS voxel size | TODO | `elif gt_max < 0.1:` |
| `FMT-SimGen/scripts/validate_dataset.py` | 366 | `0.1` | ATLAS voxel size | TODO | `"WEAK (gt_max < 0.1)": anomaly_counts["WEAK"],` |
| `FMT-SimGen/scripts/validate_dataset.py` | 477 | `0.1` | ATLAS voxel size | TODO | `usable = [s for s in all_stats if s["gt_max"] >= 0.1]` |
| `FMT-SimGen/scripts/validate_dataset.py` | 480 | `0.1` | ATLAS voxel size | TODO | `print(f"  Usable samples (gt_max >= 0.1): {len(usable)}/{len(all_stats)} ({len(usable)/len(all_stats)*100:.1f}%)")` |
| `FMT-SimGen/scripts/validate_dataset.py` | 613 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.hist(gt_max_vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)` |
| `FMT-SimGen/scripts/validate_dataset.py` | 614 | `0.1` | ATLAS voxel size | TODO | `ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1.2, label="threshold=0.1")` |
| `FMT-SimGen/scripts/validate_dataset.py` | 624 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.hist(b_max_vals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5)` |
| `FMT-SimGen/scripts/validate_dataset.py` | 633 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.hist(gt_nonzero_frac_vals, bins=40, color="#C44E52", edgecolor="white", linewidth=0.5)` |
| `FMT-SimGen/scripts/validate_dataset.py` | 662 | `200` | TRUNK_GRID_SHAPE Y | TODO | `ha="center", va="center", color="black" if heatmap_data[i, j] < 200 else "white",` |
| `FMT-SimGen/scripts/validate_dataset.py` | 928 | `0.2` | VOXEL_SIZE_MM | TODO | `voxel_spacing = 0.2  # mm` |
| `FMT-SimGen/scripts/verify_dual_channel.py` | 251 | `0.2` | VOXEL_SIZE_MM | TODO | `k_de = max(10, int(len(b) * 0.2))` |
| `FMT-SimGen/scripts/verify_dual_channel.py` | 309 | `30` | TRUNK_OFFSET legacy Y | TODO | `"mcx_align_pass": bool(mcx_gt_dist < 30.0),` |
| `FMT-SimGen/scripts/verify_dual_channel.py` | 594 | `40` | TRUNK_SIZE_MM Y | TODO | `print(f"\n{'=' * 40}")` |
| `FMT-SimGen/scripts/verify_dual_channel.py` | 596 | `40` | TRUNK_SIZE_MM Y | TODO | `print(f"{'=' * 40}")` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 49 | `20` | VOLUME_CENTER_WORLD Y | TODO | `ax.set_title(f"Trunk-only Mesh (unified frame)\n{len(nodes)} nodes | Red box = MCX volume (38×40×20.8mm)",` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 49 | `38` | TRUNK_SIZE_MM X | TODO | `ax.set_title(f"Trunk-only Mesh (unified frame)\n{len(nodes)} nodes | Red box = MCX volume (38×40×20.8mm)",` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 49 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.set_title(f"Trunk-only Mesh (unified frame)\n{len(nodes)} nodes | Red box = MCX volume (38×40×20.8mm)",` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 67 | `38` | TRUNK_SIZE_MM X | TODO | `ax.axvline(0, color="red", lw=1); ax.axvline(38, color="red", lw=1)` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 68 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.axhline(0, color="red", lw=1); ax.axhline(40, color="red", lw=1)` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 69 | `19` | VOLUME_CENTER_WORLD X | TODO | `ax.text(19, 42, "MCX bbox", color="red", fontsize=8, ha="center")` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 74 | `40` | TRUNK_SIZE_MM Y | TODO | `fig.suptitle(f"Trunk Mesh Orthographic Projections (crop Y=[30,70]atlas → trunk-local Y=[0,40])", fontsize=12)` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 74 | `30` | TRUNK_OFFSET legacy Y | TODO | `fig.suptitle(f"Trunk Mesh Orthographic Projections (crop Y=[30,70]atlas → trunk-local Y=[0,40])", fontsize=12)` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 82 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.hist(nodes[:, 1], bins=40, edgecolor="black", alpha=0.7, color="steelblue")` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 84 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.axvline(40, color="orange", lw=2, label="Y=40 (MCX bbox top)")` |
| `FMT-SimGen/scripts/visualize_mesh_only.py` | 87 | `30` | TRUNK_OFFSET legacy Y | TODO | `ax.set_title(f"Node Y distribution — {len(nodes)} nodes | atlas Y=[30, 70]mm")` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 29 | `0.2` | VOXEL_SIZE_MM | TODO | `ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=0.5, c="gray", alpha=0.2)` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 71 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig, axes = plt.subplots(2, 4, figsize=(20, 10))` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 97 | `0.1` | ATLAS voxel size | TODO | `ax.text(0.1, 0.5, foci_str, fontsize=7, transform=ax.transAxes, va="center")` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 113 | `20` | VOLUME_CENTER_WORLD Y | TODO | `mcx_bbox_max = np.array([38, 40, 20.8])` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 113 | `38` | TRUNK_SIZE_MM X | TODO | `mcx_bbox_max = np.array([38, 40, 20.8])` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 113 | `40` | TRUNK_SIZE_MM Y | TODO | `mcx_bbox_max = np.array([38, 40, 20.8])` |
| `FMT-SimGen/scripts/visualize_new_data.py` | 113 | `20.8` | TRUNK_SIZE_MM Z | TODO | `mcx_bbox_max = np.array([38, 40, 20.8])` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 36 | `20` | VOLUME_CENTER_WORLD Y | TODO | `mcx_bbox_max = np.array([38.0, 40.0, 20.8])` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 36 | `38` | TRUNK_SIZE_MM X | TODO | `mcx_bbox_max = np.array([38.0, 40.0, 20.8])` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 36 | `40` | TRUNK_SIZE_MM Y | TODO | `mcx_bbox_max = np.array([38.0, 40.0, 20.8])` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 36 | `20.8` | TRUNK_SIZE_MM Z | TODO | `mcx_bbox_max = np.array([38.0, 40.0, 20.8])` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 39 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 39 | `38` | TRUNK_SIZE_MM X | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 39 | `40` | TRUNK_SIZE_MM Y | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 39 | `20.8` | TRUNK_SIZE_MM Z | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 61 | `200` | TRUNK_GRID_SHAPE Y | TODO | `CAMERA_DIST = 200.0  # mm, matches mcx_projection.py` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 84 | `19` | VOLUME_CENTER_WORLD X | TODO | `cx, cy = 19.0, 20.0  # approximate center of MCX volume` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 84 | `20` | VOLUME_CENTER_WORLD Y | TODO | `cx, cy = 19.0, 20.0  # approximate center of MCX volume` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 234 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig = plt.figure(figsize=(20, 14))` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 241 | `0.2` | VOXEL_SIZE_MM | TODO | `s=0.5, c="gray", alpha=0.2)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 244 | `38` | TRUNK_SIZE_MM X | TODO | `ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 244 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 251 | `0.2` | VOXEL_SIZE_MM | TODO | `s=0.5, c="gray", alpha=0.2)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 254 | `38` | TRUNK_SIZE_MM X | TODO | `ax_union.set_xlim(0, 38); ax_union.set_ylim(0, 40); ax_union.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 254 | `40` | TRUNK_SIZE_MM Y | TODO | `ax_union.set_xlim(0, 38); ax_union.set_ylim(0, 40); ax_union.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 267 | `40` | TRUNK_SIZE_MM Y | TODO | `half_fov = 40.0` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 271 | `0.1` | ATLAS voxel size | TODO | `ax.scatter(u_all[~valid], v_all[~valid], s=0.3, c="gray", alpha=0.1)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 281 | `0.1` | ATLAS voxel size | TODO | `ax.scatter(u_all[~union_visible], v_all[~union_visible], s=0.3, c="gray", alpha=0.1)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 291 | `0.2` | VOXEL_SIZE_MM | TODO | `colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(angles)))` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 352 | `200` | TRUNK_GRID_SHAPE Y | TODO | `im = ax.imshow(d_cap, cmap="coolwarm", origin="lower", vmin=180, vmax=200)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 385 | `20` | VOLUME_CENTER_WORLD Y | TODO | `fig = plt.figure(figsize=(20, 5 * ((n_plot + 3) // 4)))` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 413 | `38` | TRUNK_SIZE_MM X | TODO | `ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 413 | `40` | TRUNK_SIZE_MM Y | TODO | `ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 488 | `38` | TRUNK_SIZE_MM X | TODO | `ax1.set_xlim(0, 38); ax1.set_ylim(0, 40); ax1.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 488 | `40` | TRUNK_SIZE_MM Y | TODO | `ax1.set_xlim(0, 38); ax1.set_ylim(0, 40); ax1.set_zlim(0, 21)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 497 | `38` | TRUNK_SIZE_MM X | TODO | `ax2.axvline(0, color="red", lw=0.8); ax2.axvline(38, color="red", lw=0.8)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 498 | `40` | TRUNK_SIZE_MM Y | TODO | `ax2.axhline(0, color="red", lw=0.8); ax2.axhline(40, color="red", lw=0.8)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 507 | `38` | TRUNK_SIZE_MM X | TODO | `ax3.axvline(0, color="red", lw=0.8); ax3.axvline(38, color="red", lw=0.8)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 508 | `20` | VOLUME_CENTER_WORLD Y | TODO | `ax3.axhline(0, color="red", lw=0.8); ax3.axhline(20.8, color="red", lw=0.8)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 508 | `20.8` | TRUNK_SIZE_MM Z | TODO | `ax3.axhline(0, color="red", lw=0.8); ax3.axhline(20.8, color="red", lw=0.8)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 523 | `0.2` | VOXEL_SIZE_MM | TODO | `s=0.5, c="gray", alpha=0.2, label="Behind (self-occ)")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 532 | `0.2` | VOXEL_SIZE_MM | TODO | `ax4.set_aspect("equal"); ax4.grid(True, alpha=0.2)` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 584 | `20` | VOLUME_CENTER_WORLD Y | TODO | `f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8] mm\n"` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 584 | `38` | TRUNK_SIZE_MM X | TODO | `f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8] mm\n"` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 584 | `40` | TRUNK_SIZE_MM Y | TODO | `f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8] mm\n"` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 584 | `20.8` | TRUNK_SIZE_MM Z | TODO | `f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8] mm\n"` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 615 | `20` | VOLUME_CENTER_WORLD Y | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 615 | `38` | TRUNK_SIZE_MM X | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 615 | `40` | TRUNK_SIZE_MM Y | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
| `FMT-SimGen/scripts/visualize_sample_qa.py` | 615 | `20.8` | TRUNK_SIZE_MM Z | TODO | `print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")` |
