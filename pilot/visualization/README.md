# FMT-SimGen Paper Visualization

This package generates paper-quality figures and tables for the GS-FMT paper.

## Quick Start

Generate all figures and tables with one command:

```bash
uv run python pilot/visualization/run_all_plots.py
```

## Generated Outputs

All outputs are saved to `pilot/visualization/figures/`:

### Figures (PDF + PNG)

| Figure | Description | Data Source |
|--------|-------------|-------------|
| `fig1_e0_psf_vs_mcx` | E0: PSF comparison (2×3 grid) | `pilot/e0_psf_validation/results/profiles/` |
| `fig2_e0_residual_vs_depth` | E0: Residual analysis | `pilot/e0_psf_validation/results/profiles/` |
| `fig3_e1c_kernel_selection` | E1c: Kernel selection | `pilot/e1c_green_function_selection/results/summary.json` |
| `fig4_e1d_atlas_vs_flat` | E1d: Atlas vs Flat geometry | `pilot/e1d_finite_source_local_surface/results/e1d_r2_summary.json` |
| `fig5_e1d_quadrature_comparison` | E1d: Quadrature accuracy-speed tradeoff | `pilot/e1d_finite_source_local_surface/results/e1d_r2_summary.json` |

### Tables (LaTeX + CSV)

| Table | Description |
|-------|-------------|
| `table1_e0_summary` | E0 validation results per configuration |
| `table2_e1d_summary` | E1d experiment results (geometry, quadrature, inverse) |

## File Structure

```
pilot/visualization/
├── __init__.py                     # Package init
├── README.md                       # This file
├── run_all_plots.py               # Main entry point
├── plot_style.py                  # Unified styling & colors
├── plot_e0_psf_comparison.py      # Figure 1 & 2
├── plot_e1c_kernel_selection.py   # Figure 3
├── plot_e1d_atlas_geometry.py     # Figure 4
├── plot_e1d_quadrature.py         # Figure 5
├── generate_tables.py             # Table 1 & 2
└── figures/                       # Output directory
    ├── *.pdf                      # Vector figures
    ├── *.png                      # Raster figures
    ├── *.tex                      # LaTeX tables
    └── *.csv                      # CSV tables
```

## Design Principles

1. **Unified Styling**: All figures use `plot_style.py` for consistent:
   - Color palette (colorblind-friendly)
   - Font sizes (TMI/IEEE paper style)
   - Line widths and markers

2. **Graceful Degradation**: Scripts handle missing data gracefully:
   - Skip panels with missing data
   - Show placeholder text for unavailable experiments

3. **Dual Output**: All figures saved as both PDF (publication) and PNG (web/preview)

## Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| MCX/Ref | `#1a1a1a` | Ground truth (black) |
| Green (half-space) | `#2166ac` | Main method (dark blue) |
| Green (infinite) | `#67a9cf` | Ablation (light blue) |
| Gaussian PSF | `#ef8a62` | Baseline (orange-red) |
| Atlas | `#2166ac` | Atlas geometry (blue) |
| Flat | `#ef8a62` | Flat assumption (orange) |

## Adding New Figures

To add a new figure:

1. Create `plot_new_figure.py` with a `main()` function
2. Import from `plot_style` for consistent styling
3. Add to `run_all_plots.py` module list
4. Follow the existing pattern for data loading and error handling

## Troubleshooting

**Issue**: Missing data warnings
- **Solution**: Check that experiment results exist in the expected directories
- Scripts will skip unavailable data and continue

**Issue**: Font warnings
- **Solution**: Install DejaVu fonts or modify `plot_style.py` to use available fonts

**Issue**: Permission denied
- **Solution**: Ensure the `figures/` directory is writable
