"""Generate all paper figures and tables.

Usage:
    uv run python pilot/visualization/run_all_plots.py

Outputs:
    - figures/fig1_e0_psf_vs_mcx.pdf
    - figures/fig2_e0_residual_vs_depth.pdf
    - figures/fig3_e1c_kernel_selection.pdf
    - figures/fig4_e1d_atlas_vs_flat.pdf
    - figures/fig5_e1d_quadrature_comparison.pdf
    - figures/table1_e0_summary.tex/csv
    - figures/table2_e1d_summary.tex/csv
"""

import sys
import traceback
from pathlib import Path

# Add visualization directory to path
viz_dir = Path(__file__).parent
sys.path.insert(0, str(viz_dir))


def run_module(module_name: str, description: str) -> bool:
    """Run a plotting module and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print("=" * 60)

    try:
        # Import and run the module
        module = __import__(module_name)
        if hasattr(module, "main"):
            module.main()
        print(f"✓ {description} completed")
        return True
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all plotting scripts."""
    print("=" * 60)
    print("FMT-SimGen Paper Visualization Pipeline")
    print("=" * 60)
    print(f"Output directory: {viz_dir / 'figures'}")

    # Ensure output directory exists
    figures_dir = viz_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # List of modules to run
    modules = [
        ("plot_e0_psf_comparison", "Figure 1 & 2: E0 PSF Validation"),
        ("plot_e1c_kernel_selection", "Figure 3: E1c Kernel Selection"),
        ("plot_e1d_atlas_geometry", "Figure 4: E1d Atlas Geometry"),
        ("plot_e1d_quadrature", "Figure 5: E1d Quadrature Comparison"),
        ("generate_tables", "Table 1 & 2: Summary Tables"),
    ]

    results = []
    for module_name, description in modules:
        success = run_module(module_name, description)
        results.append((description, success))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {description}")

    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} modules completed successfully")

    if passed == total:
        print("\n✓ All figures and tables generated successfully!")
        print(f"\nOutput files in: {figures_dir}")
        print("\nGenerated files:")
        for f in sorted(figures_dir.glob("*.pdf")):
            print(f"  - {f.name}")
        for f in sorted(figures_dir.glob("*.csv")):
            print(f"  - {f.name}")
        return 0
    else:
        print("\n⚠ Some modules failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
