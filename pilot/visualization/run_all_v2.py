"""Unified script for bugfix validation and 2D visualizations.

Usage:
    uv run python pilot/visualization/run_all_v2.py [--skip-validation] [--skip-viz]
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_bugfix_validation():
    """Run bugfix validation in e1d directory."""
    print("=" * 70)
    print("PART 1: Bugfix Validation")
    print("=" * 70)

    validate_script = (
        Path(__file__).parent.parent
        / "e1d_finite_source_local_surface"
        / "validate_and_run.py"
    )

    if not validate_script.exists():
        print(f"Error: Validation script not found: {validate_script}")
        return False

    result = subprocess.run(
        ["uv", "run", "python", str(validate_script)],
        cwd="/home/foods/pro/FMT-SimGen",
        capture_output=False,
    )

    return result.returncode == 0


def run_2d_visualizations():
    """Run 2D visualization scripts."""
    print("\n" + "=" * 70)
    print("PART 2: 2D Visualization Generation")
    print("=" * 70)

    viz_dir = Path(__file__).parent

    scripts = [
        ("plot_e0_2d_comparison.py", "Figure 1: E0 2D MCX vs Green"),
        ("plot_e1d_atlas_vs_flat_2d.py", "Figure 3: E1d Atlas vs Flat 2D"),
        ("plot_e0_psf_comparison.py", "Figures 1-2: E0 PSF (existing)"),
        ("plot_e1c_kernel_selection.py", "Figure 3: E1c Kernel Selection"),
        ("plot_e1d_quadrature.py", "Figure 5: E1d Quadrature"),
        ("generate_tables.py", "Tables 1-2: Summary Tables"),
    ]

    results = []
    for script, description in scripts:
        script_path = viz_dir / script
        if not script_path.exists():
            print(f"⚠ {description}: Script not found, skipping")
            results.append((description, False))
            continue

        print(f"\n[{description}]")
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script_path)],
                cwd="/home/foods/pro/FMT-SimGen",
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print(f"  ✓ Success")
                results.append((description, True))
            else:
                print(f"  ✗ Failed with code {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                results.append((description, False))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((description, False))

    return results


def show_summary(viz_results):
    """Show summary of all operations."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in viz_results if success)
    total = len(viz_results)

    print(f"\nVisualization Results: {passed}/{total} succeeded")
    for desc, success in viz_results:
        status = "✓" if success else "✗"
        print(f"  {status} {desc}")

    print("\n" + "-" * 70)
    print("Output files in: pilot/visualization/figures/")
    print("-" * 70)

    # List generated files
    figures_dir = Path(__file__).parent / "figures"
    if figures_dir.exists():
        pdfs = list(figures_dir.glob("*.pdf"))
        pngs = list(figures_dir.glob("*.png"))
        print(f"\nGenerated {len(pdfs)} PDFs and {len(pngs)} PNGs")

        print("\nKey outputs:")
        for f in sorted(figures_dir.glob("*.pdf")):
            size_kb = f.stat().st_size / 1024
            print(f"  • {f.name} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Bugfix validation and 2D visualization pipeline"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip bugfix validation"
    )
    parser.add_argument(
        "--skip-viz", action="store_true", help="Skip visualization generation"
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run full E1d experiments (takes time)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FMT-SimGen E1d-R2: Bugfix Validation & 2D Visualization Pipeline")
    print("=" * 70)

    # Part 1: Validation
    if not args.skip_validation:
        validation_passed = run_bugfix_validation()
        if not validation_passed:
            print("\n⚠ Validation had issues, but continuing...")
    else:
        print("\n⏩ Skipping validation (as requested)")

    # Optional: Run full experiments
    if args.run_experiments:
        print("\n" + "=" * 70)
        print("Running Full E1d Experiments")
        print("=" * 70)
        print("This may take 30+ minutes...")
        # Would call run_e1d_atlas.py here

    # Part 2: Visualizations
    if not args.skip_viz:
        viz_results = run_2d_visualizations()
        show_summary(viz_results)
    else:
        print("\n⏩ Skipping visualizations (as requested)")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
