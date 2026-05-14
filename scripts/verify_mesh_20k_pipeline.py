#!/usr/bin/env python3
"""
Verify Mesh 20k Pipeline

This script runs the complete verification pipeline for the 20k mesh:
1. Check mesh file exists
2. Generate system matrix (step0c)
3. Generate visibility config (step0g)
4. Generate voxel grid (step0d) - optional
5. Generate graph laplacian (step0e) - optional
6. Generate test samples (02_generate_dataset.py)
7. Generate comparison visualizations

Usage:
    python scripts/verify_mesh_20k_pipeline.py [--skip-optional]
"""

import subprocess
import sys
from pathlib import Path
import time
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
MESH_FILE = PROJECT_ROOT / "output" / "shared_mesh_20k" / "digimouse_trunk_mesh_20k.npz"
OUTPUT_DIR = PROJECT_ROOT / "output" / "shared_mesh_20k"
CONFIG_FILE = PROJECT_ROOT / "config" / "mesh_20k.yaml"


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"[PASS] {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"[FAIL] {description} failed with code {result.returncode}")
        return False


def main():
    print("="*60)
    print("20k Mesh Pipeline Verification")
    print("="*60)
    print(f"Mesh: {MESH_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Config: {CONFIG_FILE}")
    
    # Check mesh file exists
    if not MESH_FILE.exists():
        print(f"\n[ERROR] Mesh file not found: {MESH_FILE}")
        print("Please run mesh generation first:")
        print("  uv run python scripts/step0b_generate_mesh_cgalmesh.py \\")
        print("    --maxvol 5.0 --radbound 2.8 --distbound 2.5 \\")
        print("    --output-name digimouse_trunk_mesh_20k")
        sys.exit(1)
    
    print(f"\n[OK] Mesh file exists: {MESH_FILE.stat().st_size / 1e6:.2f} MB")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {}
    
    # Step 1: System Matrix
    results["step0c"] = run_command(
        ["uv", "run", "python", "scripts/step0c_fem_matrix.py",
         "--mesh", str(MESH_FILE),
         "--output-dir", str(OUTPUT_DIR)],
        "Step 0c: FEM System Matrix Assembly"
    )
    
    # Step 2: Visibility Config
    results["step0g"] = run_command(
        ["uv", "run", "python", "scripts/step0g_view_config.py",
         "--mesh", str(MESH_FILE),
         "--output-dir", str(OUTPUT_DIR)],
        "Step 0g: View Configuration"
    )
    
    # Check if we should skip optional steps
    skip_optional = "--skip-optional" in sys.argv
    
    if not skip_optional:
        # Step 3: Voxel Grid (optional)
        results["step0d"] = run_command(
            ["uv", "run", "python", "scripts/step0d_voxel_grid.py",
             "--mesh", str(MESH_FILE),
             "--output-dir", str(OUTPUT_DIR)],
            "Step 0d: Voxel Grid Definition (optional)"
        )
        
        # Step 4: Graph Laplacian (optional)
        results["step0e"] = run_command(
            ["uv", "run", "python", "scripts/step0e_v2_full_graph_laplacian.py",
             "--mesh", str(MESH_FILE),
             "--output-dir", str(OUTPUT_DIR)],
            "Step 0e: Graph Laplacian (optional)"
        )
    
    # Step 5: Generate test samples
    results["samples"] = run_command(
        ["uv", "run", "python", "scripts/02_generate_dataset.py",
         "--config", str(CONFIG_FILE),
         "-n", "3"],
        "Step 1-4: Generate Test Samples"
    )
    
    # Step 6: Generate comparison visualizations
    results["viz"] = run_command(
        ["uv", "run", "python", "scripts/compare_mesh_20k_52k.py"],
        "Step 5: Comparison Visualizations"
    )
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {step}: [{status}]")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n[SUCCESS] All verification steps passed!")
        print(f"\nOutput files in: {OUTPUT_DIR}")
        print("  - system_matrix.*.npz")
        print("  - visible_mask.npy")
        print("  - view_config.json")
        if not skip_optional:
            print("  - voxel_grid.npz")
            print("  - graph_laplacian_full.*.npz")
        print(f"\nTest samples in: data/mesh_20k_test/samples/")
        print("\nVisualizations in: output/visualizations/verification_20k/")
    else:
        print("\n[FAILURE] Some verification steps failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
