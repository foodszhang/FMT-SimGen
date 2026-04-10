#!/usr/bin/env python3
"""
Step 3m: Run MCX simulations for all samples.

Generates 3D fluence volumes (.jnii) from MCX JSON configs produced by Step 2m.

Usage:
    python scripts/step3m_mcx_simulate.py --samples_dir data/gaussian_1000/samples

Output per sample:
    {sample_id}/{session_id}.jnii  -- 3D fluence volume

Exit codes:
    0  -- all samples succeeded or already have outputs
    1  -- MCX not found in PATH
    2  -- no sample directories found
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure fmt_simgen is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.mcx_runner import detect_mcx_executable, run_mcx_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3m: Run MCX simulations")
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Root directory containing sample subdirectories",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, GPU simulations should not parallelize)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        sys.stderr.write(f"Error: samples_dir not found: {samples_dir}\n")
        sys.exit(2)

    # Check MCX availability
    try:
        mcx_exec, gpu_supported = detect_mcx_executable()
    except RuntimeError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    # Run batch
    results = run_mcx_batch(samples_dir, max_workers=args.max_workers)

    print(f"\nStep 3m complete: {len(results)} simulations succeeded")
    for path in results:
        print(f"  {path}")


if __name__ == "__main__":
    main()
