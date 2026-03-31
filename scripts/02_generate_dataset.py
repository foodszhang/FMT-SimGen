#!/usr/bin/env python3
"""
Step 1-4: Generate dataset samples.

This script generates N tumor samples with forward measurements.
Run after 01_generate_mesh.py.

Usage:
    python scripts/02_generate_dataset.py [--num_samples N]
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from fmt_simgen.dataset.builder import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Generate FMT dataset samples")
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to generate (uses config default if not specified)",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    builder = DatasetBuilder(config)

    print("=" * 60)
    print("Step 1-4: Generating dataset samples")
    print("=" * 60)

    if args.num_samples is not None:
        print(f"Generating {args.num_samples} samples...")
    else:
        n = config.get("dataset", {}).get("num_samples", 50)
        print(f"Generating {n} samples (from config)...")

    samples = builder.build_samples(num_samples=args.num_samples)

    print(f"\nDataset generation complete! {len(samples)} samples saved.")


if __name__ == "__main__":
    main()
