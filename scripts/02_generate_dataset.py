#!/usr/bin/env python3
"""
Step 1-4: Generate dataset samples (DE channel).

This script generates N tumor samples with forward measurements.
Run after the asset generation steps (0b-0g).

Usage:
    python scripts/02_generate_dataset.py --config config/gaussian_1000.yaml -n 50
    python scripts/02_generate_dataset.py --config config/default.yaml --start_index 0 --end_index 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.pipeline.de_pipeline import run_de_pipeline
from fmt_simgen.pipeline.shared import derive_samples_dir, load_config_with_inheritance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FMT dataset samples (DE channel)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/default.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=None,
        help="Total number of samples to generate (config default if not specified)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting sample index (for batched subprocess runs)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End sample index (exclusive). Use for batched runs to generate [start_index, end_index).",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / args.config
    config = load_config_with_inheritance(str(config_path))

    experiment_name = config.get("dataset", {}).get("experiment_name", "default")
    print(f"Experiment: {experiment_name}")
    print(f"Source type: {config.get('tumor', {}).get('source_type', 'gaussian')}")
    print("=" * 60)
    print("Step 1-4: Generating dataset samples")
    print("=" * 60)

    # Determine target: use end_index if provided, otherwise num_samples
    if args.end_index is not None:
        target_num = args.end_index
        if args.num_samples is not None:
            print(f"Generating samples {args.start_index} to {args.end_index - 1} (absolute range)...")
    elif args.num_samples is not None:
        target_num = args.num_samples
        print(f"Generating {args.num_samples} samples...")
    else:
        target_num = config.get("dataset", {}).get("num_samples", 50)
        print(f"Generating {target_num} samples (from config)...")

    run_de_pipeline(config, num_samples=target_num, start_index=args.start_index)

    print(
        f"\nDataset generation complete! "
        f"See data/{experiment_name}/ for output."
    )


if __name__ == "__main__":
    main()
