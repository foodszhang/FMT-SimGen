#!/usr/bin/env python3
"""
Step 1-4: Generate dataset samples.

This script generates N tumor samples with forward measurements.
Run after 01_generate_mesh.py.

Usage:
    python scripts/02_generate_dataset.py --config config/gaussian_1000.yaml
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from fmt_simgen.dataset.builder import DatasetBuilder


def deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def load_config_with_inheritance(config_path: str) -> dict:
    """Load configuration file with _base_ inheritance support.

    Parameters
    ----------
    config_path : str
        Path to the config file.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = config_path.parent / base_name
        base_cfg = load_config_with_inheritance(str(base_path))
        deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate FMT dataset samples")
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

    builder = DatasetBuilder(config)

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

    builder.build_samples(num_samples=target_num, start_index=args.start_index)

    print(f"\nDataset generation complete! See data/{experiment_name}/ for output.")


if __name__ == "__main__":
    main()
