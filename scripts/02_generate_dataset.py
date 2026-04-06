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
        help="Number of samples to generate (uses config default if not specified)",
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

    if args.num_samples is not None:
        print(f"Generating {args.num_samples} samples...")
    else:
        n = config.get("dataset", {}).get("num_samples", 50)
        print(f"Generating {n} samples (from config)...")

    samples = builder.build_samples(num_samples=args.num_samples)

    print(f"\nDataset generation complete! {len(samples)} samples saved.")


if __name__ == "__main__":
    main()
