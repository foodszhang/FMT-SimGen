#!/usr/bin/env python3
"""
Step 2m: Generate MCX pattern3d source files from tumor_params.json.

Reads tumor_params.json from each sample directory and generates:
    - MCX JSON config (sample_XXXX.json)
    - Source binary (source-XXXX.bin)

Usage:
    python scripts/step2m_generate_mcx_sources.py [--experiment EXPERIMENT] [--sample_start N] [--sample_end M]

Output:
    data/{experiment}/samples/sample_XXXX/ containing JSON and BIN files
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.mcx_config import generate_mcx_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config/default.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2m: Generate MCX pattern3d sources from tumor_params.json"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
    )
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Starting sample index",
    )
    parser.add_argument(
        "--sample_end",
        type=int,
        default=None,
        help="Ending sample index (exclusive, default: all)",
    )
    args = parser.parse_args()

    config = load_config()
    mcx_cfg = config.get("mcx", {})

    if not mcx_cfg:
        logger.error("No 'mcx' section found in config/default.yaml")
        sys.exit(1)

    experiment_dir = Path("data") / args.experiment / "samples"

    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Find all sample directories
    sample_dirs = sorted(experiment_dir.glob("sample_*"))
    sample_dirs = [d for d in sample_dirs if d.is_dir()]

    if args.sample_end is not None:
        sample_dirs = sample_dirs[args.sample_start : args.sample_end]
    else:
        sample_dirs = sample_dirs[args.sample_start:]

    logger.info(f"Processing {len(sample_dirs)} samples from {experiment_dir}")

    success_count = 0
    error_count = 0

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name  # e.g., "sample_0000"

        tumor_params_path = sample_dir / "tumor_params.json"
        if not tumor_params_path.exists():
            logger.warning(f"  Skipping {sample_id}: no tumor_params.json")
            continue

        try:
            with open(tumor_params_path, "r") as f:
                tumor_params = json.load(f)

            json_path = generate_mcx_config(
                sample_id=sample_id,
                tumor_params=tumor_params,
                mcx_config=mcx_cfg,
                output_dir=sample_dir,
            )
            logger.info(f"  {sample_id}: Generated {json_path}")
            success_count += 1
        except Exception as e:
            logger.error(f"  {sample_id}: ERROR - {e}")
            error_count += 1

    logger.info(f"\nStep 2m complete: {success_count} succeeded, {error_count} errors.")


if __name__ == "__main__":
    main()
