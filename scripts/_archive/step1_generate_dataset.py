#!/usr/bin/env python3
"""
Step 1: Generate FMT Dataset Samples

This script generates tumor samples and computes forward measurements.

Usage:
    python scripts/step1_generate_dataset.py [--num_samples N] [--regenerate]

Output:
    data/sample_XXXX/ containing measurement_b.npy, gt_nodes.npy, gt_voxels.npy, tumor_params.json
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from fmt_simgen.dataset.builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate FMT Dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: from config)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate shared assets even if they exist",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on first sample",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Step 1: FMT Dataset Generation")
    logger.info("=" * 60)

    config = load_config()

    builder = DatasetBuilder(config)

    logger.info("Loading/generating shared assets...")
    assets = builder.build_shared_assets(force_regenerate=args.regenerate)
    logger.info(f"  Mesh: {assets['mesh']}")
    logger.info(f"  Matrices: {assets['matrix']}")

    logger.info("\nGenerating tumor samples...")
    samples = builder.build_samples(num_samples=args.num_samples)

    logger.info("=" * 60)
    logger.info(f"DATASET GENERATION COMPLETE: {len(samples)} samples")
    logger.info("=" * 60)

    if args.validate and samples:
        sample = samples[0]
        logger.info("\nFirst sample validation:")
        logger.info(
            f"  gt_nodes: shape={sample.gt_nodes.shape}, "
            f"max={sample.gt_nodes.max():.4f}, sum={sample.gt_nodes.sum():.4f}"
        )
        logger.info(
            f"  gt_voxels: shape={sample.gt_voxels.shape}, "
            f"max={sample.gt_voxels.max():.4f}, sum={sample.gt_voxels.sum():.4f}"
        )
        logger.info(
            f"  measurement_b: shape={sample.measurement_b.shape}, "
            f"max={sample.measurement_b.max():.4f}, sum={sample.measurement_b.sum():.4f}"
        )
        logger.info(f"  tumor_params: {sample.tumor_params}")


if __name__ == "__main__":
    main()
