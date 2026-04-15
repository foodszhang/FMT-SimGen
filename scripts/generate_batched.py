#!/usr/bin/env python3
"""
Batched sample generation with subprocess isolation.

Each batch runs in a subprocess, guaranteeing C-layer memory (numpy/scipy malloc
碎片化) is fully reclaimed between batches.

Usage:
    # Generate 1000 samples in batches of 50 (subprocess-isolated)
    uv run python scripts/generate_batched.py \
        --config config/uniform_1000_v2.yaml \
        --total 1000 --batch_size 50

    # Resume: starts from first incomplete/missing batch
    uv run python scripts/generate_batched.py \
        --config config/uniform_1000_v2.yaml \
        --total 1000 --batch_size 50

Exit code: 0 on success, 1 on any batch failure.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# Reusable imports
REQUIRED_FILES = ["measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy", "tumor_params.json"]


def deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def load_config_with_inheritance(config_path):
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = __import__("yaml").safe_load(f)
    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = config_path.parent / base_name
        base_cfg = load_config_with_inheritance(str(base_path))
        deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def check_batch_complete(samples_dir, batch_start, batch_size):
    """Return True if all samples in batch are complete."""
    for i in range(batch_start, batch_start + batch_size):
        d = samples_dir / f"sample_{i:04d}"
        if not d.is_dir():
            return False
        for fn in REQUIRED_FILES:
            if not (d / fn).exists():
                return False
    return True


def remove_batch(samples_dir, batch_start, batch_size):
    """Remove all sample directories in a batch."""
    for i in range(batch_start, batch_start + batch_size):
        d = samples_dir / f"sample_{i:04d}"
        if d.exists():
            shutil.rmtree(d)


def run_batch(batch_start, batch_end, config_path, experiment_name):
    """Run one batch as a subprocess."""
    script = Path(__file__).parent / "02_generate_dataset.py"

    cmd = [
        sys.executable,  # use current interpreter (uv run)
        str(script),
        "-c", str(config_path),
        "--start_index", str(batch_start),
        "--end_index", str(batch_end),
    ]

    batch_size = batch_end - batch_start
    logger.info(f"Batch {batch_start}-{batch_start + batch_size - 1}: starting subprocess")
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        logger.error(f"Batch {batch_start}-{batch_start + batch_size - 1}: FAILED")
        logger.error(result.stdout)
        logger.error(result.stderr)
        return False

    logger.info(f"Batch {batch_start}-{batch_start + batch_size - 1}: OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batched sample generation with subprocess isolation")
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to experiment config file"
    )
    parser.add_argument(
        "--total", "-t", type=int, required=True,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=50,
        help="Number of samples per batch (default: 50)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run all batches (overwrites existing)"
    )
    args = parser.parse_args()

    # Load config to get experiment output dir
    config = load_config_with_inheritance(args.config)
    experiment_name = config.get("dataset", {}).get("experiment_name", "default")
    samples_dir = Path("data") / experiment_name / "samples"

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    num_batches = (args.total + args.batch_size - 1) // args.batch_size

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Total samples: {args.total}, batch_size: {args.batch_size}, num_batches: {num_batches}")
    logger.info(f"Samples dir: {samples_dir}")
    logger.info("")

    success = True

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.total)

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: samples {batch_start}-{batch_end - 1}")

        if not args.force and check_batch_complete(samples_dir, batch_start, args.batch_size):
            logger.info(f"  [SKIP] all {args.batch_size} samples already complete")
            continue

        if not args.force:
            remove_batch(samples_dir, batch_start, args.batch_size)

        ok = run_batch(batch_start, batch_end, config_path, experiment_name)
        if not ok:
            success = False
            logger.error(f"Batch {batch_idx + 1}/{num_batches} failed, stopping")
            break

    if success:
        logger.info("")
        logger.info(f"All batches complete. {args.total} samples in {samples_dir}")
        logger.info("Run scripts/generate_manifest.py to generate dataset_manifest.json and splits/")
    else:
        logger.error("Some batches failed.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
