#!/usr/bin/env python3
"""
Post-process a completed sample directory to generate dataset_manifest.json and splits.

Run after all sample directories are complete (e.g., after generate_batched.py finishes).

Usage:
    uv run python scripts/generate_manifest.py \
        --experiment uniform_1000_v2 \
        --num_samples 1000

Exit code: 0 on success, 1 if samples are missing or incomplete.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FILES = ["measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy", "tumor_params.json"]


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest and splits")
    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        help="Experiment name (e.g., uniform_1000_v2)"
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, required=True,
        help="Expected total number of samples"
    )
    args = parser.parse_args()

    base_dir = Path("data") / args.experiment
    samples_dir = base_dir / "samples"
    splits_dir = base_dir / "splits"

    # Scan all sample directories
    all_ids = []
    missing_samples = []

    for i in range(args.num_samples):
        d = samples_dir / f"sample_{i:04d}"
        if not d.is_dir():
            missing_samples.append(i)
            continue
        missing_files = [fn for fn in REQUIRED_FILES if not (d / fn).exists()]
        if missing_files:
            logger.error(f"  sample_{i:04d}: incomplete — missing {missing_files}")
            missing_samples.append(i)
            continue
        all_ids.append(f"sample_{i:04d}")

    if missing_samples:
        logger.error(f"{len(missing_samples)} samples missing or incomplete: {missing_samples[:10]}...")
        sys.exit(1)

    logger.info(f"All {len(all_ids)} samples complete.")

    # Build metadata
    import numpy as np

    samples_metadata = []
    for sample_id in sorted(all_ids):
        d = samples_dir / sample_id
        tumor_params_path = d / "tumor_params.json"

        with open(tumor_params_path) as f:
            tumor_params = json.load(f)

        gt_nodes = np.load(d / "gt_nodes.npy")
        measurement_b = np.load(d / "measurement_b.npy")

        samples_metadata.append({
            "id": sample_id,
            "num_foci": tumor_params.get("num_foci", 1),
            "depth_tier": tumor_params.get("depth_tier", "unknown"),
            "depth_mm": tumor_params.get("depth_mm"),
            "b_max": float(np.max(np.abs(measurement_b))),
            "b_mean": float(np.mean(np.abs(measurement_b))),
            "gt_max": float(np.max(gt_nodes)),
            "gt_nonzero_count": int(np.count_nonzero(gt_nodes)),
            "gt_nonzero_frac": float(np.count_nonzero(gt_nodes)) / len(gt_nodes),
            "has_gt_voxels": True,
        })

    # Generate splits (80/20, seed=42)
    random.seed(42)
    shuffled = list(sorted(all_ids))
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_ids = sorted(shuffled[:split_idx])
    val_ids = sorted(shuffled[split_idx:])

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(splits_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids) + "\n")
    logger.info(f"Splits: {len(train_ids)} train, {len(val_ids)} val")

    # Generate manifest
    manifest = {
        "experiment_name": args.experiment,
        "source_type": "uniform",
        "num_samples": len(all_ids),
        "samples": samples_metadata,
    }

    manifest_path = base_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
