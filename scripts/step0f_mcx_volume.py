#!/usr/bin/env python3
"""
Step 0f: Prepare MCX volume and material configuration.

Crops the torso region from the full Digimouse atlas, downsamples 2x,
and generates MCX simulation inputs.

Usage:
    python scripts/step0f_mcx_volume.py

Output:
    output/shared/mcx_volume_trunk.bin - uint8 label volume (ZYX order)
    output/shared/mcx_material.yaml - MCX material parameter list
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from fmt_simgen.mcx_volume import (
    prepare_mcx_volume,
    save_mcx_volume,
    print_volume_statistics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"


def load_config() -> dict:
    """Load configuration from config/default.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Step 0f: Prepare MCX volume and material configuration"
    )
    parser.add_argument(
        "--atlas_path",
        type=str,
        default=None,
        help="Path to atlas npz file (default: output/shared/atlas_full.npz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: output/shared)",
    )
    parser.add_argument(
        "--skip_stats",
        action="store_true",
        help="Skip printing detailed statistics",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 0f: MCX Volume Preparation")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    mcx_config = config.get("mcx", {})
    logger.info(f"MCX config: {mcx_config}")

    # Atlas path
    if args.atlas_path:
        atlas_path = Path(args.atlas_path)
    else:
        atlas_path = output_dir / "atlas_full.npz"

    if not atlas_path.exists():
        logger.error(f"Atlas file not found: {atlas_path}")
        logger.error("Please run step0a or equivalent to generate atlas_full.npz")
        sys.exit(1)

    logger.info(f"Loading atlas from: {atlas_path}")

    # Load atlas to get original shape and voxel size
    atlas_data = np.load(atlas_path, allow_pickle=True)

    # Find label key (check original_labels first since tissue_labels may not exist)
    label_key = None
    for key in ["original_labels", "tissue_labels", "labels", "data", "volume"]:
        if key in atlas_data:
            label_key = key
            break

    if label_key is None:
        logger.error(f"No label key found in atlas. Available keys: {list(atlas_data.keys())}")
        sys.exit(1)

    original_labels = atlas_data[label_key]
    original_shape = original_labels.shape
    voxel_size = float(atlas_data["voxel_size"])
    logger.info(f"Original atlas shape: {original_shape}, voxel size: {voxel_size} mm")

    # Prepare MCX volume
    trunk_crop = mcx_config.get("trunk_crop", {})
    y_start = trunk_crop.get("y_start", 300)
    y_end = trunk_crop.get("y_end", 700)
    downsample_factor = mcx_config.get("downsample_factor", 2)

    volume_zyx, material_list = prepare_mcx_volume(atlas_path, config)

    # Save outputs
    volume_bin_path, material_yaml_path = save_mcx_volume(
        volume_zyx, material_list, output_dir
    )

    # Print statistics
    if not args.skip_stats:
        print_volume_statistics(
            volume_zyx=volume_zyx,
            voxel_size=voxel_size,
            original_shape=original_shape,
            y_start=y_start,
            y_end=y_end,
            downsample_factor=downsample_factor,
            material_list=material_list,
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("OUTPUT FILES")
    logger.info("=" * 60)
    logger.info(f"  Volume: {volume_bin_path}")
    logger.info(f"  Material: {material_yaml_path}")
    logger.info("")
    logger.info("MCX config section for config/default.yaml:")
    logger.info(f"  volume_path: {str(volume_bin_path.absolute())}")
    logger.info(f"  material_path: {str(material_yaml_path.absolute())}")
    logger.info(f"  trunk_offset_mm: [0, {y_start * voxel_size}, 0]")
    logger.info(f"  volume_shape (ZYX): {list(volume_zyx.shape)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 0f COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
