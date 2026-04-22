#!/usr/bin/env python3
"""
run_mcx_pipeline.py — Standalone MCX channel entry point for FMT-SimGen.

Orchestrates Steps 2m (source config) → 3m (MCX simulate) → 4m (projection)
on existing dataset samples. Can be run independently or via run_all.py --phase mcx.

**This script is the standalone CLI wrapper.** It loads config internally
(including default.yaml) for backward compatibility. For programmatic use
with an already-loaded config, use ``fmt_simgen.pipeline.mcx_pipeline.run_mcx_pipeline()``.

Usage:
    # Full MCX pipeline on existing samples (auto-generates source configs if missing)
    python scripts/run_mcx_pipeline.py --samples_dir data/gaussian_1000/samples

    # Projection only (if .jnii files already exist)
    python scripts/run_mcx_pipeline.py --samples_dir data/gaussian_1000/samples --projection_only

    # Re-run everything including MCX simulation
    python scripts/run_mcx_pipeline.py --samples_dir data/gaussian_1000/samples --force_mcx

Exit codes:
    0  -- all samples succeeded
    1  -- MCX not found, or critical error
    2  -- samples_dir not found
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.pipeline.mcx_pipeline import run_mcx_pipeline
from fmt_simgen.pipeline.shared import load_config_with_inheritance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_shared_config() -> dict:
    """Load the full config (mcx + view_config sections) for standalone CLI use.

    Reads ``config/default.yaml`` and merges ``output/shared/view_config.json``
    into ``config["view_config"]``.
    """
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    view_json_path = Path("output/shared/view_config.json")
    if view_json_path.exists():
        import json
        with open(view_json_path, "r") as f:
            cfg["view_config"] = json.load(f)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_mcx_pipeline: MCX channel (Steps 2m→3m→4m) for FMT-SimGen"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Root directory containing sample subdirectories",
    )
    parser.add_argument(
        "--projection_only",
        action="store_true",
        help="Skip MCX simulation; only run projection (use when .jnii files exist)",
    )
    parser.add_argument(
        "--force_mcx",
        action="store_true",
        help="Re-run MCX simulation even if .jnii already exists (not yet wired to pipeline)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples that already have proj.npz (default: True)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-generate proj.npz even if it already exists",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel projection workers (default: 1, GPU MCX is sequential)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        sys.stderr.write(f"Error: samples_dir not found: {samples_dir}\n")
        sys.exit(2)

    skip_existing = args.skip_existing and not args.no_skip

    # Load config for standalone CLI (reads default.yaml + view_config.json)
    config = load_shared_config()

    logger.info(
        "run_mcx_pipeline (standalone): samples_dir=%s, projection_only=%s, force_mcx=%s, max_workers=%d",
        samples_dir,
        args.projection_only,
        args.force_mcx,
        args.max_workers,
    )

    try:
        run_mcx_pipeline(
            config=config,
            samples_dir=samples_dir,
            projection_only=args.projection_only,
            max_workers=args.max_workers,
            no_skip=args.no_skip,
        )
        sys.exit(0)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)
    except RuntimeError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
