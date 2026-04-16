#!/usr/bin/env python3
"""
run_all.py — Dual-channel (DE + MCX) pipeline entry point for FMT-SimGen.

Generates N dataset samples with DE channel (forward measurement) and optionally
MCX channel (3D fluence simulation + multi-angle projections).

Usage:
    # DE + MCX dual channel (50 samples, default)
    python scripts/run_all.py --num_samples 50

    # DE channel only (no MCX)
    python scripts/run_all.py --num_samples 50 --disable_mcx

    # MCX channel only (on existing DE samples)
    python scripts/run_mcx_pipeline.py --samples_dir output/samples

Exit codes:
    0  -- all phases succeeded
    1  -- DE or MCX phase failed
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.dataset.builder import DatasetBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def load_config_with_inheritance(config_path: str) -> dict:
    """Load configuration file with _base_ inheritance support."""
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


def run_de_pipeline(num_samples: int | None, config_path: str) -> bool:
    """Run the DE channel (Steps 1-4 via DatasetBuilder).

    Returns True if successful.
    """
    logger.info("=" * 60)
    logger.info("Phase DE: Generating %s samples with forward measurements",
                "N" if num_samples is None else num_samples)
    logger.info("=" * 60)

    config = load_config_with_inheritance(config_path)
    experiment_name = config.get("dataset", {}).get("experiment_name", "default")

    builder = DatasetBuilder(config)

    try:
        builder.build_samples(num_samples=num_samples)
    except Exception as e:
        logger.error("DE phase failed: %s", e)
        return False

    logger.info("DE phase complete: samples in data/%s/", experiment_name)
    return True


def run_mcx_pipeline(samples_dir: Path, projection_only: bool = False,
                       max_workers: int = 1, no_skip: bool = False) -> bool:
    """Run the MCX channel via run_mcx_pipeline.py subprocess.

    Returns True if script exits with code 0.
    """
    logger.info("=" * 60)
    logger.info("Phase MCX: Running MCX pipeline on data/%s/", samples_dir.name)
    logger.info("=" * 60)

    cmd = [
        sys.executable,  # use current interpreter (uv run)
        str(Path(__file__).parent / "run_mcx_pipeline.py"),
        "--samples_dir", str(samples_dir),
    ]
    if projection_only:
        cmd.append("--projection_only")
    if no_skip:
        cmd.append("--no_skip")
    if max_workers > 1:
        cmd.extend(["--max_workers", str(max_workers)])

    try:
        result = subprocess.run(
            cmd,
            check=False,  # don't raise on non-zero exit
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error("MCX pipeline subprocess failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_all.py: DE + MCX dual-channel pipeline for FMT-SimGen"
    )
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
    parser.add_argument(
        "--disable_mcx",
        action="store_true",
        help="Skip MCX channel (DE only)",
    )
    parser.add_argument(
        "--mcx_projection_only",
        action="store_true",
        help="Skip MCX simulation; only generate projections (for re-running on existing .jnii)",
    )
    parser.add_argument(
        "--mcx_max_workers",
        type=int,
        default=1,
        help="Number of parallel workers for MCX projection (default: 1)",
    )
    parser.add_argument(
        "--mcx_no_skip",
        action="store_true",
        help="Re-generate proj.npz even if they already exist",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(__file__).parent.parent / args.config
    config = load_config_with_inheritance(str(config_path))
    experiment_name = config.get("dataset", {}).get("experiment_name", "default")
    samples_base = Path(__file__).parent.parent / "data" / experiment_name / "samples"

    t0 = time.time()

    # Phase DE
    de_ok = run_de_pipeline(args.num_samples, str(config_path))

    # Phase MCX (enabled by default)
    mcx_ok = True
    if not args.disable_mcx and de_ok:
        mcx_ok = run_mcx_pipeline(
            samples_dir=samples_base,
            projection_only=args.mcx_projection_only,
            max_workers=args.mcx_max_workers,
            no_skip=args.mcx_no_skip,
        )

    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete (%.1fs)", elapsed)
    logger.info("  DE phase:  %s", "OK" if de_ok else "FAILED")
    if not args.disable_mcx:
        logger.info("  MCX phase: %s", "OK" if mcx_ok else "FAILED")
    logger.info("  Samples:   data/%s/samples/", experiment_name)
    logger.info("=" * 60)

    if de_ok and (args.disable_mcx or mcx_ok):
        logger.info("All phases succeeded.")
        sys.exit(0)
    else:
        logger.warning("Some phases failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
