#!/usr/bin/env python3
"""
run_all.py — Unified pipeline entry point for FMT-SimGen.

Generates N dataset samples with DE channel (forward measurement) and optionally
MCX channel (3D fluence simulation + multi-angle projections).

Usage:
    # Full pipeline: DE + MCX (default)
    python scripts/run_all.py --config config/default.yaml -n 50

    # DE channel only
    python scripts/run_all.py --config config/default.yaml -n 50 --phase de

    # MCX channel only (on existing DE samples)
    python scripts/run_all.py --config config/default.yaml --phase mcx

    # MCX projection only (skip simulation)
    python scripts/run_all.py --config config/default.yaml --phase mcx --mcx_projection_only

    # Post-processing: validate manifest and sample completeness
    python scripts/run_all.py --config config/default.yaml --phase post

Exit codes:
    0  -- all phases succeeded
    1  -- one or more phases failed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.pipeline.de_pipeline import run_de_pipeline
from fmt_simgen.pipeline.mcx_pipeline import run_mcx_pipeline
from fmt_simgen.pipeline.shared import derive_samples_dir, load_config_with_inheritance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_post_phase(config: dict) -> bool:
    """Run post-processing: dataset manifest validation.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    bool
        True if validation passes, False otherwise.
    """
    dataset_cfg = config.get("dataset", {})
    experiment_name = dataset_cfg.get("experiment_name", "default")
    experiment_output = (
        Path(dataset_cfg.get("output_path", "data/")) / experiment_name
    )
    manifest_path = experiment_output / "dataset_manifest.json"
    samples_dir = experiment_output / "samples"

    logger.info("=" * 60)
    logger.info("Phase POST: Validating dataset manifest and sample completeness")
    logger.info("=" * 60)

    if not manifest_path.exists():
        logger.error("dataset_manifest.json not found at %s", manifest_path)
        return False

    # Validate required files exist for all samples
    incomplete = []
    for sd in sorted(samples_dir.glob("sample_*")):
        if not sd.is_dir():
            continue
        for fname in [
            "measurement_b.npy",
            "gt_nodes.npy",
            "gt_voxels.npy",
            "tumor_params.json",
        ]:
            if not (sd / fname).exists():
                incomplete.append(f"{sd.name}/{fname}")

    if incomplete:
        logger.error(
            "Incomplete samples found (%d files missing):", len(incomplete)
        )
        for m in incomplete[:10]:
            logger.error("  %s", m)
        if len(incomplete) > 10:
            logger.error("  ... and %d more", len(incomplete) - 10)
        return False

    logger.info("Post phase: dataset validation passed for %s", experiment_name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run_all.py: Unified pipeline entry point for FMT-SimGen"
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
        "--phase",
        type=str,
        choices=["de", "mcx", "post", "all"],
        default="all",
        help="Pipeline phase to run: "
        "'de' (DE generation only), "
        "'mcx' (MCX pipeline on existing DE samples), "
        "'post' (manifest validation), "
        "'all' (de+mcx, default)",
    )
    # MCX-phase flags
    parser.add_argument(
        "--mcx_projection_only",
        action="store_true",
        help="[MCX phase] Skip MCX simulation; only generate projections",
    )
    parser.add_argument(
        "--mcx_max_workers",
        type=int,
        default=1,
        help="[MCX phase] Number of parallel projection workers (default: 1)",
    )
    parser.add_argument(
        "--mcx_no_skip",
        action="store_true",
        help="[MCX phase] Re-generate proj.npz even if they already exist",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config

    # Load config once — used by all phases
    config = load_config_with_inheritance(str(config_path))

    # Merge view_config.json only when config has no view_config section.
    # This mirrors the priority in mcx_pipeline.py: YAML config wins,
    # view_config.json is only a fallback for standalone CLI use.
    if "view_config" not in config or not config["view_config"].get("angles"):
        view_json_path = project_root / "output" / "shared" / "view_config.json"
        if view_json_path.exists():
            with open(view_json_path, "r") as f:
                config["view_config"] = json.load(f)
                logger.info("view_config: loaded from output/shared/view_config.json (fallback)")
        else:
            logger.warning(
                "view_config not in config and view_config.json not found. "
                "Camera-based filtering may be missing."
            )

    experiment_name = config.get("dataset", {}).get("experiment_name", "default")
    samples_dir = derive_samples_dir(config)

    logger.info("Experiment: %s", experiment_name)
    logger.info("Samples dir: %s", samples_dir)

    t0 = time.time()

    # ── Phase dispatch ────────────────────────────────────────────────────────
    de_ok = True
    mcx_ok = True
    post_ok = True

    if args.phase in ("de", "all"):
        try:
            run_de_pipeline(config, num_samples=args.num_samples)
        except Exception as e:
            logger.error("DE phase failed: %s", e)
            de_ok = False

    if args.phase in ("mcx", "all"):
        if args.phase == "all" and not de_ok:
            logger.warning("Skipping MCX phase because DE phase failed.")
        else:
            # Pre-check: samples must exist for MCX phase
            if not samples_dir.exists():
                logger.error(
                    "Samples directory not found: %s", samples_dir
                )
                logger.error("Run --phase de first to generate DE samples.")
                mcx_ok = False
            else:
                try:
                    run_mcx_pipeline(
                        config=config,
                        samples_dir=samples_dir,
                        projection_only=args.mcx_projection_only,
                        max_workers=args.mcx_max_workers,
                        no_skip=args.mcx_no_skip,
                    )
                except Exception as e:
                    logger.error("MCX phase failed: %s", e)
                    mcx_ok = False

    if args.phase == "post":
        post_ok = run_post_phase(config)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete (%.1fs)", elapsed)
    if args.phase in ("de", "all"):
        logger.info("  DE phase:  %s", "OK" if de_ok else "FAILED")
    if args.phase in ("mcx", "all"):
        logger.info("  MCX phase: %s", "OK" if mcx_ok else "FAILED")
    if args.phase == "post":
        logger.info("  POST phase: %s", "OK" if post_ok else "FAILED")
    logger.info("  Samples:   data/%s/samples/", experiment_name)
    logger.info("=" * 60)

    # Exit code
    phase_ok = de_ok
    if args.phase in ("mcx", "all"):
        phase_ok = phase_ok and mcx_ok
    if args.phase == "post":
        phase_ok = post_ok

    if phase_ok:
        logger.info("All phases succeeded.")
        sys.exit(0)
    else:
        logger.warning("Some phases failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
