#!/usr/bin/env python3
"""
run_mcx_pipeline.py — Standalone MCX channel entry point for FMT-SimGen.

Orchestrates Steps 2m (source config) → 3m (MCX simulate) → 4m (projection)
on existing dataset samples. Can be run independently or via run_all.py --enable_mcx.

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
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD, TRUNK_OFFSET_ATLAS_MM
from fmt_simgen.mcx_projection import project_sample, load_jnii_volume
from fmt_simgen.mcx_runner import detect_mcx_executable, run_mcx_single
from fmt_simgen.mcx_config import generate_mcx_config
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOXEL_SIZE_MM


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_shared_config() -> dict:
    """Load shared config (mcx section from default.yaml)."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_view_config() -> dict:
    """Load view configuration from output/shared/view_config.json."""
    view_path = Path("output/shared/view_config.json")
    if not view_path.exists():
        raise FileNotFoundError(
            f"view_config.json not found at {view_path}. "
            "Run step0g_view_config.py first."
        )
    with open(view_path, "r") as f:
        return json.load(f)


def discover_samples(samples_dir: Path) -> dict[str, dict]:
    """Discover sample status and return structured info.

    Returns
    -------
    dict[str, dict]
        Mapping from sample_id to dict with keys:
        has_tumor_params, has_mcx_json, has_source_bin, has_jnii, has_proj, sample_dir
    """
    samples = {}
    for d in sorted(samples_dir.glob("sample_*")):
        if not d.is_dir() or d.name.startswith("."):
            continue
        sid = d.name
        jnii_files = list(d.glob("*.jnii"))
        # Determine session ID from JSON if available
        json_files = list(d.glob("*.json"))
        session_id = sid
        if json_files:
            try:
                with open(json_files[0]) as f:
                    cfg = json.load(f)
                session_id = cfg.get("Session", {}).get("ID", sid)
            except Exception:
                pass
        samples[sid] = {
            "sample_dir": d,
            "has_tumor_params": (d / "tumor_params.json").exists(),
            "has_mcx_json": bool(json_files),
            "has_source_bin": (d / f"source-{sid}.bin").exists(),
            "has_jnii": bool(jnii_files),
            "has_proj": (d / "proj.npz").exists(),
            "session_id": session_id,
        }
    return samples


def generate_sources_single(
    sample_dir: Path,
    sample_id: str,
    mcx_cfg: dict,
) -> tuple[str, bool, str]:
    """Generate MCX JSON config and source binary for a single sample."""
    tumor_params_path = sample_dir / "tumor_params.json"
    if not tumor_params_path.exists():
        return sample_id, False, "no tumor_params.json"
    try:
        with open(tumor_params_path, "r") as f:
            tumor_params = json.load(f)
        json_path = generate_mcx_config(
            sample_id=sample_id,
            tumor_params=tumor_params,
            mcx_config=mcx_cfg,
            output_dir=sample_dir,
        )
        return sample_id, True, json_path
    except Exception as e:
        return sample_id, False, str(e)


def run_projection_single(
    sample_dir: Path,
    camera: TurntableCamera,
    skip_existing: bool = True,
    voxel_size_mm: float = VOXEL_SIZE_MM,
    volume_center_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[str, bool, str]:
    """Run projection for a single sample (for parallel execution)."""
    sample_id = sample_dir.name
    try:
        proj_path = project_sample(
            sample_dir, camera,
            skip_existing=skip_existing,
            voxel_size_mm=voxel_size_mm,
            volume_center_world=volume_center_world,
        )
        return sample_id, True, str(proj_path)
    except FileNotFoundError as e:
        return sample_id, False, str(e)
    except Exception as e:
        logger.warning("Sample %s projection failed: %s", sample_id, e)
        return sample_id, False, str(e)


def run_mcx_single_sample(
    sample_dir: Path,
    mcx_exec: str,
) -> tuple[str, bool, str]:
    """Run MCX simulation for a single sample (for parallel execution)."""
    sample_id = sample_dir.name
    try:
        output = run_mcx_single(sample_dir, mcx_exec=mcx_exec)
        return sample_id, True, output
    except Exception as e:
        logger.warning("Sample %s MCX failed: %s", sample_id, e)
        return sample_id, False, str(e)


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
        help="Re-run MCX simulation even if .jnii already exists",
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

    # Load configs
    shared_cfg = load_shared_config()
    view_cfg = load_view_config()
    camera = TurntableCamera(view_cfg)
    voxel_size_mm = shared_cfg.get("mcx", {}).get("voxel_size_mm", VOXEL_SIZE_MM)

    # Compute MCX trunk volume center in world coordinates (mm)
    # The reference projection centers at (nx/2, ny/2, nz/2) in voxel space,
    # converts to world mm, then SUBTRACTS volume_center_world.
    # The trunk_offset_mm=[0, 34, 0] shifts the Y center by 34mm.
    # X and Z are centered at 0 in the atlas, so cx=cz=0.
    # Y: center = (200/2)*0.2 + 34 = 54.0mm, but we subtract the center
    #    so that world Y is centered at 0 for projection.
    # The working centering that produced valid projections was (0, 30, 0),
    # which means the centering is relative to the ATLAS center (not the
    # trunk center). The trunk_offset IS the centering shift.
    mcx_cfg = shared_cfg.get("mcx", {})
    trunk_offset_y = mcx_cfg.get("trunk_offset_mm", TRUNK_OFFSET_ATLAS_MM.tolist())[1]
    volume_center_world = tuple(VOLUME_CENTER_WORLD)

    logger.info(
        "run_mcx_pipeline: samples_dir=%s, projection_only=%s, force_mcx=%s, max_workers=%d",
        samples_dir,
        args.projection_only,
        args.force_mcx,
        args.max_workers,
    )
    logger.info(
        "  Camera: pose=%s, distance=%.0fmm, fov=%.0fmm, detector=%s, angles=%s",
        camera.pose,
        camera.camera_distance_mm,
        camera.fov_mm,
        camera.detector_resolution,
        camera.angles,
    )
    logger.info(
        "  MCX volume center world: (%.2f, %.2f, %.2f) mm",
        volume_center_world[0], volume_center_world[1], volume_center_world[2],
    )

    # Discover samples
    sample_info = discover_samples(samples_dir)
    if not sample_info:
        sys.stderr.write(f"Error: no sample directories found in {samples_dir}\n")
        sys.exit(2)
    logger.info("Found %d samples", len(sample_info))

    # MCX simulation phase (Step 3m)
    mcx_exec = None
    mcx_results: list[tuple[str, bool, str]] = []
    if not args.projection_only:
        try:
            mcx_exec, gpu_supported = detect_mcx_executable()
        except RuntimeError as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.exit(1)

        # Phase 2m: Generate source configs for samples missing .json or .bin
        samples_needing_sources = [
            (sid, info)
            for sid, info in sample_info.items()
            if not info["has_mcx_json"] or not info["has_source_bin"]
        ]
        if samples_needing_sources:
            logger.info(
                "Phase 2m (Source config): %d samples need MCX config", len(samples_needing_sources)
            )
            t0_src = time.time()
            src_results: list[tuple[str, bool, str]] = []
            for sid, info in samples_needing_sources:
                _, success, msg = generate_sources_single(
                    info["sample_dir"], sid, shared_cfg.get("mcx", {})
                )
                src_results.append((sid, success, msg))
                status = "OK" if success else "FAIL"
                logger.info("  Src %s: %s — %s", sid, status, msg)
            elapsed_src = time.time() - t0_src
            src_ok = sum(1 for r in src_results if r[1])
            src_fail = sum(1 for r in src_results if not r[1])
            logger.info(
                "Phase 2m complete: %d succeeded, %d failed (%.1fs)",
                src_ok,
                src_fail,
                elapsed_src,
            )
            # Refresh sample info to pick up generated configs
            sample_info = discover_samples(samples_dir)

        samples_needing_mcx = [
            (sid, info)
            for sid, info in sample_info.items()
            if not info["has_jnii"] or args.force_mcx
        ]
        if samples_needing_mcx:
            logger.info(
                "Phase 3m (MCX simulation): %d samples need .jnii", len(samples_needing_mcx)
            )
            t0_mcx = time.time()

            # MCX simulations must run sequentially (GPU-bound)
            for sid, info in samples_needing_mcx:
                _, success, msg = run_mcx_single_sample(info["sample_dir"], mcx_exec)
                mcx_results.append((sid, success, msg))
                status = "OK" if success else "FAIL"
                logger.info("  MCX %s: %s — %s", sid, status, msg)

            elapsed_mcx = time.time() - t0_mcx
            mcx_ok = sum(1 for r in mcx_results if r[1])
            mcx_fail = sum(1 for r in mcx_results if not r[1])
            logger.info(
                "Phase 3m complete: %d succeeded, %d failed (%.1fs)",
                mcx_ok,
                mcx_fail,
                elapsed_mcx,
            )
        else:
            logger.info("Phase 3m (MCX simulation): all .jnii files exist, skipping")

    # Projection phase (Step 4m)
    samples_needing_proj = [
        (sid, info)
        for sid, info in sample_info.items()
        if not info["has_proj"] or not skip_existing
    ]
    if samples_needing_proj:
        logger.info(
            "Phase 4m (Projection): %d samples need proj.npz", len(samples_needing_proj)
        )
        t0_proj = time.time()
        proj_results: list[tuple[str, bool, str]] = []

        if args.max_workers == 1:
            for sid, info in samples_needing_proj:
                _, success, msg = run_projection_single(
                    info["sample_dir"], camera,
                    skip_existing=skip_existing,
                    voxel_size_mm=voxel_size_mm,
                    volume_center_world=volume_center_world,
                )
                proj_results.append((sid, success, msg))
                status = "OK" if success else "FAIL"
                logger.info("  Proj %s: %s — %s", sid, status, msg)
        else:
            futures = {}
            for sid, info in samples_needing_proj:
                fut = ProcessPoolExecutor(max_workers=args.max_workers).submit(
                    run_projection_single,
                    info["sample_dir"],
                    camera,
                    skip_existing,
                    voxel_size_mm,
                    volume_center_world,
                )
                futures[fut] = sid
            for fut in as_completed(futures):
                sid = futures[fut]
                _, success, msg = fut.result()
                proj_results.append((sid, success, msg))
                status = "OK" if success else "FAIL"
                logger.info("  Proj %s: %s — %s", sid, status, msg)

        elapsed_proj = time.time() - t0_proj
        proj_ok = sum(1 for r in proj_results if r[1])
        proj_fail = sum(1 for r in proj_results if not r[1])
        logger.info(
            "Phase 4m complete: %d succeeded, %d failed (%.1fs)",
            proj_ok,
            proj_fail,
            elapsed_proj,
        )
    else:
        logger.info("Phase 4m (Projection): all proj.npz exist, skipping")
        proj_results = []

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("MCX pipeline summary")
    logger.info("=" * 60)

    total = len(sample_info)
    has_proj_count = sum(1 for s in sample_info.values() if s["has_proj"])
    has_jnii_count = sum(1 for s in sample_info.values() if s["has_jnii"])
    logger.info("Total samples:        %d", total)
    logger.info("Have .jnii:          %d", has_jnii_count)
    logger.info("Have proj.npz:       %d", has_proj_count)

    if mcx_results:
        mcx_ok = sum(1 for r in mcx_results if r[1])
        mcx_fail = sum(1 for r in mcx_results if not r[1])
        logger.info("MCX simulation:       %d ok, %d failed", mcx_ok, mcx_fail)

    if proj_results:
        proj_ok = sum(1 for r in proj_results if r[1])
        proj_fail = sum(1 for r in proj_results if not r[1])
        logger.info("Projection:           %d ok, %d failed", proj_ok, proj_fail)

    # Non-zero pixel stats
    nonzero_stats: list[tuple[str, float]] = []
    for sid, info in sample_info.items():
        proj_path = info["sample_dir"] / "proj.npz"
        if proj_path.exists():
            try:
                data = np.load(proj_path)
                total_nonzero = sum(np.count_nonzero(data[k]) for k in data.files)
                total_pixels = sum(data[k].size for k in data.files)
                frac = total_nonzero / max(total_pixels, 1)
                nonzero_stats.append((sid, frac))
            except Exception:
                pass

    if nonzero_stats:
        avg_frac = sum(s[1] for s in nonzero_stats) / len(nonzero_stats)
        logger.info(
            "Non-zero pixel fraction (proj.npz): avg=%.3f, min=%.3f, max=%.3f",
            avg_frac,
            min(s[1] for s in nonzero_stats),
            max(s[1] for s in nonzero_stats),
        )

    # Determine exit code
    failed = 0
    if mcx_results:
        failed += sum(1 for r in mcx_results if not r[1])
    if proj_results:
        failed += sum(1 for r in proj_results if not r[1])

    if failed > 0:
        logger.warning("\nFailed samples:")
        for r in mcx_results:
            if not r[1]:
                logger.warning("  MCX   %s: %s", r[0], r[2])
        for r in proj_results:
            if not r[1]:
                logger.warning("  Proj  %s: %s", r[0], r[2])

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
