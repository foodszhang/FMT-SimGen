"""MCX pipeline entry point for FMT-SimGen.

Handles Steps 2m (source config) → 3m (MCX simulate) → 4m (projection).

All configuration comes from the passed-in ``config`` dict — this module
never reads ``default.yaml`` or any other config file. The sole exception
is that ``output/shared/view_config.json`` is used as a fallback for the
``view_config`` key when ``config["view_config"]`` is not populated
(backward compatibility for direct ``run_mcx_pipeline.py`` CLI usage).

For ``run_all.py`` usage: always pass ``config["view_config"]`` already
merged from ``view_config.json`` so DE and MCX use identical camera params.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from fmt_simgen.frame_contract import TRUNK_OFFSET_ATLAS_MM, VOLUME_CENTER_WORLD, VOXEL_SIZE_MM
from fmt_simgen.mcx_config import generate_mcx_config
from fmt_simgen.mcx_projection import project_sample
from fmt_simgen.mcx_runner import detect_mcx_executable, run_mcx_single
from fmt_simgen.view_config import TurntableCamera

logger = logging.getLogger(__name__)


def _discover_samples(samples_dir: Path) -> dict[str, dict]:
    """Discover sample status and return structured info.

    Returns
    -------
    dict[str, dict]
        Mapping from sample_id to dict with keys:
        ``has_tumor_params``, ``has_mcx_json``, ``has_source_bin``,
        ``has_jnii``, ``has_proj``, ``sample_dir``, ``session_id``.
    """
    samples = {}
    for d in sorted(samples_dir.glob("sample_*")):
        if not d.is_dir() or d.name.startswith("."):
            continue
        sid = d.name
        jnii_files = list(d.glob("*.jnii"))
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


def _generate_sources_single(
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


def _run_mcx_single_sample(
    sample_dir: Path,
    mcx_exec: str,
) -> tuple[str, bool, str]:
    """Run MCX simulation for a single sample."""
    sample_id = sample_dir.name
    try:
        output = run_mcx_single(sample_dir, mcx_exec=mcx_exec)
        return sample_id, True, output
    except Exception as e:
        logger.warning("Sample %s MCX failed: %s", sample_id, e)
        return sample_id, False, str(e)


def _run_projection_single(
    sample_dir: Path,
    camera: TurntableCamera,
    skip_existing: bool = True,
    voxel_size_mm: float = VOXEL_SIZE_MM,
    volume_center_world: tuple[float, float, float] | None = None,
) -> tuple[str, bool, str]:
    """Run projection for a single sample."""
    if volume_center_world is None:
        volume_center_world = tuple(VOLUME_CENTER_WORLD)
    sample_id = sample_dir.name
    try:
        proj_path = project_sample(
            sample_dir,
            camera,
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


def run_mcx_pipeline(
    config: dict,
    samples_dir: Path,
    projection_only: bool = False,
    max_workers: int = 1,
    no_skip: bool = False,
) -> None:
    """Run the MCX (3D fluence simulation + projection) pipeline.

    All configuration comes from ``config`` which is ALREADY LOADED.
    This function does NOT read ``default.yaml`` or any config file.

    Parameters
    ----------
    config : dict
        Full configuration dictionary. Must contain:
        - ``config["mcx"]``: MCX section (voxel_size_mm, trunk_offset_mm, etc.)
        - ``config["view_config"]``: view config for TurntableCamera.
          If not present, falls back to ``output/shared/view_config.json``.
    samples_dir : Path
        Root directory containing ``sample_*`` subdirectories.
        Computed by the caller using ``derive_samples_dir(config)``.
    projection_only : bool, default False
        If True, skip MCX simulation and only run projection.
    max_workers : int, default 1
        Number of parallel projection workers. MCX simulation always runs
        sequentially (GPU-bound).
    no_skip : bool, default False
        If True, regenerate proj.npz even if it already exists.

    Raises
    ------
    FileNotFoundError
        If ``samples_dir`` does not exist.
        If ``view_config`` is missing from config AND
        ``output/shared/view_config.json`` does not exist.
        If ``output/shared/mcx_volume_trunk.bin`` does not exist (simulation phase).
    RuntimeError
        If MCX executable is not found and simulation phase is required.
    """
    samples_dir = Path(samples_dir)

    # ── Pre-checks ───────────────────────────────────────────────────────────
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"samples_dir not found: {samples_dir}. "
            "Run DE pipeline first or check --phase."
        )

    # Derive project root for shared artifacts path (only used for fallback)
    project_root = Path(__file__).parent.parent.parent
    shared_dir = project_root / "output" / "shared"

    # view_config: use config["view_config"] if present, else fallback to JSON
    if "view_config" in config and config["view_config"].get("angles"):
        view_cfg = config["view_config"]
    else:
        view_json_path = shared_dir / "view_config.json"
        if not view_json_path.exists():
            raise FileNotFoundError(
                f"view_config not in config and {view_json_path} not found. "
                "Run step0g_view_config.py or pass view_config in config dict."
            )
        with open(view_json_path, "r") as f:
            view_cfg = json.load(f)

    camera = TurntableCamera(view_cfg)
    mcx_cfg = config.get("mcx", {})
    voxel_size_mm = mcx_cfg.get("voxel_size_mm", VOXEL_SIZE_MM)
    volume_center_world = tuple(VOLUME_CENTER_WORLD)

    logger.info(
        "run_mcx_pipeline: samples_dir=%s, projection_only=%s, max_workers=%d",
        samples_dir, projection_only, max_workers
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

    # MCX volume file pre-check (only needed for simulation phase)
    if not projection_only:
        mcx_volume_bin = shared_dir / "mcx_volume_trunk.bin"
        if not mcx_volume_bin.exists():
            raise FileNotFoundError(
                f"mcx_volume_trunk.bin not found at {mcx_volume_bin}. "
                "Run step0f_mcx_volume.py first."
            )

    # Discover samples
    sample_info = _discover_samples(samples_dir)
    if not sample_info:
        raise FileNotFoundError(f"No sample directories found in {samples_dir}")
    logger.info("Found %d samples", len(sample_info))

    # ── Phase 2m + 3m: Source config + MCX simulation ─────────────────────────
    mcx_exec: str | None = None
    mcx_results: list[tuple[str, bool, str]] = []

    if not projection_only:
        # Pre-check MCX executable
        try:
            mcx_exec, _ = detect_mcx_executable()
        except RuntimeError as e:
            raise RuntimeError(f"MCX executable not found: {e}") from e

        # Phase 2m: Generate source configs for samples missing them
        samples_needing_sources = [
            (sid, info)
            for sid, info in sample_info.items()
            if not info["has_mcx_json"] or not info["has_source_bin"]
        ]
        if samples_needing_sources:
            logger.info(
                "Phase 2m (Source config): %d samples need MCX config",
                len(samples_needing_sources),
            )
            t0_src = time.time()
            src_results: list[tuple[str, bool, str]] = []
            for sid, info in samples_needing_sources:
                _, success, msg = _generate_sources_single(
                    info["sample_dir"], sid, mcx_cfg
                )
                src_results.append((sid, success, msg))
                logger.info("  Src %s: %s — %s", sid, "OK" if success else "FAIL", msg)
            elapsed_src = time.time() - t0_src
            src_ok = sum(1 for r in src_results if r[1])
            src_fail = sum(1 for r in src_results if not r[1])
            logger.info(
                "Phase 2m complete: %d succeeded, %d failed (%.1fs)",
                src_ok, src_fail, elapsed_src,
            )
            # Refresh sample info
            sample_info = _discover_samples(samples_dir)

        # Phase 3m: MCX simulation for samples missing .jnii
        samples_needing_mcx = [
            (sid, info)
            for sid, info in sample_info.items()
            if not info["has_jnii"]
        ]
        if samples_needing_mcx:
            logger.info(
                "Phase 3m (MCX simulation): %d samples need .jnii",
                len(samples_needing_mcx),
            )
            t0_mcx = time.time()
            for sid, info in samples_needing_mcx:
                _, success, msg = _run_mcx_single_sample(info["sample_dir"], mcx_exec)
                mcx_results.append((sid, success, msg))
                logger.info("  MCX %s: %s — %s", sid, "OK" if success else "FAIL", msg)
            elapsed_mcx = time.time() - t0_mcx
            mcx_ok = sum(1 for r in mcx_results if r[1])
            mcx_fail = sum(1 for r in mcx_results if not r[1])
            logger.info(
                "Phase 3m complete: %d succeeded, %d failed (%.1fs)",
                mcx_ok, mcx_fail, elapsed_mcx,
            )
            # Refresh sample info after MCX simulation to pick up new .jnii files
            sample_info = _discover_samples(samples_dir)
        else:
            logger.info("Phase 3m (MCX simulation): all .jnii files exist, skipping")
            # Refresh to confirm current state
            sample_info = _discover_samples(samples_dir)

    # ── Phase 4m: Projection ─────────────────────────────────────────────────
    skip_existing = not no_skip
    samples_needing_proj = [
        (sid, info)
        for sid, info in sample_info.items()
        if not info["has_proj"] or not skip_existing
    ]
    if samples_needing_proj:
        logger.info(
            "Phase 4m (Projection): %d samples need proj.npz",
            len(samples_needing_proj),
        )
        t0_proj = time.time()
        proj_results: list[tuple[str, bool, str]] = []

        if max_workers == 1:
            for sid, info in samples_needing_proj:
                _, success, msg = _run_projection_single(
                    info["sample_dir"],
                    camera,
                    skip_existing=skip_existing,
                    voxel_size_mm=voxel_size_mm,
                    volume_center_world=volume_center_world,
                )
                proj_results.append((sid, success, msg))
                logger.info("  Proj %s: %s — %s", sid, "OK" if success else "FAIL", msg)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for sid, info in samples_needing_proj:
                    fut = executor.submit(
                        _run_projection_single,
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
                    logger.info("  Proj %s: %s — %s", sid, "OK" if success else "FAIL", msg)

        elapsed_proj = time.time() - t0_proj
        proj_ok = sum(1 for r in proj_results if r[1])
        proj_fail = sum(1 for r in proj_results if not r[1])
        logger.info(
            "Phase 4m complete: %d succeeded, %d failed (%.1fs)",
            proj_ok, proj_fail, elapsed_proj,
        )
    else:
        logger.info("Phase 4m (Projection): all proj.npz exist, skipping")
        proj_results = []

    # Refresh sample info to reflect final state before summary
    sample_info = _discover_samples(samples_dir)

    # ── Summary statistics ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MCX pipeline summary")
    logger.info("=" * 60)

    total = len(sample_info)
    has_proj_count = sum(1 for s in sample_info.values() if s["has_proj"])
    has_jnii_count = sum(1 for s in sample_info.values() if s["has_jnii"])
    logger.info("Total samples:   %d", total)
    logger.info("Have .jnii:      %d", has_jnii_count)
    logger.info("Have proj.npz:   %d", has_proj_count)

    if mcx_results:
        mcx_ok = sum(1 for r in mcx_results if r[1])
        mcx_fail = sum(1 for r in mcx_results if not r[1])
        logger.info("MCX simulation:   %d ok, %d failed", mcx_ok, mcx_fail)

    if proj_results:
        proj_ok = sum(1 for r in proj_results if r[1])
        proj_fail = sum(1 for r in proj_results if not r[1])
        logger.info("Projection:      %d ok, %d failed", proj_ok, proj_fail)

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

    # Raise on failures
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
        raise RuntimeError(f"MCX pipeline failed: {failed} sample(s) had errors")

    logger.info("MCX pipeline succeeded.")
