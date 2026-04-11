#!/usr/bin/env python3
"""
Step 4m: Generate MCX multi-angle projections from .jnii fluence volumes.

Reads .jnii files from each sample directory and generates:
    - proj.npz: multi-angle 2D projections [H×W] per angle

Usage:
    python scripts/step4m_mcx_projection.py --samples_dir data/gaussian_1000/samples

Output per sample:
    {sample_id}/proj.npz  -- dict of angle → projection array

Exit codes:
    0  -- all samples succeeded or already have outputs
    1  -- MCX .jnii not found for any sample
    2  -- samples_dir not found
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.mcx_projection import project_sample, load_jnii_volume
from fmt_simgen.view_config import TurntableCamera
import yaml


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


def process_single_sample(
    sample_dir: Path,
    camera: TurntableCamera,
    trunk_offset_mm: np.ndarray,
    voxel_size_mm: float,
    skip_existing: bool,
) -> tuple[str, bool, str]:
    """Process a single sample (for parallel execution).

    Returns
    -------
    tuple[str, bool, str]
        (sample_id, success, message)
    """
    sample_id = sample_dir.name
    try:
        proj_path = project_sample(
            sample_dir,
            camera,
            trunk_offset_mm,
            voxel_size_mm,
            skip_existing=skip_existing,
        )
        return sample_id, True, str(proj_path)
    except FileNotFoundError as e:
        return sample_id, False, str(e)
    except Exception as e:
        logger.warning("Sample %s failed: %s", sample_id, e)
        return sample_id, False, str(e)


def visualize_sample(
    sample_dir: Path,
    camera: TurntableCamera,
    output_path: Optional[Path] = None,
) -> None:
    """Generate a 7-angle projection visualization for one sample."""
    import matplotlib.pyplot as plt

    jnii_files = list(sample_dir.glob("*.jnii"))
    if not jnii_files:
        logger.warning("No .jnii for visualization in %s", sample_dir.name)
        return

    fluence = load_jnii_volume(jnii_files[0])
    origin = np.array([0.0, 30.0, 0.0])
    projections = {}

    for angle in camera.angles:
        proj = camera.project_volume(fluence, angle, 0.2, origin)
        projections[str(angle)] = proj

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, angle in enumerate(camera.angles):
        ax = axes[idx]
        proj = projections[str(angle)]
        im = ax.imshow(proj, cmap="hot", interpolation="nearest")
        ax.set_title(f"{angle}°")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Remove the 8th subplot
    axes[7].axis("off")

    title = f"MCX Projections — {sample_dir.name}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path = output_path or (sample_dir / "proj_vis.png")
    fig.savefig(out_path, dpi=150)
    logger.info("Saved visualization: %s", out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4m: Generate MCX projections from .jnii fluence volumes"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Root directory containing sample subdirectories",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
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
        "--visualize",
        action="store_true",
        help="Generate proj_vis.png for each sample",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
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
    mcx_cfg = shared_cfg.get("mcx", {})
    trunk_offset_mm = np.array(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]))
    voxel_size_mm = float(mcx_cfg.get("voxel_size_mm", 0.2))

    view_cfg = load_view_config()
    camera = TurntableCamera(view_cfg)

    logger.info(
        "Step 4m: MCX projection, samples=%s, max_workers=%d, skip_existing=%s",
        samples_dir,
        args.max_workers,
        skip_existing,
    )
    logger.info(
        "  Camera: pose=%s, distance=%.0fmm, fov=%.0fmm, detector=%s, angles=%s",
        camera.pose,
        camera.camera_distance_mm,
        camera.fov_mm,
        camera.detector_resolution,
        camera.angles,
    )

    # Find all sample directories
    sample_dirs = sorted(
        d for d in samples_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not sample_dirs:
        sys.stderr.write(f"Error: no sample directories found in {samples_dir}\n")
        sys.exit(2)

    # Filter to samples that have .jnii files
    sample_dirs = [d for d in sample_dirs if list(d.glob("*.jnii"))]
    logger.info("Found %d samples with .jnii files", len(sample_dirs))

    # Process
    t0 = time.time()
    results: list[tuple[str, bool, str]] = []

    if args.max_workers == 1:
        # Sequential
        for sample_dir in sample_dirs:
            r = process_single_sample(
                sample_dir, camera, trunk_offset_mm, voxel_size_mm, skip_existing
            )
            results.append(r)
            status = "OK" if r[1] else "FAIL"
            logger.info("  %s: %s — %s", r[0], status, r[2] if not r[1] else r[2])
    else:
        # Parallel
        futures = {}
        for sample_dir in sample_dirs:
            fut = ProcessPoolExecutor(max_workers=args.max_workers).submit(
                process_single_sample,
                sample_dir,
                camera,
                trunk_offset_mm,
                voxel_size_mm,
                skip_existing,
            )
            futures[fut] = sample_dir.name

        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "OK" if r[1] else "FAIL"
            logger.info("  %s: %s — %s", r[0], status, r[2] if not r[1] else r[2])

    elapsed = time.time() - t0

    # Summary
    success = sum(1 for r in results if r[1])
    failed = sum(1 for r in results if not r[1])
    skipped = sum(1 for r in results if r[1] and "already exists" in r[2])

    logger.info(
        "\nStep 4m complete: %d succeeded, %d failed, %d skipped (%.1fs)",
        success,
        failed,
        skipped,
        elapsed,
    )

    if failed > 0:
        logger.warning("Failed samples:")
        for r in results:
            if not r[1]:
                logger.warning("  %s: %s", r[0], r[2])

    # Per-sample non-zero stats
    nonzero_stats: list[tuple[str, float]] = []
    for sample_dir in sample_dirs:
        proj_path = sample_dir / "proj.npz"
        if proj_path.exists():
            try:
                data = np.load(proj_path)
                total_nonzero = sum(np.count_nonzero(data[k]) for k in data.files)
                total_pixels = sum(data[k].size for k in data.files)
                frac = total_nonzero / max(total_pixels, 1)
                nonzero_stats.append((sample_dir.name, frac))
            except Exception:
                pass

    if nonzero_stats:
        avg_frac = sum(s[1] for s in nonzero_stats) / len(nonzero_stats)
        logger.info(
            "Non-zero pixel fraction: avg=%.3f, min=%.3f, max=%.3f",
            avg_frac,
            min(s[1] for s in nonzero_stats),
            max(s[1] for s in nonzero_stats),
        )

    # Visualization for first 3 samples
    if args.visualize:
        logger.info("Generating visualizations for first 3 samples...")
        for sample_dir in sample_dirs[:3]:
            try:
                visualize_sample(sample_dir, camera)
            except Exception as e:
                logger.warning("Visualization failed for %s: %s", sample_dir.name, e)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
