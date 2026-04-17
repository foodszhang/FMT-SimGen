#!/usr/bin/env python3
"""Population-level FLT × FOV coverage statistics.

Reports what fraction of FLT sphere is visible in MCX FOV for each sample,
given a specific FOV setting. Used to validate FOV parameter changes.

Usage:
    python scripts/geometry_health_check.py \\
        --samples_dir data/uniform_1000_v2/samples \\
        --fov_mm 80 \\
        --output data/uniform_1000_v2/geometry_report_fov80.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_flt_coverage(flt_y_center: float, flt_radius: float, fov_y: tuple[float, float]) -> float:
    """Compute fraction of FLT sphere within FOV Y range.

    Parameters
    ----------
    flt_y_center : float
        World Y of FLT sphere center (mm).
    flt_radius : float
        FLT sphere radius (mm).
    fov_y : tuple[float, float]
        (fov_min, fov_max) in world Y coordinates.

    Returns
    -------
    float
        Fraction of FLT sphere diameter that falls within FOV [0, 1].
    """
    flt_bottom = flt_y_center - flt_radius
    flt_top = flt_y_center + flt_radius

    overlap_bottom = max(flt_bottom, fov_y[0])
    overlap_top = min(flt_top, fov_y[1])
    overlap = max(0.0, overlap_top - overlap_bottom)

    return overlap / (2.0 * flt_radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="FLT × FOV coverage statistics")
    parser.add_argument("--samples_dir", required=True, help="Root samples directory")
    parser.add_argument("--fov_mm", type=float, default=80.0, help="Projection FOV in mm")
    parser.add_argument("--trunk_offset", type=float, default=30.0, help="Trunk offset Y (mm)")
    parser.add_argument("--output", help="Output JSON path (optional)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        logger.error("samples_dir not found: %s", samples_dir)
        sys.exit(1)

    # FOV Y range at given offset
    half_fov = args.fov_mm / 2.0
    fov_y = (args.trunk_offset - half_fov, args.trunk_offset + half_fov)
    logger.info("FOV world Y: [%.1f, %.1f] (fov_mm=%.0f, offset=%.1f)", *fov_y, args.fov_mm, args.trunk_offset)

    # Query grid Y range (from Stage1 ROI)
    query_y = (46.38, 67.90)
    query_size = query_y[1] - query_y[0]
    overlap_q = max(0.0, min(query_y[1], fov_y[1]) - max(query_y[0], fov_y[0]))
    logger.info("Query grid Y: [%.1f, %.1f] (%.1fmm)", *query_y, query_size)
    logger.info("Query overlap with FOV: %.2fmm / %.2fmm = %.1f%%", overlap_q, query_size, 100 * overlap_q / query_size)

    # Collect FLT positions
    tumor_files = sorted(samples_dir.glob("sample_*/tumor_params.json"))
    logger.info("Found %d tumor_params files", len(tumor_files))

    all_coverage = []
    sample_results = []

    for tf in tumor_files:
        with open(tf) as f:
            tp = json.load(f)

        sample_id = tf.parent.name
        foci = tp.get("foci", [])
        if not foci:
            sample_results.append({"sample_id": sample_id, "n_foci": 0, "coverage": None})
            continue

        coverage_per_focus = []
        for focus in foci:
            center = focus["center"]
            flt_y = center[1]
            flt_radius = focus.get("radius", 3.0) * 3  # 3-sigma
            cov = compute_flt_coverage(flt_y, flt_radius, fov_y)
            coverage_per_focus.append(cov)

        mean_cov = float(np.mean(coverage_per_focus))
        all_coverage.append(mean_cov)
        sample_results.append({
            "sample_id": sample_id,
            "n_foci": len(foci),
            "coverage": mean_cov,
            "foci_coverage": [float(c) for c in coverage_per_focus],
        })

    all_coverage = np.array(all_coverage)

    logger.info("\n" + "=" * 60)
    logger.info("FLT FOV Coverage Summary (fov_mm=%.0f)", args.fov_mm)
    logger.info("=" * 60)
    logger.info("Samples:          %d", len(all_coverage))
    logger.info("Mean coverage:   %.1f%%", 100 * all_coverage.mean())
    logger.info("Median coverage: %.1f%%", 100 * np.median(all_coverage))
    logger.info("Min coverage:    %.1f%%", 100 * all_coverage.min())
    logger.info("Max coverage:    %.1f%%", 100 * all_coverage.max())
    logger.info("Std coverage:    %.1f%%", 100 * all_coverage.std())
    logger.info("Samples < 50%%:   %d (%.1f%%)", (all_coverage < 0.5).sum(), 100 * (all_coverage < 0.5).mean())
    logger.info("Samples < 80%%:   %d (%.1f%%)", (all_coverage < 0.8).sum(), 100 * (all_coverage < 0.8).mean())
    logger.info("Samples >= 90%%:  %d (%.1f%%)", (all_coverage >= 0.9).sum(), 100 * (all_coverage >= 0.9).mean())

    report = {
        "fov_mm": args.fov_mm,
        "trunk_offset": args.trunk_offset,
        "fov_y": list(fov_y),
        "query_y": list(query_y),
        "query_overlap_fraction": float(overlap_q / query_size),
        "coverage": {
            "mean": float(all_coverage.mean()),
            "median": float(np.median(all_coverage)),
            "min": float(all_coverage.min()),
            "max": float(all_coverage.max()),
            "std": float(all_coverage.std()),
        },
        "fraction_below_50pct": float((all_coverage < 0.5).mean()),
        "fraction_below_80pct": float((all_coverage < 0.8).mean()),
        "fraction_above_90pct": float((all_coverage >= 0.9).mean()),
        "samples": sample_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to %s", args.output)

    # Pass/fail
    if all_coverage.mean() >= 0.90:
        logger.info("\nPASS: Mean coverage %.1f%% >= 90%%", 100 * all_coverage.mean())
    elif all_coverage.mean() >= 0.80:
        logger.info("\nWARN: Mean coverage %.1f%% >= 80%% (target is 90%%)", 100 * all_coverage.mean())
    else:
        logger.warning("\nFAIL: Mean coverage %.1f%% < 80%%", 100 * all_coverage.mean())


if __name__ == "__main__":
    main()
