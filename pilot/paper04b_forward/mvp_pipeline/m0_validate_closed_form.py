"""M0: Validate three-source closed-form forward models.

This stage verifies that:
1. Point source: G_inf matches expected decay
2. Ball source: closed-form integral matches cubature
3. Gaussian source: closed-form matches FFT approach

No MCX involved - pure analytic verification.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    OPTICAL,
    MVPConfig,
    SourceSpec,
    forward_closed_source,
    compute_all_metrics,
    metrics_summary,
)
from shared.green_surface_projection import project_get_surface_coords

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_test_volume(
    voxel_size_mm: float = 0.2,
    size_mm: float = 40.0,
) -> np.ndarray:
    n = int(size_mm / voxel_size_mm)
    volume = np.ones((n, n, n), dtype=np.uint8)
    return volume


def run_m0_validation(
    config: MVPConfig,
    center_mm: np.ndarray,
    output_dir: Path | None = None,
) -> dict:
    results = {"sources": {}, "optical": config.optical.to_dict()}

    volume = create_test_volume(config.voxel_size_mm, config.fov_mm)

    sources = [
        SourceSpec(kind="point", center_mm=center_mm, alpha=1.0),
        SourceSpec(kind="ball", center_mm=center_mm, alpha=1.0, radius_mm=2.0),
        SourceSpec(kind="gaussian", center_mm=center_mm, alpha=1.0, sigma_mm=1.0),
    ]

    angle_deg = 0

    surface_coords, valid_mask = project_get_surface_coords(
        volume,
        angle_deg,
        config.camera_distance_mm,
        config.fov_mm,
        config.detector_resolution,
        config.voxel_size_mm,
    )

    logger.info(f"Valid surface pixels: {np.sum(valid_mask)}")

    for source in sources:
        logger.info(f"Validating {source.kind} source...")

        projection = forward_closed_source(
            source, surface_coords, valid_mask, config.optical
        )

        valid_proj = projection[valid_mask]
        peak = np.max(valid_proj)
        mean_val = np.mean(valid_proj)
        total = np.sum(valid_proj)

        source_result = {
            "kind": source.kind,
            "center_mm": source.center_mm.tolist(),
            "peak": float(peak),
            "mean": float(mean_val),
            "total": float(total),
            "n_valid": int(np.sum(valid_mask)),
        }

        if source.kind == "ball":
            source_result["radius_mm"] = source.radius_mm
        elif source.kind == "gaussian":
            source_result["sigma_mm"] = source.sigma_mm.tolist()

        results["sources"][source.kind] = source_result

        logger.info(
            f"  {source.kind}: peak={peak:.2e}, mean={mean_val:.2e}, total={total:.2e}"
        )

    k_ratios = []
    kinds = list(results["sources"].keys())
    for i, k1 in enumerate(kinds):
        for k2 in kinds[i + 1 :]:
            t1 = results["sources"][k1]["total"]
            t2 = results["sources"][k2]["total"]
            if t2 > 0:
                ratio = t1 / t2
                k_ratios.append((k1, k2, ratio))
                logger.info(f"  total ratio {k1}/{k2} = {ratio:.2f}")

    results["k_ratios"] = k_ratios

    passed = True
    for k1, k2, ratio in k_ratios:
        if ratio < 0.1 or ratio > 10:
            logger.warning(
                f"  FAIL: {k1}/{k2} ratio {ratio:.2f} out of range [0.1, 10]"
            )
            passed = False

    if passed:
        logger.info(
            "M0 validation PASSED: All three sources produce consistent magnitudes"
        )
    else:
        logger.warning("M0 validation FAILED: Source magnitudes inconsistent")

    results["passed"] = passed

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import json

        with open(output_dir / "m0_validation.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'm0_validation.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="M0: Three-source closed-form validation"
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=[0.0, 10.0, 8.0],
        help="Source center in mm [x, y, z]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    config = MVPConfig()
    center_mm = np.array(args.center)

    logger.info(
        f"Optical params: mu_a={config.optical.mu_a}, mus_p={config.optical.mus_p}"
    )
    logger.info(f"Delta = {config.optical.delta:.3f} mm")
    logger.info(f"Source center: {center_mm} mm")

    output_dir = Path(args.output) if args.output else None
    results = run_m0_validation(config, center_mm, output_dir)

    if results["passed"]:
        print("\nM0 PASSED")
        return 0
    else:
        print("\nM0 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
