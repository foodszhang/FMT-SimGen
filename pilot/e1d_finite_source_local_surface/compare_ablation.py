#!/usr/bin/env python3
"""Compare ablation study for E1d.

Compares:
- Sampling levels: 1-point, 7-point, 19-point, 27-point
- Kernel types: gaussian_psf, green_infinite, green_halfspace
- ROI cutoff ratios
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict
import numpy as np
import yaml

from source_models import create_source, GaussianEllipsoidSource
from local_surface_renderer import render_local_surface_response
from generate_gt import compute_metrics

logger = logging.getLogger(__name__)


def ablation_sampling(
    source_config: dict,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    sampling_levels: list,
    z_surface: float = 10.0,
) -> dict:
    """Ablation study on source sampling levels.

    Uses highest sampling level as reference.

    Args:
        source_config: source configuration
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        sampling_levels: list of sampling levels to compare
        z_surface: surface z coordinate

    Returns:
        results dict with metrics for each sampling level
    """
    source = create_source(source_config)
    kernel_type = "green_halfspace"

    gt_level = "27-point"
    gt_image = render_local_surface_response(
        source=source,
        kernel_type=kernel_type,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level=gt_level,
        z_surface=z_surface,
        use_roi=False,
    )

    results = {}

    for level in sampling_levels:
        start_time = time.time()

        pred_image = render_local_surface_response(
            source=source,
            kernel_type=kernel_type,
            tissue_params=tissue_params,
            image_size=image_size,
            pixel_size_mm=pixel_size_mm,
            sampling_level=level,
            z_surface=z_surface,
            use_roi=False,
        )

        render_time_ms = (time.time() - start_time) * 1000

        metrics = compute_metrics(pred_image, gt_image)
        metrics["render_time_ms"] = render_time_ms

        results[level] = metrics
        logger.info(f"  {level}: NCC={metrics['ncc']:.4f}, time={render_time_ms:.2f}ms")

    return results


def ablation_kernel(
    source_config: dict,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    kernels: list,
    z_surface: float = 10.0,
) -> dict:
    """Ablation study on kernel types.

    Args:
        source_config: source configuration
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        kernels: list of kernel types to compare
        z_surface: surface z coordinate

    Returns:
        results dict with metrics for each kernel
    """
    source = create_source(source_config)

    gt_kernel = "green_halfspace"
    gt_sampling = "27-point"
    gt_image = render_local_surface_response(
        source=source,
        kernel_type=gt_kernel,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level=gt_sampling,
        z_surface=z_surface,
        use_roi=False,
    )

    results = {}

    for kernel in kernels:
        start_time = time.time()

        pred_image = render_local_surface_response(
            source=source,
            kernel_type=kernel,
            tissue_params=tissue_params,
            image_size=image_size,
            pixel_size_mm=pixel_size_mm,
            sampling_level="7-point",
            z_surface=z_surface,
            use_roi=False,
        )

        render_time_ms = (time.time() - start_time) * 1000

        metrics = compute_metrics(pred_image, gt_image)
        metrics["render_time_ms"] = render_time_ms

        results[kernel] = metrics
        logger.info(
            f"  {kernel}: NCC={metrics['ncc']:.4f}, FWHM ratio={metrics['fwhm_ratio']:.3f}"
        )

    return results


def ablation_cutoff(
    source_config: dict,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    cutoff_ratios: list,
    z_surface: float = 10.0,
) -> dict:
    """Ablation study on ROI cutoff ratios.

    Args:
        source_config: source configuration
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        cutoff_ratios: list of cutoff ratios to compare
        z_surface: surface z coordinate

    Returns:
        results dict with metrics for each cutoff ratio
    """
    source = create_source(source_config)
    kernel_type = "green_halfspace"
    sampling_level = "7-point"

    gt_image = render_local_surface_response(
        source=source,
        kernel_type=kernel_type,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level="27-point",
        z_surface=z_surface,
        use_roi=False,
    )

    results = {}

    for cutoff in cutoff_ratios:
        start_time = time.time()

        use_roi = cutoff > 0

        pred_image = render_local_surface_response(
            source=source,
            kernel_type=kernel_type,
            tissue_params=tissue_params,
            image_size=image_size,
            pixel_size_mm=pixel_size_mm,
            sampling_level=sampling_level,
            cutoff_ratio=cutoff,
            z_surface=z_surface,
            use_roi=use_roi,
        )

        render_time_ms = (time.time() - start_time) * 1000

        metrics = compute_metrics(pred_image, gt_image)
        metrics["render_time_ms"] = render_time_ms

        key = f"cutoff_{cutoff:.0e}" if cutoff > 0 else "no_roi"
        results[key] = metrics
        logger.info(f"  {key}: NCC={metrics['ncc']:.4f}, time={render_time_ms:.2f}ms")

    return results


def run_ablation_study(
    config: dict,
    output_dir: str,
) -> dict:
    """Run all ablation studies.

    Args:
        config: experiment configuration
        output_dir: output directory

    Returns:
        summary dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tissue_params = config["tissue"]["muscle"]
    image_size = config["rendering"]["image_size"]
    pixel_size_mm = config["rendering"]["pixel_size_mm"]
    z_surface = config["rendering"]["z_surface"]

    all_results = {}

    for ablation in config["ablation_configs"]:
        ablation_id = ablation["id"]
        logger.info(f"\nRunning ablation: {ablation_id}")

        source_config = {
            "type": ablation["source_type"],
            "center": ablation["source_center"],
        }
        if ablation["source_type"] == "gaussian":
            source_config["sigmas"] = ablation["source_sigmas"]
            source_config["alpha"] = 1.0
        elif ablation["source_type"] == "uniform":
            source_config["axes"] = ablation.get("source_axes", [1.0, 1.0, 1.0])
            source_config["alpha"] = 1.0

        if ablation_id == "ablation_sampling":
            results = ablation_sampling(
                source_config=source_config,
                tissue_params=tissue_params,
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                sampling_levels=ablation["sampling_levels"],
                z_surface=z_surface,
            )
        elif ablation_id == "ablation_kernel":
            results = ablation_kernel(
                source_config=source_config,
                tissue_params=tissue_params,
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                kernels=ablation["kernels"],
                z_surface=z_surface,
            )
        elif ablation_id == "ablation_cutoff":
            results = ablation_cutoff(
                source_config=source_config,
                tissue_params=tissue_params,
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                cutoff_ratios=ablation["cutoff_ratios"],
                z_surface=z_surface,
            )
        else:
            logger.warning(f"Unknown ablation: {ablation_id}")
            continue

        all_results[ablation_id] = results

    summary_path = output_path / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAblation summary saved: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for E1d")
    parser.add_argument(
        "--config",
        default="pilot/e1d_finite_source_local_surface/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1d_finite_source_local_surface/results/ablations",
        help="Output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_ablation_study(config, args.output)


if __name__ == "__main__":
    main()
