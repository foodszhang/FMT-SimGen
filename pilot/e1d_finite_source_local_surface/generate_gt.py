#!/usr/bin/env python3
"""GT generation for E1d experiments.

Generates two types of GT:
- GT-A: inverse-crime GT (same forward)
- GT-B: finite-source mismatch GT
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import yaml

from source_models import create_source, BaseSource
from local_surface_renderer import render_local_surface_response

logger = logging.getLogger(__name__)


def generate_inverse_crime_gt(
    source: BaseSource,
    kernel_type: str,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    sampling_level: str = "7-point",
    z_surface: float = 10.0,
) -> np.ndarray:
    """Generate GT using same forward model (inverse crime setup).

    Args:
        source: source model
        kernel_type: kernel for GT generation
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        sampling_level: source sampling level
        z_surface: surface z coordinate

    Returns:
        gt_image: 2D array [H, W]
    """
    return render_local_surface_response(
        source=source,
        kernel_type=kernel_type,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level=sampling_level,
        z_surface=z_surface,
        use_roi=False,
    )


def generate_finite_source_gt(
    source: BaseSource,
    gt_kernel_type: str,
    gt_sampling_level: str,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    z_surface: float = 10.0,
) -> np.ndarray:
    """Generate GT with higher-fidelity finite source model.

    Args:
        source: source model
        gt_kernel_type: kernel for GT (typically "green_halfspace")
        gt_sampling_level: high-fidelity sampling (e.g., "27-point")
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        z_surface: surface z coordinate

    Returns:
        gt_image: 2D array [H, W]
    """
    return render_local_surface_response(
        source=source,
        kernel_type=gt_kernel_type,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        sampling_level=gt_sampling_level,
        z_surface=z_surface,
        use_roi=False,
    )


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute forward metrics between prediction and GT.

    Args:
        pred: predicted image
        gt: ground truth image

    Returns:
        dict with NCC, peak error, FWHM ratio, RMSE
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    pred_norm = pred_flat - pred_flat.mean()
    gt_norm = gt_flat - gt_flat.mean()

    ncc = np.sum(pred_norm * gt_norm) / (
        np.sqrt(np.sum(pred_norm**2) * np.sum(gt_norm**2)) + 1e-10
    )

    pred_peak_idx = np.argmax(pred)
    gt_peak_idx = np.argmax(gt)

    pred_peak_pos = np.unravel_index(pred_peak_idx, pred.shape)
    gt_peak_pos = np.unravel_index(gt_peak_idx, gt.shape)

    peak_error = np.sqrt(
        (pred_peak_pos[0] - gt_peak_pos[0]) ** 2
        + (pred_peak_pos[1] - gt_peak_pos[1]) ** 2
    )

    def compute_fwhm(img):
        threshold = img.max() / 2
        mask = img >= threshold
        if mask.sum() == 0:
            return 1.0
        return np.sqrt(mask.sum() / np.pi)

    pred_fwhm = compute_fwhm(pred)
    gt_fwhm = compute_fwhm(gt)
    fwhm_ratio = pred_fwhm / (gt_fwhm + 1e-10)

    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    return {
        "ncc": float(ncc),
        "peak_error_px": float(peak_error),
        "fwhm_ratio": float(fwhm_ratio),
        "rmse": float(rmse),
    }


def run_gt_generation(
    config: dict,
    output_dir: str,
) -> dict:
    """Generate all GT data for E1d.

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

    results = {}

    for gt_config in config["gt_configs"]:
        gt_id = gt_config["id"]
        logger.info(f"Generating GT: {gt_id}")

        source = create_source(gt_config["source"])

        gt_type = gt_config.get("gt_type", "inverse_crime")

        if gt_type == "inverse_crime":
            gt_image = generate_inverse_crime_gt(
                source=source,
                kernel_type=gt_config["kernel"],
                tissue_params=tissue_params,
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                sampling_level=gt_config.get("sampling_level", "7-point"),
                z_surface=z_surface,
            )
        elif gt_type == "finite_source":
            gt_image = generate_finite_source_gt(
                source=source,
                gt_kernel_type=gt_config["kernel"],
                gt_sampling_level=gt_config.get("gt_sampling_level", "27-point"),
                tissue_params=tissue_params,
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                z_surface=z_surface,
            )
        else:
            raise ValueError(f"Unknown GT type: {gt_type}")

        gt_path = output_path / f"{gt_id}_gt.npz"
        np.savez(
            gt_path,
            image=gt_image,
            source_params=source.get_params(),
            kernel=gt_config["kernel"],
            tissue_params=tissue_params,
        )

        meta_path = output_path / f"{gt_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "id": gt_id,
                    "gt_type": gt_type,
                    "source": source.get_params(),
                    "kernel": gt_config["kernel"],
                    "sampling_level": gt_config.get("sampling_level", "7-point"),
                    "image_size": image_size,
                    "pixel_size_mm": pixel_size_mm,
                },
                f,
                indent=2,
            )

        results[gt_id] = {
            "gt_path": str(gt_path),
            "meta_path": str(meta_path),
            "source_type": source.get_params()["type"],
        }

        logger.info(f"  Saved: {gt_path}")

    summary_path = output_path / "gt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"GT summary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate GT for E1d")
    parser.add_argument(
        "--config",
        default="pilot/e1d_finite_source_local_surface/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1d_finite_source_local_surface/results/gt",
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

    run_gt_generation(config, args.output)


if __name__ == "__main__":
    main()
