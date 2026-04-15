#!/usr/bin/env python3
"""Compare kernel predictions against MCX surface GT.

Computes metrics and generates comparison plots.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def peak_location_error_mm(
    gt: np.ndarray, pred: np.ndarray, pixel_size_mm: float
) -> float:
    """Compute peak location error in mm.

    Args:
        gt: ground truth image [H, W]
        pred: predicted image [H, W]
        pixel_size_mm: pixel size

    Returns:
        error in mm
    """
    gt_peak = np.unravel_index(np.argmax(gt), gt.shape)
    pred_peak = np.unravel_index(np.argmax(pred), pred.shape)

    dy = (gt_peak[0] - pred_peak[0]) * pixel_size_mm
    dx = (gt_peak[1] - pred_peak[1]) * pixel_size_mm

    return np.sqrt(dx**2 + dy**2)


def ncc(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute normalized cross-correlation.

    Args:
        gt: ground truth
        pred: prediction

    Returns:
        NCC value in [-1, 1]
    """
    gt_flat = gt.flatten().astype(np.float64)
    pred_flat = pred.flatten().astype(np.float64)

    gt_mean = gt_flat.mean()
    pred_mean = pred_flat.mean()

    numerator = np.sum((gt_flat - gt_mean) * (pred_flat - pred_mean))
    denominator = np.sqrt(
        np.sum((gt_flat - gt_mean) ** 2) * np.sum((pred_flat - pred_mean) ** 2)
    )

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)


def normalize_peak(image: np.ndarray) -> np.ndarray:
    """Peak normalize image to [0, 1]."""
    img_max = image.max()
    if img_max < 1e-12:
        return image
    return image / img_max


def normalize_energy(image: np.ndarray) -> np.ndarray:
    """Energy normalize image (sum = 1)."""
    img_sum = image.sum()
    if img_sum < 1e-12:
        return image
    return image / img_sum


def radial_profile(
    image: np.ndarray,
    center: Tuple[int, int] = None,
    peak_mode: str = "argmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract radial profile from image.

    Args:
        image: 2D image [H, W]
        center: (row, col) center, if None use peak_mode
        peak_mode: "argmax" or "source_projection"

    Returns:
        (rho_mm, intensity): radial distance and mean intensity
    """
    H, W = image.shape

    if center is None:
        if peak_mode == "argmax":
            peak = np.unravel_index(np.argmax(image), image.shape)
            center = (peak[0], peak[1])
        else:
            center = (H // 2, W // 2)

    row_coords, col_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    dist_voxels = np.sqrt((row_coords - center[0]) ** 2 + (col_coords - center[1]) ** 2)

    max_r = int(np.ceil(dist_voxels.max()))
    rho_voxels = np.arange(max_r + 1)
    intensity = np.zeros(len(rho_voxels))

    for i, r in enumerate(rho_voxels):
        if i == 0:
            mask = dist_voxels < 0.5
        else:
            mask = (dist_voxels >= r - 0.5) & (dist_voxels < r + 0.5)

        if mask.sum() > 0:
            intensity[i] = image[mask].mean()
        elif i > 0:
            intensity[i] = intensity[i - 1]

    return rho_voxels, intensity


def fwhm_mm(image: np.ndarray, pixel_size_mm: float, mode: str = "radial") -> float:
    """Compute full width at half maximum.

    Args:
        image: 2D image
        pixel_size_mm: pixel size
        mode: "radial" or "1d"

    Returns:
        FWHM in mm
    """
    if mode == "radial":
        rho_voxels, intensity = radial_profile(image, peak_mode="argmax")
    else:
        row_center = image.shape[0] // 2
        intensity = image[row_center, :]
        rho_voxels = np.arange(len(intensity))

    intensity = normalize_peak(intensity)

    half_max = 0.5
    above_half = intensity >= half_max

    if not above_half.any():
        return 0.0

    indices = np.where(above_half)[0]
    fwhm_voxels = indices[-1] - indices[0] + 1

    return fwhm_voxels * pixel_size_mm


def radial_rmse(profile_gt: np.ndarray, profile_pred: np.ndarray) -> float:
    """Compute RMSE between radial profiles.

    Args:
        profile_gt: GT radial profile
        profile_pred: Predicted radial profile

    Returns:
        RMSE value
    """
    min_len = min(len(profile_gt), len(profile_pred))
    if min_len == 0:
        return float("inf")

    gt = profile_gt[:min_len]
    pred = profile_pred[:min_len]

    gt = normalize_peak(gt)
    pred = normalize_peak(pred)

    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def compare_config(
    config_id: str,
    gt_dir: Path,
    kernels_module,
    render_module,
    config: dict,
) -> Dict:
    """Compare all kernels for a single configuration.

    Args:
        config_id: configuration ID
        gt_dir: GT directory
        kernels_module: kernels module
        render_module: render_surface module
        config: full config dict

    Returns:
        dict with metrics for all kernels
    """
    gt_path = gt_dir / config_id / f"{config_id}_surface_gt.npz"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    gt_data = np.load(gt_path)
    gt_image = gt_data["surface_image"]
    source_world = gt_data["source_world"]

    tissue_params = {
        "mua_mm": float(gt_data["mua_mm"]),
        "mus_mm": float(gt_data["mus_mm"]),
        "g": float(gt_data["g"]),
        "n": float(gt_data["n"]),
    }

    image_size = config["image"]["size"]
    pixel_size_mm = config["image"]["pixel_size_mm"]

    calib_params = config.get("calibration", {}).get("gaussian", {})

    rendered = render_module.render_all_kernels(
        source_world=source_world,
        tissue_params=tissue_params,
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        calib_params=calib_params,
    )

    metrics = {}
    kernels = ["gaussian_2d", "green_infinite", "green_halfspace"]

    for kernel_name in kernels:
        pred_image = rendered[kernel_name]

        gt_peak_norm = normalize_peak(gt_image)
        pred_peak_norm = normalize_peak(pred_image)

        gt_energy_norm = normalize_energy(gt_image)
        pred_energy_norm = normalize_energy(pred_image)

        peak_error = peak_location_error_mm(gt_image, pred_image, pixel_size_mm)

        ncc_peak = ncc(gt_peak_norm, pred_peak_norm)
        ncc_energy = ncc(gt_energy_norm, pred_energy_norm)

        _, gt_profile = radial_profile(gt_image, peak_mode="argmax")
        _, pred_profile = radial_profile(pred_image, peak_mode="argmax")

        gt_profile_norm = normalize_peak(gt_profile)
        pred_profile_norm = normalize_peak(pred_profile)

        radial_rmse_val = radial_rmse(gt_profile_norm, pred_profile_norm)

        fwhm_gt = fwhm_mm(gt_image, pixel_size_mm)
        fwhm_pred = fwhm_mm(pred_image, pixel_size_mm)

        fwhm_ratio = fwhm_pred / fwhm_gt if fwhm_gt > 0 else 0.0

        metrics[kernel_name] = {
            "peak_error_mm": float(peak_error),
            "ncc_peak_norm": float(ncc_peak),
            "ncc_energy_norm": float(ncc_energy),
            "radial_rmse_peak_norm": float(radial_rmse_val),
            "fwhm_gt_mm": float(fwhm_gt),
            "fwhm_pred_mm": float(fwhm_pred),
            "fwhm_ratio": float(fwhm_ratio),
        }

    return {
        "config_id": config_id,
        "metrics": metrics,
        "images": {
            "gt": gt_image,
            **rendered,
        },
    }


def plot_comparison(
    config_id: str,
    results: Dict,
    output_dir: Path,
    pixel_size_mm: float,
):
    """Generate comparison plots.

    Args:
        config_id: configuration ID
        results: results dict from compare_config
        output_dir: output directory
        pixel_size_mm: pixel size
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    images = results["images"]
    metrics = results["metrics"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    gt = images["gt"]
    gt_norm = normalize_peak(gt)

    ax = axes[0, 0]
    im = ax.imshow(gt_norm, cmap="hot", origin="lower")
    ax.set_title(f"MCX GT\npeak={gt_norm.max():.4f}")
    plt.colorbar(im, ax=ax)

    peak = np.unravel_index(np.argmax(gt), gt.shape)
    ax.plot(peak[1], peak[0], "cx", markersize=10, markeredgewidth=2)

    for i, kernel_name in enumerate(
        ["gaussian_2d", "green_infinite", "green_halfspace"]
    ):
        pred = images[kernel_name]
        pred_norm = normalize_peak(pred)

        ax = axes[0, i + 1] if i < 2 else axes[1, 0]
        if i == 2:
            ax = axes[1, 0]
        else:
            ax = axes[0, i + 1]

        im = ax.imshow(pred_norm, cmap="hot", origin="lower")
        m = metrics[kernel_name]
        ax.set_title(
            f"{kernel_name}\nNCC={m['ncc_peak_norm']:.4f}, PE={m['peak_error_mm']:.2f}mm"
        )
        plt.colorbar(im, ax=ax)

        pred_peak = np.unravel_index(np.argmax(pred), pred.shape)
        ax.plot(pred_peak[1], pred_peak[0], "cx", markersize=10, markeredgewidth=2)

    ax = axes[1, 1]
    for kernel_name in ["gaussian_2d", "green_infinite", "green_halfspace"]:
        _, profile = radial_profile(images[kernel_name], peak_mode="argmax")
        profile_norm = normalize_peak(profile)
        ax.plot(profile_norm, label=kernel_name, alpha=0.8)

    _, gt_profile = radial_profile(gt, peak_mode="argmax")
    gt_profile_norm = normalize_peak(gt_profile)
    ax.plot(gt_profile_norm, "k--", label="MCX GT", linewidth=2)

    ax.set_xlabel("Radial distance (pixels)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Radial Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    kernel_names = ["gaussian_2d", "green_infinite", "green_halfspace"]
    x = np.arange(len(kernel_names))
    width = 0.35

    ncc_vals = [metrics[k]["ncc_peak_norm"] for k in kernel_names]
    peak_errs = [metrics[k]["peak_error_mm"] for k in kernel_names]

    bars1 = ax.bar(x - width / 2, ncc_vals, width, label="NCC")
    ax.set_ylabel("NCC")
    ax.set_ylim([0, 1.1])

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, peak_errs, width, label="Peak Error", color="orange")
    ax2.set_ylabel("Peak Error (mm)", color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(["Gaussian", "Infinite", "Half-space"])
    ax.set_title("Metrics Comparison")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    output_path = output_dir / f"{config_id}_kernel_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare kernels against MCX GT")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config IDs to compare (default: all with GT)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).parent
    gt_dir = base_dir / config["output"]["gt_dir"]
    comparison_dir = base_dir / config["output"]["comparison_dir"]
    comparison_dir.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(base_dir))
    import kernels
    import render_surface

    if args.configs:
        config_ids = args.configs
    else:
        config_ids = []
        for d in gt_dir.iterdir():
            if d.is_dir() and (d / f"{d.name}_surface_gt.npz").exists():
                config_ids.append(d.name)

    all_results = {}

    for config_id in sorted(config_ids):
        logger.info(f"Comparing {config_id}...")

        try:
            results = compare_config(
                config_id=config_id,
                gt_dir=gt_dir,
                kernels_module=kernels,
                render_module=render_surface,
                config=config,
            )

            pixel_size_mm = config["image"]["pixel_size_mm"]
            plot_comparison(config_id, results, comparison_dir, pixel_size_mm)

            metrics_path = comparison_dir / f"{config_id}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(results["metrics"], f, indent=2)

            all_results[config_id] = results["metrics"]

            logger.info(f"  {config_id} metrics:")
            for kernel_name, m in results["metrics"].items():
                logger.info(
                    f"    {kernel_name}: NCC={m['ncc_peak_norm']:.4f}, PE={m['peak_error_mm']:.3f}mm, FWHM_ratio={m['fwhm_ratio']:.3f}"
                )

        except Exception as e:
            logger.error(f"Failed {config_id}: {e}")
            import traceback

            traceback.print_exc()

    summary_path = base_dir / "results" / "summary.json"
    compute_summary(all_results, summary_path)

    logger.info(f"\nComparison complete. Summary: {summary_path}")


def compute_summary(all_results: Dict, summary_path: Path):
    """Compute aggregate metrics and select best kernel.

    Args:
        all_results: dict of config_id -> metrics
        summary_path: output path for summary.json
    """
    kernels = ["gaussian_2d", "green_infinite", "green_halfspace"]

    mean_metrics = {}
    for kernel_name in kernels:
        ncc_vals = []
        peak_errs = []
        fwhm_ratios = []

        for config_id, metrics in all_results.items():
            m = metrics.get(kernel_name, {})
            if "ncc_peak_norm" in m:
                ncc_vals.append(m["ncc_peak_norm"])
            if "peak_error_mm" in m:
                peak_errs.append(m["peak_error_mm"])
            if "fwhm_ratio" in m:
                fwhm_ratios.append(m["fwhm_ratio"])

        mean_metrics[kernel_name] = {
            "mean_ncc": float(np.mean(ncc_vals)) if ncc_vals else 0.0,
            "mean_peak_error_mm": float(np.mean(peak_errs))
            if peak_errs
            else float("inf"),
            "mean_fwhm_ratio": float(np.mean(fwhm_ratios)) if fwhm_ratios else 0.0,
            "n_configs": len(ncc_vals),
        }

    sanity_path = (
        Path(__file__).parent / "results" / "sanity" / "coordinate_sanity.json"
    )
    sanity_passed = False
    if sanity_path.exists():
        with open(sanity_path) as f:
            sanity_data = json.load(f)
            sanity_passed = sanity_data.get("sanity_passed", False)

    selected = "green_halfspace"
    selection_reason = "Default selection: half-space Green's function accounts for boundary condition."

    inf_ncc = mean_metrics.get("green_infinite", {}).get("mean_ncc", 0)
    hs_ncc = mean_metrics.get("green_halfspace", {}).get("mean_ncc", 0)
    inf_pe = mean_metrics.get("green_infinite", {}).get(
        "mean_peak_error_mm", float("inf")
    )
    hs_pe = mean_metrics.get("green_halfspace", {}).get(
        "mean_peak_error_mm", float("inf")
    )

    if inf_ncc > hs_ncc + 0.02 and inf_pe < hs_pe:
        selected = "green_infinite"
        selection_reason = (
            f"Infinite Green's function selected: NCC={inf_ncc:.4f} > half-space NCC={hs_ncc:.4f}, "
            f"and peak error={inf_pe:.3f}mm < half-space={hs_pe:.3f}mm."
        )

    decision = "NOGO"
    hs_metrics = mean_metrics.get("green_halfspace", {})
    if (
        hs_metrics.get("mean_ncc", 0) >= 0.98
        and hs_metrics.get("mean_peak_error_mm", 1) < 0.2
    ):
        decision = "GO"
    elif (
        hs_metrics.get("mean_ncc", 0) >= 0.95
        and hs_metrics.get("mean_peak_error_mm", 1) < 0.5
    ):
        decision = "CAUTION"

    summary = {
        "coordinate_sanity_passed": sanity_passed,
        "selected_green_function": selected,
        "decision": decision,
        "mean_metrics": mean_metrics,
        "per_config": all_results,
        "selection_reason": selection_reason,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Sanity passed: {sanity_passed}")
    logger.info(f"Selected kernel: {selected}")
    logger.info(f"Decision: {decision}")
    logger.info(f"\nMean metrics:")
    for kernel_name, m in mean_metrics.items():
        logger.info(f"  {kernel_name}:")
        logger.info(f"    NCC: {m['mean_ncc']:.4f}")
        logger.info(f"    Peak Error: {m['mean_peak_error_mm']:.3f} mm")
        logger.info(f"    FWHM Ratio: {m['mean_fwhm_ratio']:.3f}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
