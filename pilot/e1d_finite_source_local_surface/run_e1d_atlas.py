#!/usr/bin/env python3
"""Run E1d-R2 atlas-aware optimization experiments.

Experiment matrix:
- Part A: Geometry experiments (A1-A3)
- Part B: Quadrature experiments (B1-B2)
- Part C: Inverse degradation experiments (C1-C3)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
import yaml

import sys

sys.path.insert(0, "/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")

from surface_data import AtlasSurfaceData
from atlas_surface_renderer import (
    render_atlas_surface_local_depth,
    render_atlas_surface_flat,
)
from atlas_surface_renderer_torch import (
    DifferentiableAtlasForward,
    DifferentiableGaussianSourceAtlas,
    render_atlas_surface_torch,
)
from source_quadrature import sample_gaussian, sample_uniform

logger = logging.getLogger(__name__)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    """Compute metrics between prediction and GT."""
    pred_norm = pred - pred.mean()
    gt_norm = gt - gt.mean()

    ncc = np.sum(pred_norm * gt_norm) / (
        np.sqrt(np.sum(pred_norm**2) * np.sum(gt_norm**2)) + 1e-10
    )

    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    max_diff = np.abs(pred - gt).max()

    peak_pred_idx = np.argmax(pred)
    peak_gt_idx = np.argmax(gt)

    return {
        "ncc": float(ncc),
        "rmse": float(rmse),
        "max_diff": float(max_diff),
        "peak_pred_idx": int(peak_pred_idx),
        "peak_gt_idx": int(peak_gt_idx),
    }


def optimize_source_atlas(
    gt_response: np.ndarray,
    surface_coords: np.ndarray,
    surface_z_values: np.ndarray,
    tissue_params: dict,
    init_center: np.ndarray,
    init_sigmas: tuple,
    init_alpha: float,
    n_steps: int = 500,
    lr_center: float = 0.05,
    lr_size: float = 0.02,
    lr_alpha: float = 0.02,
    sampling_scheme: str = "sr-6",
    geometry_mode: str = "local_depth",
    device: torch.device = None,
    verbose: bool = False,
) -> Dict:
    """Optimize source parameters to match GT response on atlas surface.

    Args:
        gt_response: [N] ground truth response
        surface_coords: [N, 3] surface coordinates
        surface_z_values: [N] surface Z values
        tissue_params: optical properties
        init_center: initial center
        init_sigmas: initial sigmas
        init_alpha: initial alpha
        n_steps: optimization steps
        lr_center: learning rate for center
        lr_size: learning rate for size
        lr_alpha: learning rate for alpha
        sampling_scheme: quadrature scheme
        geometry_mode: geometry mode
        device: torch device
        verbose: print progress

    Returns:
        optimization result dict
    """
    if device is None:
        device = torch.device("cpu")

    gt_torch = torch.tensor(gt_response, dtype=torch.float32, device=device)

    forward_model = DifferentiableAtlasForward(
        surface_coords=surface_coords,
        surface_z_values=surface_z_values,
        tissue_params=tissue_params,
        sampling_scheme=sampling_scheme,
        kernel_type="green_halfspace",
        geometry_mode=geometry_mode,
        device=device,
    )

    source = DifferentiableGaussianSourceAtlas(
        center_init=init_center.astype(np.float32),
        sigmas_init=init_sigmas,
        alpha_init=init_alpha,
        device=device,
    )

    optimizer = optim.Adam(
        [
            {"params": [source.center], "lr": lr_center},
            {"params": [source.log_sigmas], "lr": lr_size},
            {"params": [source.log_alpha], "lr": lr_alpha},
        ]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    loss_history = []
    center_history = []
    sigma_history = []
    alpha_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        pred = forward_model(source.center, source.sigmas, source.alpha)

        loss = torch.mean((pred - gt_torch) ** 2)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [source.center, source.log_sigmas, source.log_alpha], max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            source.center.data[0].clamp_(5.0, 30.0)
            source.center.data[1].clamp_(20.0, 80.0)
            source.center.data[2].clamp_(2.0, 15.0)
            source.log_sigmas.data.clamp_(np.log(0.1), np.log(5.0))
            source.log_alpha.data.clamp_(np.log(0.1), np.log(10.0))

        loss_history.append(loss.item())
        center_history.append(source.center.detach().cpu().numpy().copy())
        sigma_history.append(source.sigmas.detach().cpu().numpy().copy())
        alpha_history.append(source.alpha.item())

        if verbose and step % 100 == 0:
            logger.info(
                f"Step {step}: loss={loss.item():.6e}, "
                f"center={source.center.detach().cpu().numpy()}, "
                f"sigmas={source.sigmas.detach().cpu().numpy()}"
            )

    return {
        "center_final": source.center.detach().cpu().numpy(),
        "sigmas_final": source.sigmas.detach().cpu().numpy(),
        "alpha_final": source.alpha.item(),
        "loss_history": loss_history,
        "center_history": center_history,
        "sigma_history": sigma_history,
        "alpha_history": alpha_history,
        "final_loss": loss_history[-1],
    }


def run_geometry_experiment(
    gt_id: str,
    gt_dir: str,
    atlas: AtlasSurfaceData,
    tissue_params: dict,
    inverse_surface_mode: str,
    inverse_sampling_scheme: str,
    config: dict,
    device: torch.device,
    output_dir: str,
    seed: Optional[int] = None,
) -> Dict:
    """Run geometry experiment (A1-A3).

    Args:
        gt_id: GT configuration ID
        gt_dir: GT directory
        atlas: atlas surface data
        tissue_params: optical properties
        inverse_surface_mode: "atlas_local_depth" or "flat"
        inverse_sampling_scheme: quadrature scheme for inverse
        config: experiment config
        device: torch device
        output_dir: output directory
        seed: random seed

    Returns:
        result dict
    """
    gt_path = Path(gt_dir) / f"{gt_id}_gt.npz"
    meta_path = Path(gt_dir) / f"{gt_id}_meta.json"

    if not gt_path.exists():
        logger.warning(f"GT not found: {gt_path}")
        return None

    gt_data = np.load(gt_path)
    gt_response = gt_data["response"]
    surface_coords = gt_data["surface_coords"]

    with open(meta_path) as f:
        gt_meta = json.load(f)

    source_center = np.array(gt_meta["source_center"])
    source_sigmas = gt_meta["source_params"].get("sigmas", [1.0, 1.0, 1.0])
    source_alpha = gt_meta["source_alpha"]

    if seed is not None:
        np.random.seed(seed)

    peak_idx = np.argmax(gt_response)
    init_center = surface_coords[peak_idx].copy()
    init_center[2] = 7.0

    offset_scale = 0.1
    center_offset = (np.random.rand(3) - 0.5) * 2 * offset_scale * 5.0
    init_center = init_center + center_offset

    init_sigmas = tuple(
        s * (1 + (np.random.rand() - 0.5) * 2 * offset_scale) for s in source_sigmas
    )
    init_sigmas = tuple(max(s, 0.1) for s in init_sigmas)

    init_alpha = source_alpha * (1 + (np.random.rand() - 0.5) * 2 * offset_scale)
    init_alpha = max(init_alpha, 0.1)

    optim_config = config.get("optimization", {})

    start_time = time.time()

    if inverse_surface_mode == "atlas_local_depth":
        result = optimize_source_atlas(
            gt_response=gt_response,
            surface_coords=surface_coords,
            surface_z_values=surface_coords[:, 2],
            tissue_params=tissue_params,
            init_center=init_center,
            init_sigmas=init_sigmas,
            init_alpha=init_alpha,
            n_steps=optim_config.get("n_steps", 500),
            lr_center=optim_config.get("lr_center", 0.05),
            lr_size=optim_config.get("lr_size", 0.02),
            lr_alpha=optim_config.get("lr_alpha", 0.02),
            sampling_scheme=inverse_sampling_scheme,
            geometry_mode="local_depth",
            device=device,
            verbose=True,
        )
    elif inverse_surface_mode == "flat":
        z_mean = surface_coords[:, 2].mean()
        z_at_source = surface_coords[
            np.argmin(
                np.linalg.norm(surface_coords[:, :2] - source_center[:2], axis=1)
            ),
            2,
        ]
        z_surface = z_at_source

        from atlas_surface_renderer import render_atlas_surface_flat
        from source_quadrature import sample_gaussian

        sigmas = np.array(source_sigmas, dtype=np.float32)

        response_flat = render_atlas_surface_flat(
            source_type="gaussian",
            source_center=source_center,
            source_params={"sigmas": source_sigmas},
            tissue_params=tissue_params,
            surface_coords_mm=surface_coords,
            z_surface=z_surface,
            sampling_scheme=inverse_sampling_scheme,
            kernel_type="green_halfspace",
            source_alpha=source_alpha,
        )

        metrics_flat_vs_gt = compute_metrics(response_flat, gt_response)

        result = {
            "center_final": init_center,
            "sigmas_final": init_sigmas,
            "alpha_final": init_alpha,
            "flat_response": response_flat,
            "metrics_flat_vs_gt": metrics_flat_vs_gt,
            "z_surface_used": float(z_surface),
        }
    else:
        raise ValueError(f"Unknown inverse surface mode: {inverse_surface_mode}")

    optim_time = time.time() - start_time

    center_pred = np.array(result["center_final"])
    center_gt = np.array(source_center)
    position_error = np.linalg.norm(center_pred - center_gt)

    sigmas_pred = np.array(result["sigmas_final"])
    sigmas_gt = np.array(source_sigmas)
    size_error = np.linalg.norm(sigmas_pred - sigmas_gt)
    size_error_ratio = size_error / (np.linalg.norm(sigmas_gt) + 1e-10)

    alpha_pred = result["alpha_final"]
    alpha_gt = source_alpha
    alpha_error_ratio = abs(alpha_pred - alpha_gt) / alpha_gt

    result_out = {
        "gt_id": gt_id,
        "inverse_surface_mode": inverse_surface_mode,
        "inverse_sampling_scheme": inverse_sampling_scheme,
        "position_error_mm": float(position_error),
        "size_error_mm": float(size_error),
        "size_error_ratio": float(size_error_ratio),
        "alpha_error_ratio": float(alpha_error_ratio),
        "center_pred": center_pred.tolist(),
        "center_gt": center_gt.tolist(),
        "sigmas_pred": sigmas_pred.tolist(),
        "sigmas_gt": sigmas_gt.tolist(),
        "alpha_pred": float(alpha_pred),
        "alpha_gt": float(alpha_gt),
        "optim_time_s": optim_time,
        "seed": seed,
    }

    if "final_loss" in result:
        result_out["final_loss"] = result["final_loss"]

    if "metrics_flat_vs_gt" in result:
        result_out["flat_vs_gt_metrics"] = result["metrics_flat_vs_gt"]

    logger.info(f"Position error: {position_error:.3f} mm")
    logger.info(f"Size error: {size_error:.3f} mm ({size_error_ratio * 100:.1f}%)")

    return result_out


def run_quadrature_experiment(
    gt_response: np.ndarray,
    surface_coords: np.ndarray,
    tissue_params: dict,
    source_center: np.ndarray,
    source_sigmas: tuple,
    source_alpha: float,
    sampling_schemes: list,
    device: torch.device,
) -> Dict:
    """Run quadrature comparison experiment (B1-B2).

    Args:
        gt_response: reference GT response (high-fidelity)
        surface_coords: surface coordinates
        tissue_params: optical properties
        source_center: source center
        source_sigmas: source sigmas
        source_alpha: source alpha
        sampling_schemes: list of schemes to compare
        device: torch device

    Returns:
        comparison results
    """
    results = {}

    for scheme in sampling_schemes:
        source_points, source_weights = sample_gaussian(
            center=source_center.astype(np.float32),
            sigmas=np.array(source_sigmas, dtype=np.float32),
            alpha=source_alpha,
            scheme=scheme,
        )

        response = render_atlas_surface_local_depth(
            source_type="gaussian",
            source_center=source_center,
            source_params={"sigmas": source_sigmas},
            tissue_params=tissue_params,
            surface_coords_mm=surface_coords,
            sampling_scheme=scheme,
            kernel_type="green_halfspace",
            source_alpha=source_alpha,
        )

        metrics = compute_metrics(response, gt_response)

        results[scheme] = {
            "ncc": metrics["ncc"],
            "rmse": metrics["rmse"],
            "max_diff": metrics["max_diff"],
            "n_points": len(source_points),
        }

        logger.info(f"  {scheme}: NCC={metrics['ncc']:.4f}, RMSE={metrics['rmse']:.4e}")

    return results


def run_all_experiments(
    config_path: str,
    gt_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> Dict:
    """Run all E1d-R2 experiments.

    Args:
        config_path: config file path
        gt_dir: GT directory
        output_dir: output directory
        device: device string

    Returns:
        summary dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch_device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tissue_params = config["tissue"]["muscle"]

    atlas = AtlasSurfaceData(
        mesh_path=config.get("mesh_path", "output/shared/mesh.npz"),
        compute_normals=True,
    )

    logger.info(f"Loaded atlas: {len(atlas.surface_coords)} surface nodes")

    all_results = {}

    logger.info("\n" + "=" * 60)
    logger.info("Part A: Geometry Experiments")
    logger.info("=" * 60)

    geometry_experiments = [
        ("A1_atlas_self_consistent_shallow", "atlas_local_depth", "grid-27"),
        ("A1_atlas_self_consistent_deep", "atlas_local_depth", "grid-27"),
        ("A2_atlas_vs_flat_shallow", "flat", "grid-27"),
        ("A2_atlas_vs_flat_deep", "flat", "grid-27"),
        ("A3_lateral_source", "atlas_local_depth", "grid-27"),
    ]

    for gt_id, inverse_mode, scheme in geometry_experiments:
        logger.info(f"\n--- {gt_id} ({inverse_mode}) ---")

        result = run_geometry_experiment(
            gt_id=gt_id,
            gt_dir=gt_dir,
            atlas=atlas,
            tissue_params=tissue_params,
            inverse_surface_mode=inverse_mode,
            inverse_sampling_scheme=scheme,
            config=config,
            device=torch_device,
            output_dir=output_dir,
            seed=42,
        )

        if result is not None:
            all_results[f"A_{gt_id}"] = result

    logger.info("\n" + "=" * 60)
    logger.info("Part B: Quadrature Experiments")
    logger.info("=" * 60)

    schemes_to_test = ["1-point", "sr-6", "ut-7", "7-point", "grid-27"]

    for ref_id in ["B1_gaussian_sr6_reference", "B2_uniform_stratified_reference"]:
        gt_path = Path(gt_dir) / f"{ref_id}_gt.npz"
        if not gt_path.exists():
            continue

        logger.info(f"\n--- {ref_id} ---")

        gt_data = np.load(gt_path)
        gt_response = gt_data["response"]
        surface_coords = gt_data["surface_coords"]

        meta_path = Path(gt_dir) / f"{ref_id}_meta.json"
        with open(meta_path) as f:
            gt_meta = json.load(f)

        source_center = np.array(gt_meta["source_center"])
        source_sigmas = gt_meta["source_params"].get("sigmas", [1.0, 1.0, 1.0])
        source_alpha = gt_meta["source_alpha"]

        quad_results = run_quadrature_experiment(
            gt_response=gt_response,
            surface_coords=surface_coords,
            tissue_params=tissue_params,
            source_center=source_center,
            source_sigmas=source_sigmas,
            source_alpha=source_alpha,
            sampling_schemes=schemes_to_test,
            device=torch_device,
        )

        all_results[f"B_{ref_id}"] = quad_results

    logger.info("\n" + "=" * 60)
    logger.info("Part C: Inverse Degradation Experiments")
    logger.info("=" * 60)

    inverse_experiments = [
        ("C1_gaussian_to_gaussian", "atlas_local_depth", "grid-27"),
        ("C2_uniform_to_uniform", "atlas_local_depth", "grid-27"),
        ("C3_uniform_to_gaussian", "atlas_local_depth", "grid-27"),
    ]

    for gt_id, inverse_mode, scheme in inverse_experiments:
        logger.info(f"\n--- {gt_id} ---")

        result = run_geometry_experiment(
            gt_id=gt_id,
            gt_dir=gt_dir,
            atlas=atlas,
            tissue_params=tissue_params,
            inverse_surface_mode=inverse_mode,
            inverse_sampling_scheme=scheme,
            config=config,
            device=torch_device,
            output_dir=output_dir,
            seed=42,
        )

        if result is not None:
            all_results[f"C_{gt_id}"] = result

    summary_path = output_path / "e1d_atlas_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    logger.info(f"\nSummary saved: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run E1d-R2 atlas experiments")
    parser.add_argument(
        "--config",
        default="pilot/e1d_finite_source_local_surface/config_atlas.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--gt-dir",
        default="pilot/e1d_finite_source_local_surface/results/gt_atlas",
        help="GT data directory",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1d_finite_source_local_surface/results/atlas_experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_all_experiments(args.config, args.gt_dir, args.output, args.device)


if __name__ == "__main__":
    main()
