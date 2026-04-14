#!/usr/bin/env python3
"""Run E1d optimization experiments.

Tests finite-size source + local surface approximation degradation.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.optim as optim
import yaml

import sys

sys.path.insert(0, "/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")

from source_models import (
    create_source,
    BaseSource,
    GaussianEllipsoidSource,
    UniformEllipsoidSource,
)
from local_surface_renderer import render_local_surface_response

logger = logging.getLogger(__name__)


class DifferentiableGaussianSource(torch.nn.Module):
    """Differentiable Gaussian source for optimization."""

    def __init__(
        self,
        center_init: np.ndarray,
        sigmas_init: tuple,
        alpha_init: float,
    ):
        super().__init__()
        self.center = torch.nn.Parameter(torch.tensor(center_init, dtype=torch.float32))
        self.log_sigmas = torch.nn.Parameter(
            torch.log(torch.tensor(sigmas_init, dtype=torch.float32))
        )
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(alpha_init, dtype=torch.float32))
        )

    @property
    def sigmas(self):
        return torch.exp(self.log_sigmas)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)


def render_surface_torch(
    center: torch.Tensor,
    sigmas: torch.Tensor,
    alpha: torch.Tensor,
    surface_points: torch.Tensor,
    tissue_params: dict,
    z_surface: float = 10.0,
    sampling_level: str = "7-point",
) -> torch.Tensor:
    """Render surface response using PyTorch (differentiable).

    Simplified version for optimization - uses Gaussian source only.

    Args:
        center: [3] source center
        sigmas: [3] source sigmas
        alpha: source intensity
        surface_points: [N, 2] surface XY coordinates
        tissue_params: optical properties
        z_surface: surface z coordinate
        sampling_level: sampling level (currently uses 7-point)

    Returns:
        response: [N] surface response
    """
    import math

    mua = tissue_params["mua_mm"]
    mus = tissue_params["mus_mm"]
    g = tissue_params.get("g", 0.9)

    mus_prime = mus * (1 - g)
    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = math.sqrt(3.0 * mua * (mua + mus_prime))

    R_eff = 0.493
    A = (1 + R_eff) / (1 - R_eff)
    zb = 2 * A * D

    depth = z_surface - center[2]

    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=center.dtype,
        device=center.device,
    )

    offsets = offsets * sigmas.unsqueeze(0)
    sample_points = center.unsqueeze(0) + offsets

    n_surf = surface_points.shape[0]
    n_src = sample_points.shape[0]

    dx = surface_points[:, 0:1] - sample_points[:, 0].unsqueeze(0)
    dy = surface_points[:, 1:2] - sample_points[:, 1].unsqueeze(0)

    sample_depths = z_surface - sample_points[:, 2]

    rho_sq = dx**2 + dy**2
    rho = torch.sqrt(rho_sq)

    r1_sq = rho_sq + sample_depths.unsqueeze(0) ** 2
    r1 = torch.sqrt(torch.clamp(r1_sq, min=1e-12))

    r2_sq = rho_sq + (sample_depths.unsqueeze(0) + 2 * zb) ** 2
    r2 = torch.sqrt(torch.clamp(r2_sq, min=1e-12))

    G1 = torch.exp(-mu_eff * r1) / (4 * math.pi * D * r1)
    G2 = torch.exp(-mu_eff * r2) / (4 * math.pi * D * r2)

    G_halfspace = torch.clamp(G1 - G2, min=0.0)

    diff = sample_points - center.unsqueeze(0)
    diff_norm = diff / sigmas.unsqueeze(0)
    dist_sq = torch.sum(diff_norm**2, dim=1)
    weights = torch.exp(-0.5 * dist_sq)
    weights = weights / weights.sum() * alpha

    response = torch.sum(G_halfspace * weights.unsqueeze(0), dim=1)

    return response


def optimize_source(
    gt_image: np.ndarray,
    tissue_params: dict,
    image_size: int,
    pixel_size_mm: float,
    init_center: np.ndarray,
    init_sigmas: tuple,
    init_alpha: float,
    n_steps: int = 500,
    lr_center: float = 0.05,
    lr_size: float = 0.02,
    lr_alpha: float = 0.02,
    z_surface: float = 10.0,
    device: torch.device = None,
    verbose: bool = False,
) -> dict:
    """Optimize source parameters to match GT image.

    Args:
        gt_image: ground truth surface image
        tissue_params: optical properties
        image_size: image dimension
        pixel_size_mm: pixel size
        init_center: initial center position
        init_sigmas: initial sigmas
        init_alpha: initial alpha
        n_steps: number of optimization steps
        lr_center: learning rate for center
        lr_size: learning rate for size
        lr_alpha: learning rate for alpha
        z_surface: surface z coordinate
        device: torch device
        verbose: print progress

    Returns:
        optimization results dict
    """
    if device is None:
        device = torch.device("cpu")

    fov_mm = image_size * pixel_size_mm
    coords_mm = (np.arange(image_size) - image_size / 2 + 0.5) * pixel_size_mm
    x_grid, y_grid = np.meshgrid(coords_mm, coords_mm)
    surface_points = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)
    surface_points_torch = torch.tensor(
        surface_points, dtype=torch.float32, device=device
    )

    gt_flat = gt_image.flatten()
    gt_torch = torch.tensor(gt_flat, dtype=torch.float32, device=device)

    source = DifferentiableGaussianSource(
        center_init=init_center.astype(np.float32),
        sigmas_init=init_sigmas,
        alpha_init=init_alpha,
    ).to(device)

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

        pred = render_surface_torch(
            center=source.center,
            sigmas=source.sigmas,
            alpha=source.alpha,
            surface_points=surface_points_torch,
            tissue_params=tissue_params,
            z_surface=z_surface,
        )

        loss = torch.mean((pred - gt_torch) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        center_history.append(source.center.detach().cpu().numpy().copy())
        sigma_history.append(source.sigmas.detach().cpu().numpy().copy())
        alpha_history.append(source.alpha.item())

        if verbose and step % 50 == 0:
            logger.info(
                f"Step {step}: loss={loss.item():.6e}, "
                f"center={source.center.detach().cpu().numpy()}, "
                f"sigmas={source.sigmas.detach().cpu().numpy()}, "
                f"alpha={source.alpha.item():.4f}"
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


def compute_optimization_metrics(result: dict, gt_params: dict) -> dict:
    """Compute metrics for optimization result.

    Args:
        result: optimization result
        gt_params: ground truth parameters

    Returns:
        metrics dict
    """
    center_pred = np.array(result["center_final"])
    center_gt = np.array(gt_params["center"])

    position_error = np.linalg.norm(center_pred - center_gt)

    sigmas_pred = np.array(result["sigmas_final"])
    sigmas_gt = np.array(gt_params.get("sigmas", [1.0, 1.0, 1.0]))
    size_error = np.linalg.norm(sigmas_pred - sigmas_gt)
    size_error_ratio = size_error / (np.linalg.norm(sigmas_gt) + 1e-10)

    alpha_pred = result["alpha_final"]
    alpha_gt = gt_params.get("alpha", 1.0)
    alpha_error_ratio = abs(alpha_pred - alpha_gt) / alpha_gt

    return {
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
        "final_loss": result["final_loss"],
    }


def run_single_experiment(
    gt_id: str,
    gt_dir: str,
    output_dir: str,
    config: dict,
    gt_type: str = "inverse_crime",
    mismatch_level: str = "7-point",
    seed: Optional[int] = None,
    device: torch.device = None,
) -> dict:
    """Run single optimization experiment.

    Args:
        gt_id: GT configuration ID
        gt_dir: GT directory
        output_dir: output directory
        config: experiment config
        gt_type: "inverse_crime" or "finite_source"
        mismatch_level: sampling level for forward (lower = more mismatch)
        seed: random seed for initialization
        device: torch device

    Returns:
        result dict
    """
    gt_path = Path(gt_dir) / f"{gt_id}_gt.npz"
    meta_path = Path(gt_dir) / f"{gt_id}_meta.json"

    if not gt_path.exists():
        logger.warning(f"GT not found: {gt_path}")
        return None

    gt_data = np.load(gt_path)
    gt_image = gt_data["image"]

    with open(meta_path) as f:
        gt_meta = json.load(f)

    source_params = gt_meta["source"]
    gt_center = np.array(source_params["center"])
    gt_sigmas = tuple(source_params.get("sigmas", [1.0, 1.0, 1.0]))
    gt_alpha = source_params.get("alpha", 1.0)

    if seed is not None:
        np.random.seed(seed)

    offset_scale = 0.5
    center_offset = (np.random.rand(3) - 0.5) * 2 * offset_scale * np.abs(gt_center)
    center_offset[2] = np.random.uniform(-0.5, 1.0)
    init_center = gt_center + center_offset
    init_sigmas = tuple(
        s * (1 + (np.random.rand() - 0.5) * 2 * offset_scale) for s in gt_sigmas
    )
    init_sigmas = tuple(max(s, 0.1) for s in init_sigmas)
    init_alpha = gt_alpha * (1 + (np.random.rand() - 0.5) * 2 * offset_scale)
    init_alpha = max(init_alpha, 0.1)

    tissue_params = config["tissue"]["muscle"]

    start_time = time.time()

    result = optimize_source(
        gt_image=gt_image,
        tissue_params=tissue_params,
        image_size=config["rendering"]["image_size"],
        pixel_size_mm=config["rendering"]["pixel_size_mm"],
        init_center=init_center,
        init_sigmas=init_sigmas,
        init_alpha=init_alpha,
        n_steps=config["optimization"]["n_steps"],
        lr_center=config["optimization"]["lr_center"],
        lr_size=config["optimization"]["lr_size"],
        lr_alpha=config["optimization"]["lr_alpha"],
        z_surface=config["rendering"]["z_surface"],
        device=device,
        verbose=True,
    )

    optim_time = time.time() - start_time

    metrics = compute_optimization_metrics(result, source_params)
    metrics["optim_time_s"] = optim_time
    metrics["gt_id"] = gt_id
    metrics["gt_type"] = gt_type
    metrics["mismatch_level"] = mismatch_level
    metrics["seed"] = seed

    logger.info(f"Position error: {metrics['position_error_mm']:.3f} mm")
    logger.info(f"Size error: {metrics['size_error_mm']:.3f} mm")
    logger.info(f"Alpha error: {metrics['alpha_error_ratio'] * 100:.1f}%")

    return {**result, **metrics}


def run_all_experiments(
    config_path: str,
    gt_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> dict:
    """Run all E1d experiments.

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

    results = {}

    for gt_config in config["gt_configs"]:
        gt_id = gt_config["id"]
        gt_type = gt_config.get("gt_type", "inverse_crime")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {gt_id} ({gt_type})")
        logger.info(f"{'=' * 60}")

        result = run_single_experiment(
            gt_id=gt_id,
            gt_dir=gt_dir,
            output_dir=output_dir,
            config=config,
            gt_type=gt_type,
            device=torch_device,
        )

        if result is not None:
            results[gt_id] = result

            result_path = output_path / f"{gt_id}_optim.npz"
            np.savez(
                result_path,
                loss_history=result["loss_history"],
                center_history=np.array(result["center_history"]),
                sigma_history=np.array(result["sigma_history"]),
                alpha_history=np.array(result["alpha_history"]),
            )

    summary_path = output_path / "optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )

    logger.info(f"\nSummary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run E1d optimization experiments")
    parser.add_argument(
        "--config",
        default="pilot/e1d_finite_source_local_surface/config.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--gt-dir",
        default="pilot/e1d_finite_source_local_surface/results/gt",
        help="GT data directory",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1d_finite_source_local_surface/results/optimization",
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
