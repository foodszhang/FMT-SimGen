#!/usr/bin/env python3
"""Real surface ablation study for E1d.

Compares planar surface assumption vs real surface with local depth approximation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import yaml
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")

from source_models import create_source
from kernels import green_halfspace_finite_source
from surface_data import load_real_surface, get_surface_nodes_in_roi

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
    mean_diff = np.abs(pred - gt).mean()

    return {
        "ncc": float(ncc),
        "rmse": float(rmse),
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
    }


def run_real_surface_ablation(
    config: Dict,
    output_dir: str,
) -> Dict:
    """Run real surface ablation study."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tissue_params = config["tissue"]["muscle"]

    results = {}

    for rs_config in config.get("real_surface_configs", []):
        rs_id = rs_config["id"]
        logger.info(f"\nRunning real surface test: {rs_id}")

        source = create_source(rs_config["source"])
        source_points, source_weights = source.sample_points("7-point")

        surface_data = load_real_surface(rs_config["mesh_path"])
        surface_coords = surface_data.surface_coords

        roi_radius = rs_config.get("roi_radius_mm", 30.0)
        roi_indices = get_surface_nodes_in_roi(
            source.center, surface_coords, roi_radius
        )

        roi_surface = surface_coords[roi_indices]
        roi_xy = roi_surface[:, :2]
        roi_z = roi_surface[:, 2]

        z_mean = roi_z.mean()
        z_at_source = roi_z[
            np.argmin(np.linalg.norm(roi_xy - source.center[:2], axis=1))
        ]

        logger.info(f"  Surface Z range in ROI: [{roi_z.min():.1f}, {roi_z.max():.1f}]")
        logger.info(
            f"  Mean Z: {z_mean:.1f}, Z at source projection: {z_at_source:.1f}"
        )

        response_mean_z = green_halfspace_finite_source(
            surface_points_mm=roi_xy,
            source_points_mm=source_points,
            source_weights=source_weights,
            tissue_params=tissue_params,
            z_surface=z_mean,
            surface_z_values=None,
        )

        response_local_z = green_halfspace_finite_source(
            surface_points_mm=roi_xy,
            source_points_mm=source_points,
            source_weights=source_weights,
            tissue_params=tissue_params,
            z_surface=z_mean,
            surface_z_values=roi_z,
        )

        response_proj_z = green_halfspace_finite_source(
            surface_points_mm=roi_xy,
            source_points_mm=source_points,
            source_weights=source_weights,
            tissue_params=tissue_params,
            z_surface=z_at_source,
            surface_z_values=None,
        )

        metrics_mean_vs_local = compute_metrics(response_mean_z, response_local_z)
        metrics_proj_vs_local = compute_metrics(response_proj_z, response_local_z)

        logger.info(f"  Mean-Z vs Local-Z: NCC={metrics_mean_vs_local['ncc']:.4f}")
        logger.info(f"  Proj-Z vs Local-Z: NCC={metrics_proj_vs_local['ncc']:.4f}")

        results[rs_id] = {
            "surface_z_range": [float(roi_z.min()), float(roi_z.max())],
            "z_mean": float(z_mean),
            "z_at_projection": float(z_at_source),
            "mean_z_vs_local": metrics_mean_vs_local,
            "proj_z_vs_local": metrics_proj_vs_local,
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        scatter = ax.scatter(roi_xy[:, 0], roi_xy[:, 1], c=roi_z, cmap="viridis", s=1)
        ax.scatter([source.center[0]], [source.center[1]], c="red", s=100, marker="x")
        ax.set_title(f"Surface Z (source at z={source.center[2]:.1f})")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(scatter, ax=ax)

        ax = axes[0, 1]
        scatter = ax.scatter(
            roi_xy[:, 0], roi_xy[:, 1], c=response_mean_z, cmap="hot", s=1
        )
        ax.set_title(f"Planar (mean Z={z_mean:.1f})")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(scatter, ax=ax)

        ax = axes[1, 0]
        scatter = ax.scatter(
            roi_xy[:, 0], roi_xy[:, 1], c=response_local_z, cmap="hot", s=1
        )
        ax.set_title("Local surface depth")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(scatter, ax=ax)

        ax = axes[1, 1]
        diff = np.abs(response_local_z - response_mean_z)
        scatter = ax.scatter(roi_xy[:, 0], roi_xy[:, 1], c=diff, cmap="RdBu", s=1)
        ax.set_title(f"Difference (NCC={metrics_mean_vs_local['ncc']:.3f})")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        fig_path = output_path / f"{rs_id}_surface_comparison.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"  Saved: {fig_path}")

    mean_ncc = np.mean([r["mean_z_vs_local"]["ncc"] for r in results.values()])

    summary = {
        "decision": "GO" if mean_ncc > 0.9 else "CAUTION" if mean_ncc > 0.7 else "NOGO",
        "mean_ncc_planar_vs_local": float(mean_ncc),
        "per_config": results,
        "conclusion": f"Planar surface assumption vs local depth: mean NCC = {mean_ncc:.3f}. "
        f"{'Local-surface approximation is important.' if mean_ncc < 0.9 else 'Planar approximation is acceptable.'}",
    }

    summary_path = output_path / "real_surface_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved: {summary_path}")
    logger.info(f"Decision: {summary['decision']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Real surface ablation for E1d")
    parser.add_argument(
        "--config", default="pilot/e1d_finite_source_local_surface/config.yaml"
    )
    parser.add_argument(
        "--output", default="pilot/e1d_finite_source_local_surface/results/real_surface"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_real_surface_ablation(config, args.output)


if __name__ == "__main__":
    main()
