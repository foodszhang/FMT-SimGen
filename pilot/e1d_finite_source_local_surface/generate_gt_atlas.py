#!/usr/bin/env python3
"""Generate GT on atlas surface for E1d-R2 experiments.

Generates two types of GT:
- GT-A: Atlas self-consistent (atlas geometry + high-fidelity quadrature)
- GT-B: Atlas geometry mismatch test data
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml

import sys

sys.path.insert(0, "/home/foods/pro/FMT-SimGen/pilot/e1d_finite_source_local_surface")

from surface_data import AtlasSurfaceData
from atlas_surface_renderer import (
    render_atlas_surface_local_depth,
    render_atlas_surface_local_plane,
    render_atlas_surface_flat,
)

logger = logging.getLogger(__name__)


def generate_atlas_gt(
    source_type: str,
    source_center: np.ndarray,
    source_params: dict,
    source_alpha: float,
    tissue_params: dict,
    surface_coords: np.ndarray,
    surface_normals: Optional[np.ndarray],
    roi_indices: np.ndarray,
    sampling_scheme: str,
    kernel_type: str,
    geometry_mode: str,
) -> Dict:
    """Generate GT on atlas surface.

    Args:
        source_type: "point", "gaussian", or "uniform"
        source_center: [3] source center
        source_params: source parameters
        source_alpha: source intensity
        tissue_params: optical properties
        surface_coords: [N, 3] surface coordinates
        surface_normals: [N, 3] or None
        roi_indices: ROI indices
        sampling_scheme: quadrature scheme
        kernel_type: kernel type
        geometry_mode: geometry mode

    Returns:
        dict with response and metadata
    """
    roi_coords = surface_coords[roi_indices]
    roi_normals = surface_normals[roi_indices] if surface_normals is not None else None

    if geometry_mode == "local_depth":
        response = render_atlas_surface_local_depth(
            source_type=source_type,
            source_center=source_center,
            source_params=source_params,
            tissue_params=tissue_params,
            surface_coords_mm=roi_coords,
            sampling_scheme=sampling_scheme,
            kernel_type=kernel_type,
            source_alpha=source_alpha,
        )
    elif geometry_mode == "local_plane":
        response = render_atlas_surface_local_plane(
            source_type=source_type,
            source_center=source_center,
            source_params=source_params,
            tissue_params=tissue_params,
            surface_coords_mm=roi_coords,
            surface_normals_mm=roi_normals,
            sampling_scheme=sampling_scheme,
            kernel_type=kernel_type,
            source_alpha=source_alpha,
        )
    else:
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")

    return {
        "response": response,
        "surface_coords": roi_coords,
        "surface_normals": roi_normals,
        "roi_indices": roi_indices,
        "source_type": source_type,
        "source_center": source_center.tolist(),
        "source_params": source_params,
        "source_alpha": float(source_alpha),
        "sampling_scheme": sampling_scheme,
        "kernel_type": kernel_type,
        "geometry_mode": geometry_mode,
    }


def run_gt_generation(
    config: Dict,
    output_dir: str,
) -> Dict:
    """Generate all GT data for E1d-R2.

    Args:
        config: experiment configuration
        output_dir: output directory

    Returns:
        summary dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tissue_params = config["tissue"]["muscle"]

    atlas = AtlasSurfaceData(
        mesh_path=config.get("mesh_path", "output/shared/mesh.npz"),
        compute_normals=True,
    )

    logger.info(f"Loaded atlas: {len(atlas.surface_coords)} surface nodes")
    logger.info(f"Surface Z range: {atlas.surface_z_range}")

    results = {}

    gt_configs = config.get("atlas_gt_configs", [])

    for gt_config in gt_configs:
        gt_id = gt_config["id"]
        logger.info(f"\nGenerating GT: {gt_id}")

        source_type = gt_config["source"]["type"]
        source_center = np.array(gt_config["source"]["center"], dtype=np.float32)
        source_alpha = gt_config["source"].get("alpha", 1.0)

        if source_type == "gaussian":
            source_params = {
                "sigmas": gt_config["source"].get("sigmas", [1.0, 1.0, 1.0])
            }
        elif source_type == "uniform":
            source_params = {"axes": gt_config["source"].get("axes", [1.0, 1.0, 1.0])}
        else:
            source_params = {}

        roi_radius = gt_config.get("roi_radius_mm", 30.0)
        roi_coords, roi_normals, roi_indices = atlas.get_roi(source_center, roi_radius)

        logger.info(f"  ROI: {len(roi_indices)} nodes, radius={roi_radius}mm")

        sampling_scheme = gt_config.get("sampling_scheme", "grid-27")
        kernel_type = gt_config.get("kernel", "green_halfspace")
        geometry_mode = gt_config.get("geometry_mode", "local_depth")

        gt_data = generate_atlas_gt(
            source_type=source_type,
            source_center=source_center,
            source_params=source_params,
            source_alpha=source_alpha,
            tissue_params=tissue_params,
            surface_coords=atlas.surface_coords,
            surface_normals=atlas.surface_normals,
            roi_indices=roi_indices,
            sampling_scheme=sampling_scheme,
            kernel_type=kernel_type,
            geometry_mode=geometry_mode,
        )

        gt_path = output_path / f"{gt_id}_gt.npz"
        np.savez(
            gt_path,
            response=gt_data["response"],
            surface_coords=gt_data["surface_coords"],
            surface_normals=gt_data["surface_normals"]
            if gt_data["surface_normals"] is not None
            else np.array([]),
            roi_indices=gt_data["roi_indices"],
        )

        meta = {
            "id": gt_id,
            "source_type": source_type,
            "source_center": gt_data["source_center"],
            "source_params": source_params,
            "source_alpha": gt_data["source_alpha"],
            "sampling_scheme": sampling_scheme,
            "kernel_type": kernel_type,
            "geometry_mode": geometry_mode,
            "roi_radius_mm": roi_radius,
            "n_roi_nodes": len(roi_indices),
        }

        meta_path = output_path / f"{gt_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        results[gt_id] = {
            "gt_path": str(gt_path),
            "meta_path": str(meta_path),
            "n_roi_nodes": len(roi_indices),
        }

        logger.info(f"  Saved: {gt_path}")

    summary_path = output_path / "gt_atlas_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSummary saved: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate atlas surface GT for E1d-R2")
    parser.add_argument(
        "--config",
        default="pilot/e1d_finite_source_local_surface/config_atlas.yaml",
        help="Experiment config file",
    )
    parser.add_argument(
        "--output",
        default="pilot/e1d_finite_source_local_surface/results/gt_atlas",
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
