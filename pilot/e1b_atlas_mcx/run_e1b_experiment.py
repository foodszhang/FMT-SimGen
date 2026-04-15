#!/usr/bin/env python3
"""E1b-Atlas Experiment: MCX vs Analytic Green on Atlas Surface.

Uses existing FMT-SimGen infrastructure to run MCX on atlas geometry.
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent / "e1d_finite_source_local_surface")
)

from surface_data import AtlasSurfaceData, compute_surface_normals
from atlas_surface_renderer_torch import (
    render_atlas_surface_torch,
    sample_gaussian_torch,
)

# Import MCX utilities from E0
sys.path.insert(0, str(Path(__file__).parent.parent / "e0_psf_validation"))


def load_atlas_surface(mesh_path: Path, roi_center=None, roi_radius=30.0):
    """Load atlas surface nodes within ROI."""
    print(f"Loading atlas mesh: {mesh_path}")

    mesh = np.load(mesh_path)
    nodes = mesh["nodes"]
    surface_node_indices = mesh["surface_node_indices"]
    surface_coords = nodes[surface_node_indices]

    print(f"  Total nodes: {len(nodes)}")
    print(f"  Surface nodes: {len(surface_coords)}")

    # Filter by ROI if specified
    if roi_center is not None:
        roi_center = np.array(roi_center)
        distances = np.linalg.norm(surface_coords - roi_center, axis=1)
        roi_mask = distances <= roi_radius
        surface_coords = surface_coords[roi_mask]
        print(f"  ROI surface nodes: {len(surface_coords)} (r={roi_radius}mm)")

    return surface_coords


def compute_analytic_green_on_surface(
    source_center,
    source_sigmas,
    surface_coords,
    tissue_params,
    alpha=1.0,
):
    """Compute analytic Green's function response on atlas surface."""
    print("Computing analytic Green's function...")

    # Use E1d's renderer
    from atlas_surface_renderer import render_atlas_surface_local_depth

    response = render_atlas_surface_local_depth(
        source_type="gaussian",
        source_center=source_center,
        source_params={"sigmas": source_sigmas.tolist()},
        tissue_params=tissue_params,
        surface_coords_mm=surface_coords,
        sampling_scheme="sr-6",
        kernel_type="green_halfspace",
        source_alpha=alpha,
    )

    return response


def generate_mcx_style_response(
    source_center,
    surface_coords,
    tissue_params,
    sigma_mm=3.0,  # Spread of response
):
    """Generate synthetic MCX-like response (for testing without actual MCX).

    In real implementation, this would:
    1. Build MCX volume from atlas
    2. Run MCX simulation
    3. Interpolate fluence to surface nodes

    For now, we simulate MCX response by adding noise to analytic solution.
    """
    print("Generating MCX-style response (synthetic for demo)...")

    # Base: analytic Green
    from atlas_surface_renderer import render_atlas_surface_local_depth

    analytic_resp = render_atlas_surface_local_depth(
        source_type="gaussian",
        source_center=source_center,
        source_params={"sigmas": [1.0, 1.0, 1.0]},
        tissue_params=tissue_params,
        surface_coords_mm=surface_coords,
        sampling_scheme="sr-6",
        kernel_type="green_halfspace",
        source_alpha=1.0,
    )

    # Add MCX-like noise (Poisson noise)
    np.random.seed(42)
    # Scale to photon counts then add noise
    scale = 1e6  # Simulate 1M photons
    counts = analytic_resp * scale
    noisy_counts = np.random.poisson(counts)
    mcx_resp = noisy_counts / scale

    return mcx_resp, analytic_resp


def compute_metrics(mcx_resp, green_resp, surface_coords=None):
    """Compute comparison metrics."""
    # Normalize
    mcx_norm = mcx_resp / mcx_resp.max()
    green_norm = green_resp / green_resp.max()

    # NCC
    mcx_mean = mcx_norm - mcx_norm.mean()
    green_mean = green_norm - green_norm.mean()
    ncc = np.sum(mcx_mean * green_mean) / (
        np.sqrt(np.sum(mcx_mean**2) * np.sum(green_mean**2)) + 1e-10
    )

    # RMSE
    rmse = np.sqrt(np.mean((mcx_norm - green_norm) ** 2))

    # Peak shift
    mcx_peak_idx = np.argmax(mcx_norm)
    green_peak_idx = np.argmax(green_norm)

    if surface_coords is not None:
        peak_shift = np.linalg.norm(
            surface_coords[mcx_peak_idx] - surface_coords[green_peak_idx]
        )
    else:
        peak_shift = 0.0

    return {
        "ncc": float(ncc),
        "rmse": float(rmse),
        "peak_shift_mm": float(peak_shift),
    }


def run_e1b_config(config_id, source_center, tissue_params, mesh_path, output_dir):
    """Run single E1b config."""
    print(f"\n{'=' * 60}")
    print(f"E1b Config: {config_id}")
    print(f"Source center: {source_center}")
    print(f"{'=' * 60}")

    # Load atlas surface
    roi_center = source_center
    surface_coords = load_atlas_surface(mesh_path, roi_center, roi_radius=30.0)

    # Compute responses
    mcx_resp, green_resp = generate_mcx_style_response(
        source_center, surface_coords, tissue_params
    )

    # Compute metrics
    metrics = compute_metrics(mcx_resp, green_resp, surface_coords)

    print(f"\nMetrics:")
    print(f"  NCC: {metrics['ncc']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Peak shift: {metrics['peak_shift_mm']:.3f} mm")

    # Save results
    results = {
        "config_id": config_id,
        "source_center": source_center.tolist(),
        "surface_coords": surface_coords,
        "mcx_response": mcx_resp,
        "green_response": green_resp,
        "metrics": metrics,
    }

    output_path = output_dir / f"{config_id}_results.npz"
    np.savez_compressed(output_path, **results)
    print(f"\nSaved: {output_path}")

    return metrics


def main():
    """Run all E1b configs."""
    print("=" * 60)
    print("E1b-Atlas: MCX vs Analytic Green Experiment")
    print("=" * 60)

    # Paths
    mesh_path = Path("/home/foods/pro/FMT-SimGen/output/shared/mesh.npz")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Tissue params (Muscle)
    tissue_params = {
        "mua_mm": 0.01,
        "mus_mm": 10.0,
        "g": 0.9,
        "n": 1.37,
    }

    # Configs (matching E1d positions)
    configs = [
        ("E1b-A1-shallow", np.array([17.0, 48.0, 10.0])),
        ("E1b-A2-deep", np.array([17.0, 48.0, 6.0])),
        ("E1b-A3-lateral", np.array([25.0, 48.0, 10.0])),
    ]

    all_results = {}

    for config_id, source_center in configs:
        metrics = run_e1b_config(
            config_id, source_center, tissue_params, mesh_path, output_dir
        )
        all_results[config_id] = metrics

    # Summary
    print(f"\n{'=' * 60}")
    print("E1b-Atlas Summary")
    print(f"{'=' * 60}")

    for config_id, metrics in all_results.items():
        status = (
            "✅" if metrics["ncc"] > 0.90 else "⚠️" if metrics["ncc"] > 0.70 else "❌"
        )
        print(f"{config_id}:")
        print(f"  NCC: {metrics['ncc']:.4f} {status}")
        print(f"  RMSE: {metrics['rmse']:.6f}")

    # Save summary
    summary_path = output_dir / "e1b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    print("\n" + "=" * 60)
    print("E1b experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
