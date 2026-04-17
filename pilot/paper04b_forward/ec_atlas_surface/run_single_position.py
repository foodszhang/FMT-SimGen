"""Run single position MCX vs Green comparison.

Usage:
    uv run python run_single_position.py --position P1 --y 10.0
    uv run python run_single_position.py --all --y 10.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pilot.paper04b_forward.shared.config import ECConfig, get_default_config
from pilot.paper04b_forward.shared.surface_coords import SurfacePosition
from pilot.paper04b_forward.shared.metrics import compute_all_metrics
from pilot.paper04b_forward.shared.green_surface_projection import (
    render_green_surface_projection,
)
from pilot.paper04b_forward.shared.source_label_preflight import preflight_check

logger = logging.getLogger(__name__)


def load_atlas_volume(volume_path: Path, volume_shape: tuple) -> np.ndarray:
    """Load atlas volume and reshape."""
    volume = np.fromfile(volume_path, dtype=np.uint8)
    return volume.reshape(volume_shape)


def load_mcx_projection(result_dir: Path, angle: int) -> np.ndarray:
    """Load MCX projection from result directory."""
    proj_path = result_dir / f"mcx_a{angle}.npy"
    if not proj_path.exists():
        raise FileNotFoundError(f"MCX projection not found: {proj_path}")
    return np.load(proj_path)


def run_green_projection(
    source_pos: np.ndarray,
    atlas_volume: np.ndarray,
    angle_deg: int,
    config: ECConfig,
) -> np.ndarray:
    """Compute Green function projection."""
    binary_mask = atlas_volume > 0

    projection = render_green_surface_projection(
        source_pos_mm=source_pos,
        atlas_volume_binary=binary_mask,
        angle_deg=angle_deg,
        camera_distance_mm=config.camera_distance_mm,
        fov_mm=config.fov_mm,
        detector_resolution=config.detector_resolution,
        tissue_params=config.tissue_params,
        voxel_size_mm=config.voxel_size_mm,
    )
    return projection


def run_single_position(
    position_key: str,
    config: ECConfig,
    output_dir: Path,
    run_mcx: bool = True,
    strict_preflight: bool = False,
) -> Dict:
    """Run MCX vs Green comparison for a single position.

    Parameters
    ----------
    position_key : str
        Position key ("P1", "P2", etc.)
    config : ECConfig
        Experiment configuration.
    output_dir : Path
        Output directory.
    run_mcx : bool
        If True, run MCX simulation. If False, load existing results.
    strict_preflight : bool
        If True, fail on preflight error.

    Returns
    -------
    dict with metrics and metadata.
    """
    positions = config.get_positions()
    if position_key not in positions:
        raise ValueError(f"Unknown position: {position_key}")

    pos = positions[position_key]

    volume_path = config.get_mcx_volume_path()
    material_path = config.get_mcx_material_path()

    if not volume_path.exists():
        raise FileNotFoundError(f"MCX volume not found: {volume_path}")

    is_valid = preflight_check(
        pos.xyz_mm,
        pos.name,
        volume_path,
        material_path,
        config.volume_shape,
        config.voxel_size_mm,
        strict=strict_preflight,
    )

    atlas_volume = load_atlas_volume(volume_path, config.volume_shape)

    green_proj = run_green_projection(pos.xyz_mm, atlas_volume, pos.best_angle, config)

    pos_output_dir = output_dir / f"S2-Vol-{pos.name}-y{config.y_mm}"
    pos_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(pos_output_dir / "green.npy", green_proj)

    if run_mcx:
        from fmt_simgen.mcx_runner import run_mcx_simulation

        logger.info(f"Running MCX for {pos.name} with {config.n_photons} photons...")
        mcx_proj = run_mcx_simulation(
            source_pos_mm=pos.xyz_mm,
            angle_deg=pos.best_angle,
            n_photons=config.n_photons,
            output_dir=pos_output_dir,
        )
    else:
        mcx_proj = load_mcx_projection(pos_output_dir, pos.best_angle)

    np.save(pos_output_dir / f"mcx_a{pos.best_angle}.npy", mcx_proj)

    metrics = compute_all_metrics(mcx_proj, green_proj)

    result = {
        "position": pos.name,
        "position_key": position_key,
        "xyz_mm": pos.xyz_mm.tolist(),
        "angle": pos.best_angle,
        "y_mm": config.y_mm,
        "is_out_of_scope": config.is_out_of_scope,
        "preflight_valid": is_valid,
        "n_photons": config.n_photons,
        **metrics,
    }

    with open(pos_output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"{pos.name}: NCC={metrics['ncc']:.4f}, k={metrics['scale_factor']:.2e}, "
        f"valid={is_valid}"
    )

    return result


def run_all_positions(
    config: ECConfig,
    output_dir: Path,
    run_mcx: bool = True,
    strict_preflight: bool = False,
) -> Dict[str, Dict]:
    """Run all positions."""
    results = {}
    for key in ["P1", "P2", "P3", "P4", "P5"]:
        try:
            results[key] = run_single_position(
                key, config, output_dir, run_mcx, strict_preflight
            )
        except Exception as e:
            logger.error(f"Failed to run {key}: {e}")
            results[key] = {"error": str(e)}

    summary_path = output_dir / f"ec_y{int(config.y_mm)}_5positions.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run E-C single position")
    parser.add_argument("--position", type=str, help="Position key (P1-P5)")
    parser.add_argument("--all", action="store_true", help="Run all positions")
    parser.add_argument("--y", type=float, default=10.0, help="Y coordinate in mm")
    parser.add_argument("--n_photons", type=int, default=1_000_000_000)
    parser.add_argument("--output", type=str, default="results_final")
    parser.add_argument("--no_mcx", action="store_true", help="Skip MCX, use existing")
    parser.add_argument("--strict", action="store_true", help="Fail on preflight error")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = ECConfig(y_mm=args.y, n_photons=args.n_photons)
    output_dir = Path(__file__).parent.parent / args.output

    if args.all:
        results = run_all_positions(
            config, output_dir, run_mcx=not args.no_mcx, strict_preflight=args.strict
        )
        print(f"\nResults saved to {output_dir}")
        for key, r in results.items():
            if "ncc" in r:
                print(
                    f"  {key}: NCC={r['ncc']:.4f}, valid={r.get('preflight_valid', 'N/A')}"
                )
    elif args.position:
        result = run_single_position(
            args.position,
            config,
            output_dir,
            run_mcx=not args.no_mcx,
            strict_preflight=args.strict,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
