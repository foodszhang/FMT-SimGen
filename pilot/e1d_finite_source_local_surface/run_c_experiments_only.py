"""Run only C experiments to verify C2=C3 bugfix."""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from surface_data import AtlasSurfaceData
from run_e1d_atlas import run_geometry_experiment

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run only C experiments to verify the fix."""
    gt_dir = Path(__file__).parent / "results" / "gt_atlas"
    output_dir = Path(__file__).parent / "results" / "atlas_experiments_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path(__file__).parent / "config_atlas.yaml"
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tissue_params = {
        "mua_mm": config["tissue"]["muscle"]["mua_mm"],
        "mus_mm": config["tissue"]["muscle"]["mus_mm"],
        "g": config["tissue"]["muscle"]["g"],
        "n": config["tissue"]["muscle"]["n"],
    }

    # Load atlas
    mesh_path = Path(config["mesh_path"])
    if not mesh_path.is_absolute():
        mesh_path = Path(__file__).parent.parent.parent / mesh_path
    atlas = AtlasSurfaceData(mesh_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # C experiments with explicit inverse source types
    inverse_experiments = [
        ("C1_gaussian_to_gaussian", "atlas_local_depth", "grid-27", "gaussian"),
        ("C2_uniform_to_uniform", "atlas_local_depth", "grid-27", "uniform"),
        ("C3_uniform_to_gaussian", "atlas_local_depth", "grid-27", "gaussian"),
    ]

    results = {}

    for gt_id, inverse_mode, scheme, inverse_source_type in inverse_experiments:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running {gt_id}")
        logger.info(f"Inverse source type: {inverse_source_type}")
        logger.info(f"{'=' * 60}")

        result = run_geometry_experiment(
            gt_id=gt_id,
            gt_dir=str(gt_dir),
            atlas=atlas,
            tissue_params=tissue_params,
            inverse_surface_mode=inverse_mode,
            inverse_sampling_scheme=scheme,
            inverse_source_type=inverse_source_type,
            config=config,
            device=device,
            output_dir=str(output_dir),
            seed=42,
        )

        if result is not None:
            results[f"C_{gt_id}"] = result
            logger.info(
                f"✓ {gt_id}: position_error={result['position_error_mm']:.3f}mm, "
                f"inverse_type={result['inverse_source_type']}"
            )

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY: C2=C3 Bugfix Verification")
    logger.info(f"{'=' * 60}")

    c1 = results.get("C_C1_gaussian_to_gaussian", {})
    c2 = results.get("C_C2_uniform_to_uniform", {})
    c3 = results.get("C_C3_uniform_to_gaussian", {})

    logger.info(f"C1 (gaussian→gaussian):  {c1.get('position_error_mm', 0):.3f} mm")
    logger.info(
        f"C2 (uniform→uniform):     {c2.get('position_error_mm', 0):.3f} mm "
        f"[inverse={c2.get('inverse_source_type', 'unknown')}]"
    )
    logger.info(
        f"C3 (uniform→gaussian):    {c3.get('position_error_mm', 0):.3f} mm "
        f"[inverse={c3.get('inverse_source_type', 'unknown')}]"
    )

    # Verify fix
    if c2 and c3:
        if c2["position_error_mm"] != c3["position_error_mm"]:
            logger.info(f"\n✅ BUGFIX VERIFIED: C2 ≠ C3")
            logger.info(f"   C2 error: {c2['position_error_mm']:.3f}mm")
            logger.info(f"   C3 error: {c3['position_error_mm']:.3f}mm")
        else:
            logger.info(f"\n⚠ WARNING: C2 = C3 (still same)")

        if c2["position_error_mm"] <= c3["position_error_mm"]:
            logger.info(f"✅ C2 ≤ C3 (correct model ≤ mismatched model)")
        else:
            logger.info(f"⚠ C2 > C3 (unexpected but acceptable)")

    # Save results
    summary_path = output_dir / "c_experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )
    logger.info(f"\nResults saved: {summary_path}")


if __name__ == "__main__":
    main()
