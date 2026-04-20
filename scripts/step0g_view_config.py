#!/usr/bin/env python
"""
Step 0g: Generate view configuration and validate TurntableCamera.

This script:
1. Loads the FEM mesh to get surface nodes and faces
2. Computes surface normals from surface faces
3. Runs TurntableCamera visibility analysis for all angles
4. Generates visualization plots (XY, XZ, YZ projections per angle)
5. Saves view_config.json to output/shared/

Usage:
    python scripts/step0g_view_config.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.view_config import TurntableCamera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_mesh(mesh_path: str) -> dict:
    """Load mesh data from npz file."""
    data = np.load(mesh_path)
    return {
        "nodes": data["nodes"].astype(np.float64),
        "elements": data["elements"],
        "surface_faces": data["surface_faces"],
        "surface_node_indices": data["surface_node_indices"],
    }


def plot_visible_nodes_per_angle(
    camera: TurntableCamera,
    node_coords: np.ndarray,
    all_visible: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """Generate scatter plots of visible nodes for each angle.

    Creates a figure with 7 angles × 3 projections (XY, XZ, YZ).
    """
    angles = camera.angles
    n_angles = len(angles)

    fig, axes = plt.subplots(n_angles, 3, figsize=(12, 2.5 * n_angles))
    if n_angles == 1:
        axes = axes.reshape(1, -1)

    # Transpose so rows are projections and columns are angles
    # Actually keep as: rows = angles, cols = projections
    for row, angle in enumerate(angles):
        visible_idx = all_visible[angle]
        visible_nodes = node_coords[visible_idx]

        # XY projection (top-down view of body cross-section)
        ax = axes[row, 0]
        ax.scatter(visible_nodes[:, 0], visible_nodes[:, 1], s=0.5, alpha=0.5)
        ax.set_title(f"Angle {angle}°: XY (Top-down)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # XZ projection (lateral view, shows dorsal/ventral)
        ax = axes[row, 1]
        ax.scatter(visible_nodes[:, 0], visible_nodes[:, 2], s=0.5, alpha=0.5)
        ax.set_title(f"Angle {angle}°: XZ (Lateral)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # YZ projection (front/back view)
        ax = axes[row, 2]
        ax.scatter(visible_nodes[:, 1], visible_nodes[:, 2], s=0.5, alpha=0.5)
        ax.set_title(f"Angle {angle}°: YZ (Front)")
        ax.set_xlabel("Y (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Visualization saved to {output_path}")


def main() -> None:
    config_path = Path("config/default.yaml")
    mesh_path = Path("output/shared/mesh.npz")
    output_dir = Path("output/shared")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    view_config = full_config.get("view_config", {})
    logger.info(f"View config: {view_config}")

    # Load mesh
    logger.info(f"Loading mesh from {mesh_path}")
    mesh = load_mesh(str(mesh_path))
    nodes = mesh["nodes"]
    surface_faces = mesh["surface_faces"]
    surface_node_indices = mesh["surface_node_indices"]

    logger.info(f"Mesh: {nodes.shape[0]} nodes, {surface_faces.shape[0]} surface faces")
    logger.info(f"Surface nodes: {len(surface_node_indices)}")

    # Create camera
    camera = TurntableCamera(view_config)

    # Compute surface normals
    logger.info("Computing surface normals...")
    normals = camera.compute_surface_normals(nodes, surface_faces)

    # Verify normal direction
    # For prone pose, normals at dorsal (high Z) should point mostly upward (+Z)
    dorsal_nodes = nodes[surface_node_indices]
    dorsal_normals = normals[surface_node_indices]
    dorsal_z_mean = dorsal_nodes[:, 2].mean()
    logger.info(f"Dorsal region Z mean: {dorsal_z_mean:.2f} mm")
    logger.info(f"Dorsal normals Z component mean: {dorsal_normals[:, 2].mean():.3f}")

    # Get visible nodes for all angles
    logger.info("Computing visibility for all angles...")
    all_visible = camera.get_all_visible_nodes_per_angle(nodes, normals)

    # Compute statistics
    total_surface = len(surface_node_indices)
    stats = {}
    all_visible_combined = set()

    for angle, visible_idx in all_visible.items():
        n_visible = len(visible_idx)
        fraction = n_visible / total_surface * 100
        stats[angle] = {
            "n_visible": int(n_visible),
            "fraction_of_surface": round(fraction, 2),
        }
        all_visible_combined.update(visible_idx.tolist())

    # Build visible mask: map global node indices to local surface node indices
    # visible_mask[i] = True if surface_node_indices[i] is visible from at least one angle
    logger.info("Building visible mask...")
    visible_mask = np.zeros(total_surface, dtype=bool)
    for global_idx in all_visible_combined:
        # Find local index in surface_node_indices
        local_idx = np.where(surface_node_indices == global_idx)[0]
        if len(local_idx) > 0:
            visible_mask[local_idx[0]] = True

    n_visible_union = int(np.count_nonzero(visible_mask))
    union_fraction = n_visible_union / total_surface * 100
    logger.info(
        f"Union of all visible nodes: {n_visible_union} / {total_surface} ({union_fraction:.1f}%)"
    )

    # Save visible mask
    visible_mask_path = output_dir / "visible_mask.npy"
    np.save(visible_mask_path, visible_mask)
    logger.info(f"Visible mask saved to {visible_mask_path}")

    # Find angle with most visible nodes
    angle_most_visible = max(stats.keys(), key=lambda a: stats[a]["n_visible"])
    logger.info(
        f"Angle with most visible nodes: {angle_most_visible}° ({stats[angle_most_visible]['n_visible']})"
    )

    # Generate visualization
    vis_path = output_dir / "view_config_vis.png"
    logger.info("Generating visualization...")
    plot_visible_nodes_per_angle(camera, nodes, all_visible, vis_path)

    # Build view_config.json
    view_config_json = {
        "angles": camera.angles,
        "pose": camera.pose,
        "camera_distance_mm": camera.camera_distance_mm,
        "detector_resolution": list(camera.detector_resolution),
        "projection_type": camera.projection_type,
        "fov_mm": camera.fov_mm,
        "stats_per_angle": stats,
        "total_surface_nodes": total_surface,
        "union_visible_nodes": n_visible_union,
        "union_fraction_percent": round(union_fraction, 2),
        "visible_mask_file": "visible_mask.npy",
        "n_visible": n_visible_union,
        "visible_ratio": round(n_visible_union / total_surface, 4),
        "angle_most_visible": angle_most_visible,
        "most_visible_count": stats[angle_most_visible]["n_visible"],
    }

    json_path = output_dir / "view_config.json"
    with open(json_path, "w") as f:
        json.dump(view_config_json, f, indent=2)
    logger.info(f"View config saved to {json_path}")

    # Print summary
    logger.info("\n=== Visibility Summary ===")
    for angle in sorted(stats.keys()):
        s = stats[angle]
        logger.info(
            f"  {angle:4d}°: {s['n_visible']:4d} nodes ({s['fraction_of_surface']:.1f}%)"
        )
    logger.info(f"  Union: {len(all_visible_combined)} nodes ({union_fraction:.1f}%)")

    # Acceptance checks
    logger.info("\n=== Acceptance Checks ===")

    # Check 1: Each angle has 2000-5000 visible nodes
    checks_passed = True
    for angle in camera.angles:
        n = stats[angle]["n_visible"]
        if 2000 <= n <= 5000:
            logger.info(
                f"  [PASS] Angle {angle}°: {n} visible nodes (in 2000-5000 range)"
            )
        else:
            logger.warning(
                f"  [WARN] Angle {angle}°: {n} visible nodes (outside 2000-5000 range)"
            )
            checks_passed = False

    # Check 2: Union covers >60% of surface nodes
    if union_fraction > 60:
        logger.info(f"  [PASS] Union visible: {union_fraction:.1f}% (>60%)")
    else:
        logger.warning(f"  [FAIL] Union visible: {union_fraction:.1f}% (<=60%)")
        checks_passed = False

    # Check 3: 0° has most visible nodes
    if angle_most_visible == 0:
        logger.info(f"  [PASS] 0° has most visible nodes ({stats[0]['n_visible']})")
    else:
        logger.warning(f"  [WARN] {angle_most_visible}° has most visible nodes, not 0°")

    # Check 4: ±90° have fewer visible nodes
    avg_90 = (
        stats.get(90, {}).get("n_visible", 0) + stats.get(-90, {}).get("n_visible", 0)
    ) / 2
    avg_0 = stats.get(0, {}).get("n_visible", 0)
    if avg_90 < avg_0:
        logger.info(f"  [PASS] ±90° ({avg_90:.0f}) fewer than 0° ({avg_0:.0f})")
    else:
        logger.warning(f"  [WARN] ±90° ({avg_90:.0f}) not fewer than 0° ({avg_0:.0f})")

    if checks_passed:
        logger.info("\nAll acceptance checks passed!")
    else:
        logger.warning("\nSome acceptance checks had warnings.")


if __name__ == "__main__":
    main()
