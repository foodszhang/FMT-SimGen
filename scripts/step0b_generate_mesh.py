#!/usr/bin/env python3
"""
Step 0b: Generate tetrahedral mesh from canonical trunk volume.

U4: Mesh source changed from atlas_full.npz → trunk_volume.npz.
The trunk_volume is already cropped to the trunk bounding box at 0.2mm
and is directly in trunk-local frame (mcx_trunk_local_mm).
No crop or rebase needed.

Usage:
    python scripts/step0b_generate_mesh.py [--downsample 4]

Output:
    output/shared/mesh.npz
"""
import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fmt_simgen.mesh.mesh_generator import MeshGenerator
from fmt_simgen.frame_contract import TRUNK_SIZE_MM, assert_in_trunk_bbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"
VIS_DIR = OUTPUT_DIR / "mesh_vis"


def visualize_surface_mesh(nodes, faces, title, output_path, max_faces=5000):
    """Visualize surface mesh using matplotlib."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if len(faces) > max_faces:
        indices = np.random.choice(len(faces), max_faces, replace=False)
        faces_plot = faces[indices]
    else:
        faces_plot = faces

    for face in faces_plot:
        pts = nodes[face]
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2], "b-", linewidth=0.5)

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=1, c="gray", alpha=0.3)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved surface mesh visualization: {output_path}")


def plot_element_volume_histogram(volumes, title, output_path):
    """Plot histogram of element volumes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(volumes, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Element Volume (mm³)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    stats_text = (
        f"Min: {np.min(volumes):.6f}\n"
        f"Max: {np.max(volumes):.6f}\n"
        f"Mean: {np.mean(volumes):.6f}\n"
        f"Median: {np.median(volumes):.6f}"
    )
    ax.text(
        0.95, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved volume histogram: {output_path}")


def plot_tissue_label_distribution(labels, title, output_path):
    """Plot distribution of tissue labels in elements."""
    unique_labels, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(unique_labels, counts, edgecolor="black")
    ax.set_xlabel("Tissue Label")
    ax.set_ylabel("Element Count")
    ax.set_title(title)

    for label, count in zip(unique_labels, counts):
        ax.text(label, count, str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved tissue distribution: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 0b: Generate tetrahedral mesh")
    parser.add_argument(
        "--downsample",
        "-d",
        type=int,
        default=4,
        help="Downsampling factor for trunk volume (default: 4, gives ~0.8mm effective voxel)",
    )
    parser.add_argument(
        "--maxvol",
        type=float,
        default=None,
        help="Max tetrahedron volume (mm³). If None, auto-estimate.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 0b: Tetrahedral Mesh Generation (U4 — from trunk_volume.npz)")
    logger.info("=" * 60)

    # ── Load canonical trunk volume ─────────────────────────────────────────────
    trunk_path = OUTPUT_DIR / "trunk_volume.npz"
    logger.info(f"Loading trunk volume from: {trunk_path}")
    trunk_data = np.load(trunk_path, allow_pickle=True)
    trunk_volume = trunk_data["trunk_volume"]
    voxel_size = float(trunk_data["voxel_size_mm"])  # 0.2

    logger.info(f"Trunk volume shape: {trunk_volume.shape} (XYZ)")
    logger.info(f"Trunk volume voxel_size: {voxel_size} mm")
    logger.info(f"Expected shape: (190, 200, 104)")
    assert trunk_volume.shape == (190, 200, 104), (
        f"Unexpected trunk_volume shape {trunk_volume.shape}, expected (190, 200, 104)"
    )

    # ── Generate mesh ──────────────────────────────────────────────────────────
    # downsample=4 at 0.2mm → effective voxel = 0.8mm (same as old pipeline's 8× at 0.1mm)
    mesh_config = {
        "target_nodes": 5000,
        "surface_maxvol": 0.5,
        "deep_maxvol": 5.0,
        "roi_maxvol": 1.0,
        "output_path": str(OUTPUT_DIR),
    }

    generator = MeshGenerator(mesh_config)
    logger.info(f"Generating mesh with downsample_factor={args.downsample}...")

    mesh_data = generator.generate(
        atlas_volume=trunk_volume,
        voxel_size=voxel_size,  # 0.2 mm (trunk voxel size)
        tissue_labels=None,
        downsample_factor=args.downsample,  # 4 (effective 0.8mm, matches old 8× at 0.1mm)
        crop_to_trunk=False,  # trunk_volume is already in trunk-local frame
    )

    logger.info("=" * 60)
    logger.info("MESH GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Nodes: {mesh_data.nodes.shape[0]}")
    logger.info(f"Elements: {mesh_data.elements.shape[0]}")
    logger.info(f"Tissue labels: {np.unique(mesh_data.tissue_labels)}")
    logger.info(f"Surface faces: {mesh_data.surface_faces.shape[0]}")
    logger.info(f"Surface nodes: {mesh_data.surface_node_indices.shape[0]}")

    # ── Assertions: mesh is in trunk-local frame ────────────────────────────────
    logger.info("\n=== U4 Frame Assertions ===")
    assert_in_trunk_bbox(mesh_data.nodes, tol_mm=3.0)

    nodes_min = mesh_data.nodes.min(axis=0)
    nodes_max = mesh_data.nodes.max(axis=0)
    logger.info(f"Mesh bbox min: {nodes_min} mm")
    logger.info(f"Mesh bbox max: {nodes_max} mm")
    logger.info(f"Trunk size:   [0, 0, 0] to {TRUNK_SIZE_MM} mm")

    # Surface edge length stats
    faces = mesh_data.surface_faces
    edge_pairs = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edge_lens = np.linalg.norm(
        mesh_data.nodes[edge_pairs[:, 0]] - mesh_data.nodes[edge_pairs[:, 1]],
        axis=1,
    )
    logger.info(f"Surface edge length: median={np.median(edge_lens):.3f} mm, "
                f"p95={np.percentile(edge_lens, 95):.3f}, max={edge_lens.max():.3f}")

    # Index range check
    assert mesh_data.surface_faces.max() < mesh_data.nodes.shape[0], (
        f"Face index {mesh_data.surface_faces.max()} out of range [{mesh_data.nodes.shape[0]}]"
    )

    # Save
    mesh_file = str(OUTPUT_DIR / "mesh.npz")
    generator.save(mesh_data, str(OUTPUT_DIR / "mesh"))
    logger.info(f"Mesh saved to: {mesh_file}")

    # ── Visualizations ─────────────────────────────────────────────────────────
    logger.info("\nGenerating visualizations...")
    try:
        visualize_surface_mesh(
            mesh_data.nodes,
            mesh_data.surface_faces,
            f"Surface Mesh (downsample={args.downsample}, trunk-local)",
            VIS_DIR / "mesh_surface.png",
        )

        elem_volumes = generator._compute_tetrahedron_volumes(
            mesh_data.nodes, mesh_data.elements
        )
        plot_element_volume_histogram(
            elem_volumes,
            f"Element Volume Distribution (downsample={args.downsample})",
            VIS_DIR / "mesh_volumes.png",
        )

        plot_tissue_label_distribution(
            mesh_data.tissue_labels,
            f"Tissue Label Distribution (downsample={args.downsample})",
            VIS_DIR / "mesh_labels.png",
        )

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("OUTPUT FILES")
    logger.info("=" * 60)
    logger.info(f"  Mesh data: {mesh_file}")
    logger.info(f"  Visualizations: {VIS_DIR}/")


if __name__ == "__main__":
    main()
