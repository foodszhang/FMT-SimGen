#!/usr/bin/env python3
"""
Step 0b: Generate tetrahedral mesh from Digimouse atlas.

This script:
1. Loads the merged atlas from output/shared/atlas_full.npz
2. Generates tetrahedral mesh using iso2mesh (cgalmesh method)
3. Saves mesh to output/shared/mesh.npz
4. Prints mesh statistics

Usage:
    python scripts/step0b_generate_mesh.py [--downsample N]

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"
VIS_DIR = OUTPUT_DIR / "mesh_vis"

# Trunk bounding box in atlas frame (with margin)
# trunk_offset_mm = [0, 30, 0], trunk_size_mm = [38, 40, 20.8]
TRUNK_MARGIN_MM = 2.0
TRUNK_BBOX_ATLAS = {
    "x": (-1.0, 38.0 + 1.0),
    "y": (30.0 - 40.0 + TRUNK_MARGIN_MM, 30.0 + 40.0 + TRUNK_MARGIN_MM),  # [~ -10, ~72]
    "z": (-1.0, 20.8 + 1.0),
}


def _crop_mesh_to_trunk(mesh_data, bbox_atlas):
    """Crop mesh nodes/elements to trunk bounding box (atlas frame).

    Only keeps nodes within the bounding box and remaps element connectivity.
    Returns new MeshData with cropped mesh, or original if too few nodes remain.
    """
    from dataclasses import replace
    nodes = mesh_data.nodes
    elements = mesh_data.elements

    in_bbox = (
        (nodes[:, 0] >= bbox_atlas["x"][0]) & (nodes[:, 0] <= bbox_atlas["x"][1]) &
        (nodes[:, 1] >= bbox_atlas["y"][0]) & (nodes[:, 1] <= bbox_atlas["y"][1]) &
        (nodes[:, 2] >= bbox_atlas["z"][0]) & (nodes[:, 2] <= bbox_atlas["z"][1])
    )
    keep_idx = np.where(in_bbox)[0]
    if len(keep_idx) < 100:
        logger.warning(f"  Trunk crop: only {len(keep_idx)} nodes, keeping full mesh")
        return mesh_data

    # Remap old node indices to new consecutive indices
    old_to_new = np.full(len(nodes), -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    # Filter elements: all 4 nodes must be in keep_idx
    elem_mask = in_bbox[elements].all(axis=1)
    cropped_elements = old_to_new[elements[elem_mask]]

    cropped_nodes = nodes[keep_idx]
    cropped_tissue_labels = mesh_data.tissue_labels[elem_mask]
    cropped_surface_faces = mesh_data.surface_faces[
        in_bbox[mesh_data.surface_faces].all(axis=1)
    ]
    # Remap surface face indices
    sf_old_to_new = np.full(len(nodes), -1, dtype=np.int64)
    sf_old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)
    cropped_surface_faces = sf_old_to_new[cropped_surface_faces]

    surface_node_indices = np.where(in_bbox)[0]

    logger.info(
        f"  Trunk crop: {len(nodes)} → {len(cropped_nodes)} nodes, "
        f"{len(elements)} → {len(cropped_elements)} elements"
    )

    return replace(
        mesh_data,
        nodes=cropped_nodes,
        elements=cropped_elements,
        tissue_labels=cropped_tissue_labels,
        surface_faces=cropped_surface_faces,
        surface_node_indices=surface_node_indices,
    )


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
        ax.plot(
            pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2], "b-", linewidth=0.5
        )

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=1, c="gray", alpha=0.3)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved surface mesh visualization: {output_path}")


def plot_slice_with_mesh(nodes, elements, slice_axis, slice_idx, title, output_path):
    """Plot a 2D slice showing mesh element centers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    elem_centers = (
        nodes[elements[:, 0]]
        + nodes[elements[:, 1]]
        + nodes[elements[:, 2]]
        + nodes[elements[:, 3]]
    ) / 4.0

    if slice_axis == "x":
        mask = np.abs(elem_centers[:, 0] - slice_idx) < 2
        slice_centers = elem_centers[mask][:, 1:]
    elif slice_axis == "y":
        mask = np.abs(elem_centers[:, 1] - slice_idx) < 2
        slice_centers = elem_centers[mask][:, [0, 2]]
    else:  # z
        mask = np.abs(elem_centers[:, 2] - slice_idx) < 2
        slice_centers = elem_centers[mask][:, :2]

    ax.scatter(slice_centers[:, 0], slice_centers[:, 1], s=5, alpha=0.5)
    ax.set_xlabel("Axis 1 (mm)")
    ax.set_ylabel("Axis 2 (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved slice visualization: {output_path}")


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
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
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
        default=8,
        help="Downsampling factor (default: 8)",
    )
    parser.add_argument(
        "--atlas_path",
        type=str,
        default=None,
        help="Path to atlas npz file (default: output/shared/atlas_full.npz)",
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
    logger.info("Step 0b: Tetrahedral Mesh Generation")
    logger.info("=" * 60)

    if args.atlas_path is None:
        atlas_path = OUTPUT_DIR / "atlas_full.npz"
    else:
        atlas_path = Path(args.atlas_path)

    logger.info(f"Loading atlas from: {atlas_path}")
    atlas_data = np.load(atlas_path, allow_pickle=True)

    if "tissue_labels" in atlas_data:
        tissue_labels = atlas_data["tissue_labels"]
        logger.info(f"Using merged tissue_labels from atlas")
    else:
        tissue_labels = atlas_data["original_labels"]
        logger.info(f"Using original_labels from atlas")

    voxel_size = float(atlas_data["voxel_size"])
    logger.info(f"Voxel size: {voxel_size} mm")
    logger.info(f"Original atlas shape: {tissue_labels.shape}")

    mesh_config = {
        "target_nodes": 10000,
        "surface_maxvol": 0.5,
        "deep_maxvol": 5.0,
        "roi_maxvol": 1.0,
        "output_path": str(OUTPUT_DIR),
    }

    generator = MeshGenerator(mesh_config)

    logger.info(f"Generating mesh with downsample_factor={args.downsample}...")

    mesh_data = generator.generate(
        atlas_volume=tissue_labels,
        voxel_size=voxel_size,
        tissue_labels=None,
        downsample_factor=args.downsample,
    )

    logger.info("=" * 60)
    logger.info("MESH GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Nodes (full body): {mesh_data.nodes.shape[0]}")
    logger.info(f"Elements: {mesh_data.elements.shape[0]}")
    logger.info(f"Tissue labels: {np.unique(mesh_data.tissue_labels)}")
    logger.info(f"Surface faces: {mesh_data.surface_faces.shape[0]}")
    logger.info(f"Surface nodes: {mesh_data.surface_node_indices.shape[0]}")

    # ── Crop to trunk region ──────────────────────────────────────────────────
    mesh_data = _crop_mesh_to_trunk(mesh_data, TRUNK_BBOX_ATLAS)

    mesh_file = str(OUTPUT_DIR / "mesh.npz")
    generator.save(mesh_data, str(OUTPUT_DIR / "mesh"))
    logger.info(f"Mesh saved to: {mesh_file}")

    logger.info("\nGenerating visualizations...")
    try:
        visualize_surface_mesh(
            mesh_data.nodes,
            mesh_data.surface_faces,
            f"Surface Mesh (downsample={args.downsample})",
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

        center_z = mesh_data.nodes[:, 2].mean()
        plot_slice_with_mesh(
            mesh_data.nodes,
            mesh_data.elements,
            "z",
            int(center_z),
            f"Mesh Slice at Z={center_z:.1f}mm",
            VIS_DIR / "mesh_slice_z.png",
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
