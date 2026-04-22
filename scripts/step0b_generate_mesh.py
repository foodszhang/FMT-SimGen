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

from fmt_simgen.mesh import make_mesh_generator
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


def filter_exterior_faces(
    elements: np.ndarray,
    surface_faces: np.ndarray,
) -> np.ndarray:
    """Filter surface faces to exterior hull only (adjacent to exactly 1 tet).

    Interior label-boundary faces are adjacent to 2 tetrahedra (different labels).
    Exterior hull faces are adjacent to exactly 1 tetrahedron — the true outer surface.

    Parameters
    ----------
    elements : np.ndarray [M×4]
        Tetrahedron node indices.
    surface_faces : np.ndarray [F×3]
        All surface triangle vertex indices.

    Returns
    -------
    np.ndarray [E×3]
        Exterior hull faces only (subset of surface_faces).
    """
    # Count how many tets each face belongs to
    face_counts: dict[tuple[int, int, int], int] = {}
    for elem in elements:
        n0, n1, n2, n3 = elem
        # 4 faces per tet (sorted so (a,b,c) == (c,a,b) etc.)
        for face in [
            tuple(sorted((n0, n1, n2))),
            tuple(sorted((n0, n1, n3))),
            tuple(sorted((n0, n2, n3))),
            tuple(sorted((n1, n2, n3))),
        ]:
            face_counts[face] = face_counts.get(face, 0) + 1

    # Build set of exterior faces (adjacent to exactly 1 tet)
    exterior_set = {
        face for face, count in face_counts.items() if count == 1
    }

    # Filter surface_faces to those in exterior_set
    exterior_faces = []
    for face in surface_faces:
        key = tuple(sorted(face))
        if key in exterior_set:
            exterior_faces.append(face)

    exterior_faces = np.array(exterior_faces, dtype=surface_faces.dtype)
    n_total = len(surface_faces)
    n_exterior = len(exterior_faces)
    logger.info(
        f"Exterior hull filter: {n_exterior}/{n_total} faces "
        f"({100 * n_exterior / max(n_total, 1):.1f}%)"
    )
    return exterior_faces


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
    parser.add_argument(
        "--backend",
        type=str,
        default="iso2mesh",
        choices=["iso2mesh", "amira"],
        help="Mesh generation backend.",
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
        "mesh_backend": args.backend,
    }

    generator = make_mesh_generator(mesh_config)
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

    # ========== 诊断：这个 mesh 到底是哪种病 ==========
    from scipy.spatial import cKDTree

    mesh_nodes = mesh_data.nodes
    mesh_elem  = mesh_data.elements
    mesh_labels = mesh_data.tissue_labels
    sfaces = mesh_data.surface_faces

    N, T, F = len(mesh_nodes), len(mesh_elem), len(sfaces)
    V_surf = len(np.unique(sfaces))
    eff_vs = 0.1 * 8  # effective_voxel_size, mm

    print("\n========== DIAG ==========")
    print(f"N={N}, T={T}, F={F}, V_surf={V_surf}, F/V={F/V_surf:.3f}")

    # [1] tissue label 分布
    uniq_labels, counts = np.unique(mesh_labels, return_counts=True)
    print(f"[1] #tissue_labels = {len(uniq_labels)}: "
          f"{dict(zip(uniq_labels.tolist(), counts.tolist()))}")

    # [2] 全局 NN 距离直方图
    tree = cKDTree(mesh_nodes)
    d, _ = tree.query(mesh_nodes, k=2)
    nn = d[:, 1]
    pct = [0.1, 1, 5, 25, 50, 75, 95, 99, 100]
    q = np.percentile(nn, pct)
    print(f"[2] NN dist percentiles (mm):")
    for p, v in zip(pct, q):
        print(f"     p{p:>5}: {v:.3e}   ({v/eff_vs*100:.3f}% of voxel)")
    print(f"     min NN = {nn.min():.3e}, #(NN < 0.01*voxel) = {(nn < eff_vs*0.01).sum()}")

    # [3] 表面面元质心：有多少深埋在 AABB 内部？
    cent = mesh_nodes[sfaces].mean(axis=1)
    lo, hi = mesh_nodes.min(axis=0), mesh_nodes.max(axis=0)
    dist_to_wall = np.minimum(cent - lo, hi - cent).min(axis=1)
    bbox_half = ((hi - lo) / 2).min()
    deep = dist_to_wall > 0.2 * bbox_half
    print(f"[3] surface-face centroids deep inside AABB "
          f"(>20% from any wall): {deep.sum()}/{F} = {deep.sum()/F*100:.1f}%")

    # [4] 每条边出现在几个表面三角形中？流形应为 2，非流形会出现 1 或 >2
    e01 = np.sort(sfaces[:, [0, 1]], axis=1)
    e12 = np.sort(sfaces[:, [1, 2]], axis=1)
    e02 = np.sort(sfaces[:, [0, 2]], axis=1)
    edges = np.concatenate([e01, e12, e02], axis=0)
    _, ec = np.unique(edges, axis=0, return_counts=True)
    print(f"[4] surface edge-valence histogram: "
          f"=1:{(ec==1).sum()}  =2:{(ec==2).sum()}  >=3:{(ec>=3).sum()}")
    print("==========================\n")

    # surface_faces in mesh_data is already exterior hull (count==1)
    # from MeshGenerator._extract_exterior_faces_fast — no extra filtering needed
    generator.save(mesh_data, str(OUTPUT_DIR / "mesh"))
    logger.info(f"Mesh saved to: {OUTPUT_DIR / "mesh.npz"}")
    mesh_file = OUTPUT_DIR / "mesh.npz"

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
