#!/usr/bin/env python3
"""
Visualize visible nodes per angle in 3D.

Creates a 2x4 figure:
  - 7 subplots: one per angle showing visible nodes
  - 1 subplot: union of all visible nodes

Output: output/shared/visibility_3d_visualization.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.view_config import TurntableCamera, get_visible_surface_nodes_from_mcx_depth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_mesh(mesh_path: Path) -> dict:
    data = np.load(mesh_path)
    return {
        "nodes": data["nodes"].astype(np.float64),
        "elements": data["elements"],
        "surface_faces": data["surface_faces"],
        "surface_node_indices": data["surface_node_indices"],
    }


def load_mcx_volume(bin_path: Path, shape: tuple = (104, 200, 190)) -> np.ndarray:
    """Load MCX volume from binary file and convert to XYZ order."""
    vol = np.fromfile(bin_path, dtype=np.uint8)
    volume_zyx = vol.reshape(shape)  # (Z, Y, X)
    return volume_zyx.transpose(2, 1, 0)  # -> (X, Y, Z)


def compute_visibility_per_angle(
    camera: TurntableCamera,
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_node_indices: np.ndarray,
    mcx_volume: np.ndarray,
    voxel_size: float = 0.2,
    volume_center_world: tuple = (19.0, 20.0, 10.4),
    epsilon: float = 0.5,
) -> dict:
    exterior_faces_path = Path("output/shared/digimouse_trunk_mesh.npz")
    if exterior_faces_path.exists():
        data = np.load(exterior_faces_path)
        if "exterior_faces" in data:
            exterior_faces = data["exterior_faces"]
            logger.info(f"Using exterior_faces: {exterior_faces.shape}")
        else:
            exterior_faces = surface_faces
    else:
        exterior_faces = surface_faces
    
    visibility_per_angle = {}
    
    for angle in camera.angles:
        logger.info(f"Computing visibility for angle {angle}°...")
        _, visible_mask, _ = get_visible_surface_nodes_from_mcx_depth(
            node_coords=nodes,
            surface_faces=surface_faces,
            mask_xyz=mcx_volume,
            angle_deg=angle,
            voxel_size=voxel_size,
            volume_center_world=volume_center_world,
            epsilon=epsilon,
            exterior_faces=exterior_faces,
        )
        visible_global_idx = np.where(visible_mask)[0]
        local_idx = np.where(np.isin(surface_node_indices, visible_global_idx))[0]
        visibility_per_angle[angle] = local_idx
        logger.info(f"  Angle {angle}°: {len(local_idx)} visible surface nodes")
    
    return visibility_per_angle


def plot_mesh_surface(ax, nodes, faces, alpha=0.1, color="gray"):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    triangles = nodes[faces]
    poly = Poly3DCollection(triangles, alpha=alpha, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)


def plot_visibility_subplot(
    ax,
    surface_nodes: np.ndarray,
    visible_idx: np.ndarray,
    title: str,
    mesh_nodes: np.ndarray,
    mesh_faces: np.ndarray,
    show_mesh: bool = True,
    visible_color: str = "red",
    mesh_alpha: float = 0.05,
):
    if show_mesh:
        plot_mesh_surface(ax, mesh_nodes, mesh_faces, alpha=mesh_alpha, color="lightgray")
    
    if len(visible_idx) > 0:
        visible_coords = surface_nodes[visible_idx]
        ax.scatter(
            visible_coords[:, 0],
            visible_coords[:, 1],
            visible_coords[:, 2],
            c=visible_color,
            s=1,
            alpha=0.8,
            depthshade=True,
        )
    
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    
    ax.tick_params(axis="both", which="major", labelsize=7)
    
    x_range = mesh_nodes[:, 0].max() - mesh_nodes[:, 0].min()
    y_range = mesh_nodes[:, 1].max() - mesh_nodes[:, 1].min()
    z_range = mesh_nodes[:, 2].max() - mesh_nodes[:, 2].min()
    max_range = max(x_range, y_range, z_range) / 2
    
    mid_x = (mesh_nodes[:, 0].max() + mesh_nodes[:, 0].min()) / 2
    mid_y = (mesh_nodes[:, 1].max() + mesh_nodes[:, 1].min()) / 2
    mid_z = (mesh_nodes[:, 2].max() + mesh_nodes[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=20, azim=-60)


def main():
    output_dir = Path("output/shared")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading mesh...")
    mesh_path = output_dir / "mesh.npz"
    mesh = load_mesh(mesh_path)
    nodes = mesh["nodes"]
    surface_faces = mesh["surface_faces"]
    surface_node_indices = mesh["surface_node_indices"]
    surface_nodes = nodes[surface_node_indices]
    
    logger.info(f"Total nodes: {len(nodes)}, surface nodes: {len(surface_node_indices)}")
    
    logger.info("Loading MCX volume...")
    mcx_vol_path = output_dir / "mcx_volume_trunk.bin"
    mcx_volume = load_mcx_volume(mcx_vol_path)
    logger.info(f"MCX volume shape: {mcx_volume.shape}")
    
    logger.info("Loading view config...")
    view_config_path = output_dir / "view_config.json"
    with open(view_config_path) as f:
        view_config = json.load(f)
    
    camera_cfg = {
        "angles": view_config["angles"],
        "pose": view_config["pose"],
        "camera_distance_mm": view_config["camera_distance_mm"],
        "detector_resolution": tuple(view_config["detector_resolution"]),
        "fov_mm": view_config["fov_mm"],
        "volume_center_world": (19.0, 20.0, 10.4),
    }
    camera = TurntableCamera(camera_cfg)
    
    visibility_cache = output_dir / "visibility_per_angle.npz"
    if visibility_cache.exists():
        logger.info(f"Loading cached visibility from {visibility_cache}")
        cache_data = np.load(visibility_cache, allow_pickle=True)
        visibility_per_angle = {int(k): v for k, v in cache_data.items()}
    else:
        logger.info("Computing visibility per angle (with MCX depth occlusion)...")
        visibility_per_angle = compute_visibility_per_angle(
            camera=camera,
            nodes=nodes,
            surface_faces=surface_faces,
            surface_node_indices=surface_node_indices,
            mcx_volume=mcx_volume,
        )
        logger.info(f"Saving visibility cache to {visibility_cache}")
        np.savez(visibility_cache, **{str(k): v for k, v in visibility_per_angle.items()})
    
    stats = view_config["stats_per_angle"]
    
    logger.info("Creating visualization...")
    fig = plt.figure(figsize=(16, 8))
    
    angles = camera.angles
    angle_colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
    angle_color_map = dict(zip(angles, angle_colors))
    
    positions = [
        (2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4),
        (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8),
    ]
    
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(*positions[i], projection="3d")
        visible_idx = visibility_per_angle[angle]
        
        angle_stats = stats[str(angle)]
        title = f"Angle {angle}°: {angle_stats['n_visible']} nodes ({angle_stats['fraction_of_surface']:.1f}%)"
        
        plot_visibility_subplot(
            ax=ax,
            surface_nodes=surface_nodes,
            visible_idx=visible_idx,
            title=title,
            mesh_nodes=nodes,
            mesh_faces=surface_faces,
            show_mesh=True,
            visible_color=[angle_color_map[angle]],
            mesh_alpha=0.03,
        )
    
    ax = fig.add_subplot(*positions[7], projection="3d")
    
    visible_mask_path = output_dir / "visible_mask.npy"
    visible_mask = np.load(visible_mask_path)
    visible_idx_union = np.where(visible_mask)[0]
    
    union_title = f"Union: {len(visible_idx_union)} nodes ({view_config['union_fraction_percent']:.1f}%)"
    
    plot_visibility_subplot(
        ax=ax,
        surface_nodes=surface_nodes,
        visible_idx=visible_idx_union,
        title=union_title,
        mesh_nodes=nodes,
        mesh_faces=surface_faces,
        show_mesh=True,
        visible_color="red",
        mesh_alpha=0.05,
    )
    
    plt.tight_layout()
    
    output_path = output_dir / "visibility_3d_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")
    
    print("\n=== Visibility Summary ===")
    for angle in angles:
        s = stats[str(angle)]
        print(f"  {angle:4d}°: {s['n_visible']:4d} nodes ({s['fraction_of_surface']:.1f}%)")
    print(f"  Union: {len(visible_idx_union)} nodes ({view_config['union_fraction_percent']:.1f}%)")


if __name__ == "__main__":
    main()
