#!/usr/bin/env python3
"""
Plot visible surface nodes per angle — 3D mesh + highlighted visible nodes.
7 angles in a 2×4 combined grid (last panel empty or Union).
Step 0: Run after step0b_generate_mesh.py with tn=5000.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.view_config import TurntableCamera, get_visible_surface_nodes_from_mcx_depth
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD, VOXEL_SIZE_MM
from fmt_simgen.mcx_projection import project_volume_reference

OUTPUT_DIR = Path("output/qa/sample_0000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR = Path("output/shared")
MCX_VOL_PATH = SHARED_DIR / "mcx_volume_trunk.bin"


def load_mesh_and_mask():
    mesh = np.load(SHARED_DIR / "mesh.npz", allow_pickle=True)
    nodes = mesh["nodes"].astype(np.float64)
    exterior_faces = mesh["exterior_surface_faces"]
    mcx_raw = np.fromfile(MCX_VOL_PATH, dtype=np.uint8)
    mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)
    return nodes, exterior_faces, mcx_xyz


def plot_angle(nodes, visible_idx, angle, ax):
    """Plot mesh + visible nodes for one angle."""
    # All exterior nodes
    ax.scatter(
        nodes[:, 0], nodes[:, 1], nodes[:, 2],
        c="lightgray", s=1.0, alpha=0.25, label="All nodes"
    )
    # Visible nodes
    ax.scatter(
        nodes[visible_idx, 0],
        nodes[visible_idx, 1],
        nodes[visible_idx, 2],
        c="lime", s=8.0, alpha=0.9,
        label=f"Visible ({len(visible_idx)})"
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"θ={angle:+d}°, visible={len(visible_idx)}", fontsize=11)
    ax.legend(loc="upper right", fontsize=7)


def main():
    print("Loading mesh + MCX volume...")
    nodes, exterior_faces, mcx_xyz = load_mesh_and_mask()
    mask_xyz = mcx_xyz > 0

    # Camera and depth map
    camera_cfg = dict(
        volume_center_world=VOLUME_CENTER_WORLD,
        camera_distance_mm=200.0,
        fov_mm=80.0,
        detector_resolution=(256, 256),
    )
    camera = TurntableCamera(camera_cfg)

    # Compute depth maps for all angles (reuse)
    depth_maps = {}
    for angle in camera.angles:
        _, depth_maps[angle] = project_volume_reference(
            mask_xyz.astype(np.uint8),
            angle_deg=angle,
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=VOXEL_SIZE_MM,
            volume_center_world=VOLUME_CENTER_WORLD,
        )

    angles = camera.angles
    n_angles = len(angles)

    # ── Figure 1: 2×4 combined grid ───────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    # 2 rows, 4 cols; last cell (2,4) is empty or union
    for idx, angle in enumerate(angles):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
        # Compute visible nodes for this angle
        node_normals = camera.compute_surface_normals(nodes, exterior_faces)
        u_px, v_px, node_depths = camera.project_nodes_to_detector(nodes, angle)
        w_px, h_px = camera.detector_resolution
        in_fov = (u_px >= 0) & (u_px < w_px) & (v_px >= 0) & (v_px < h_px)
        idx_fov = np.where(in_fov)[0]
        u_i = np.clip(np.round(u_px[idx_fov]).astype(np.int32), 0, 255)
        v_i = np.clip(np.round(v_px[idx_fov]).astype(np.int32), 0, 255)
        dm = depth_maps[angle]
        mcx_d = dm[v_i, u_i]
        node_d = node_depths[idx_fov]
        has_cov = np.isfinite(mcx_d)
        not_occluded = node_d <= (mcx_d + 0.5)
        mcx_occluded = np.ones(len(nodes), dtype=bool)
        mcx_occluded[idx_fov] = ~(has_cov & not_occluded)
        is_surface = np.linalg.norm(node_normals, axis=1) > 1e-6
        view_dir = camera.get_view_direction(angle)
        facing = np.dot(node_normals, view_dir) > 0
        visible = is_surface & facing & ~mcx_occluded
        visible_idx = np.where(visible)[0]
        plot_angle(nodes, visible_idx, angle, ax)

    # Last panel: UNION of all visible nodes
    ax = fig.add_subplot(2, 4, 8, projection="3d")
    union = np.zeros(len(nodes), dtype=bool)
    for angle in angles:
        node_normals = camera.compute_surface_normals(nodes, exterior_faces)
        u_px, v_px, node_depths = camera.project_nodes_to_detector(nodes, angle)
        in_fov = (u_px >= 0) & (u_px < w_px) & (v_px >= 0) & (v_px < h_px)
        idx_fov = np.where(in_fov)[0]
        u_i = np.clip(np.round(u_px[idx_fov]).astype(np.int32), 0, 255)
        v_i = np.clip(np.round(v_px[idx_fov]).astype(np.int32), 0, 255)
        dm = depth_maps[angle]
        mcx_d = dm[v_i, u_i]
        node_d = node_depths[idx_fov]
        has_cov = np.isfinite(mcx_d)
        not_occluded = node_d <= (mcx_d + 0.5)
        mcx_occl = np.ones(len(nodes), dtype=bool)
        mcx_occl[idx_fov] = ~(has_cov & not_occluded)
        is_surf = np.linalg.norm(node_normals, axis=1) > 1e-6
        view_dir = camera.get_view_direction(angle)
        facing = np.dot(node_normals, view_dir) > 0
        visible = is_surf & facing & ~mcx_occl
        union |= visible
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="lightgray", s=1.0, alpha=0.25)
    ax.scatter(nodes[union, 0], nodes[union, 1], nodes[union, 2], c="cyan", s=8.0, alpha=0.9)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"UNION all angles\nvisible={union.sum()}", fontsize=11)

    plt.suptitle("Visibility per Angle — tn=5000 mesh + 3-Fix (platform del / vcw unify / exterior hull)", fontsize=12)
    plt.tight_layout()
    grid_path = OUTPUT_DIR / "visibility_all_angles_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {grid_path}")

    # ── Individual figures ─────────────────────────────────────────────────
    for angle in angles:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        node_normals = camera.compute_surface_normals(nodes, exterior_faces)
        u_px, v_px, node_depths = camera.project_nodes_to_detector(nodes, angle)
        in_fov = (u_px >= 0) & (u_px < w_px) & (v_px >= 0) & (v_px < h_px)
        idx_fov = np.where(in_fov)[0]
        u_i = np.clip(np.round(u_px[idx_fov]).astype(np.int32), 0, 255)
        v_i = np.clip(np.round(v_px[idx_fov]).astype(np.int32), 0, 255)
        dm = depth_maps[angle]
        mcx_d = dm[v_i, u_i]
        node_d = node_depths[idx_fov]
        has_cov = np.isfinite(mcx_d)
        not_occluded = node_d <= (mcx_d + 0.5)
        mcx_occl = np.ones(len(nodes), dtype=bool)
        mcx_occl[idx_fov] = ~(has_cov & not_occluded)
        is_surf = np.linalg.norm(node_normals, axis=1) > 1e-6
        view_dir = camera.get_view_direction(angle)
        facing = np.dot(node_normals, view_dir) > 0
        visible = is_surf & facing & ~mcx_occl
        visible_idx = np.where(visible)[0]
        plot_angle(nodes, visible_idx, angle, ax)
        fig_path = OUTPUT_DIR / f"visibility_theta_{angle:+04d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
