#!/usr/bin/env python3
"""
Overlay: DE surface fluence + MCX fluence + visible nodes per angle.
Row 1: DE gt_nodes projected to detector (linear + log) — using Gaussian tumor model
Row 2: MCX JNII fluence projected (linear + log)
Row 3: MCX proj.npz (ground truth, linear + log) + visible nodes

Mesh: tn=5000 + 3-Fix (回归 sanity check mesh)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD, VOXEL_SIZE_MM
from fmt_simgen.mcx_projection import project_volume_reference, load_jnii_volume

OUTPUT_DIR = Path("output/qa/sample_0000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHARED_DIR = Path("output/shared")
MCX_VOL_PATH = SHARED_DIR / "mcx_volume_trunk.bin"


def load_jnii_fluence(sample_dir: Path):
    jnii_files = list(sample_dir.glob("*.jnii"))
    if not jnii_files:
        raise FileNotFoundError(f"No .jnii in {sample_dir}")
    return load_jnii_volume(jnii_files[0])


def gaussian_fluence_at_nodes(nodes, foci, sigma=3.0):
    """Compute Gaussian tumor fluence at each node (analytical, for visualization).

    This approximates the DE scattered-light excitation at the surface.
    Real DE would require the full system matrix — this is the analytical source term.
    """
    fluence = np.zeros(len(nodes), dtype=np.float32)
    for focus in foci:
        c = np.array(focus["center"], dtype=np.float64)
        rx = float(focus.get("rx", sigma))
        ry = float(focus.get("ry", sigma))
        rz = float(focus.get("rz", sigma))
        r2 = ((nodes[:, 0] - c[0]) / rx) ** 2 + \
             ((nodes[:, 1] - c[1]) / ry) ** 2 + \
             ((nodes[:, 2] - c[2]) / rz) ** 2
        fluence += np.exp(-r2)
    return fluence


def project_nodes_to_image(node_values, node_indices, nodes, camera, angle_deg, res=256, fov=80.0):
    """Rasterize node values onto detector image by nearest-pixel projection.

    Returns (image [H,W], count [H,W]) — weight by fluence value.
    """
    img = np.zeros((res, res), dtype=np.float64)
    cnt = np.zeros((res, res), dtype=np.int32)

    u_px, v_px, _ = camera.project_nodes_to_detector(nodes[node_indices], angle_deg)
    half_fov = fov / 2.0
    u_mm = (u_px / res) * fov - half_fov
    v_mm = (v_px / res) * fov - half_fov

    for k, ni in enumerate(node_indices):
        pu = int((u_mm[k] + half_fov) / fov * res)
        pv = int((v_mm[k] + half_fov) / fov * res)
        if 0 <= pu < res and 0 <= pv < res:
            img[pv, pu] += node_values[ni]
            cnt[pv, pu] += 1

    # Average where multiple nodes fall in same pixel
    img = np.where(cnt > 0, img / cnt, 0.0)
    return img


def load_mesh_and_mask():
    mesh = np.load(SHARED_DIR / "mesh.npz", allow_pickle=True)
    nodes = mesh["nodes"].astype(np.float64)
    exterior_faces = mesh["exterior_surface_faces"]
    mcx_raw = np.fromfile(MCX_VOL_PATH, dtype=np.uint8)
    mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)
    return nodes, exterior_faces, mcx_xyz


def compute_visible_indices(nodes, exterior_faces, camera, depth_map, angle_deg, epsilon=0.5):
    node_normals = camera.compute_surface_normals(nodes, exterior_faces)
    is_surface = np.linalg.norm(node_normals, axis=1) > 1e-6
    u_px, v_px, node_depths = camera.project_nodes_to_detector(nodes, angle_deg)
    w_px, h_px = camera.detector_resolution
    in_fov = (u_px >= 0) & (u_px < w_px) & (v_px >= 0) & (v_px < h_px)
    idx_fov = np.where(in_fov)[0]
    u_i = np.clip(np.round(u_px[idx_fov]).astype(np.int32), 0, w_px - 1)
    v_i = np.clip(np.round(v_px[idx_fov]).astype(np.int32), 0, h_px - 1)
    mcx_d = depth_map[v_i, u_i]
    node_d = node_depths[idx_fov]
    has_cov = np.isfinite(mcx_d)
    not_occ = node_d <= (mcx_d + epsilon)
    mcx_occl = np.ones(len(nodes), dtype=bool)
    mcx_occl[idx_fov] = ~(has_cov & not_occ)
    view_dir = camera.get_view_direction(angle_deg)
    facing = np.dot(node_normals, view_dir) > 0
    visible = is_surface & facing & ~mcx_occl
    return np.where(visible)[0]


def main():
    import json

    print("Loading mesh, MCX volume, JNII fluence, tumor params...")
    nodes, exterior_faces, mcx_xyz = load_mesh_and_mask()
    mask_xyz = mcx_xyz > 0

    sample_dir = Path("data/uniform_trunk_v2_20260420_100948/samples/sample_0000")
    fluence_xyz = load_jnii_fluence(sample_dir)

    tumor_params = json.load(open(sample_dir / "tumor_params.json"))
    foci = tumor_params["foci"]

    proj_data = dict(np.load(sample_dir / "proj.npz"))

    # DE tumor fluence at all exterior nodes
    de_fluence = gaussian_fluence_at_nodes(nodes, foci)  # (N_nodes,)
    # Only exterior surface nodes have meaningful fluence
    ext_node_set = set(np.unique(exterior_faces.ravel()))
    ext_mask = np.array([i in ext_node_set for i in range(len(nodes))])

    print(f"Mesh nodes: {len(nodes)}, exterior surface: {len(ext_node_set)}")
    print(f"DE fluence nonzero (exterior): {(de_fluence[ext_mask] > 1e-6).sum()}")
    print(f"Tumor focus: {foci[0]['center']}")

    camera_cfg = dict(
        volume_center_world=VOLUME_CENTER_WORLD,
        camera_distance_mm=200.0,
        fov_mm=80.0,
        detector_resolution=(256, 256),
    )
    camera = TurntableCamera(camera_cfg)
    angles = camera.angles  # [-90, -60, -30, 0, 30, 60, 90]

    # ── 4×7 combined grid ───────────────────────────────────────────────────
    # Row 1a: DE tumor fluence (linear)
    # Row 1b: DE tumor fluence (log)
    # Row 2a: MCX JNII fluence (linear)
    # Row 2b: MCX JNII fluence (log)
    # Row 3:  MCX proj.npz + visible nodes (log)
    fig, axes = plt.subplots(5, 7, figsize=(35, 25))

    for idx, angle in enumerate(angles):
        angle_str = str(angle)

        # ── MCX depth map (reused) ───────────────────────────────────────────
        _, depth_map = project_volume_reference(
            mask_xyz.astype(np.uint8),
            angle_deg=angle,
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=VOXEL_SIZE_MM,
            volume_center_world=VOLUME_CENTER_WORLD,
        )

        # ── MCX JNII projection ─────────────────────────────────────────────
        proj_jnii, _ = project_volume_reference(
            fluence_xyz.astype(np.float32),
            angle_deg=angle,
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=VOXEL_SIZE_MM,
            volume_center_world=VOLUME_CENTER_WORLD,
        )

        # ── MCX proj.npz ────────────────────────────────────────────────────
        proj_mcx = proj_data[angle_str]
        log_mcx = np.log1p(proj_mcx)

        # ── Visible exterior surface nodes ──────────────────────────────────
        visible_idx = compute_visible_indices(
            nodes, exterior_faces, camera, depth_map, angle, epsilon=0.5
        )

        # DE fluence rasterized at visible exterior nodes
        de_img = project_nodes_to_image(
            de_fluence, visible_idx, nodes, camera, angle, res=256, fov=80.0
        )
        log_de = np.log1p(de_img)

        # Node → detector projection for overlay
        u_px, v_px, _ = camera.project_nodes_to_detector(nodes[visible_idx], angle)
        in_fov = (u_px >= 0) & (u_px < 256) & (v_px >= 0) & (v_px < 256)
        u_mm = (u_px[in_fov] / 256.0) * 80.0 - 40.0
        v_mm = (v_px[in_fov] / 256.0) * 80.0 - 40.0

        # Common vmin/vmax from MCX proj (log)
        vmax_m = np.percentile(log_mcx[proj_mcx > 0], 99) if proj_mcx.max() > 0 else 1
        vmin_m = np.percentile(log_mcx[proj_mcx > 0], 1) if proj_mcx.max() > 0 else 0
        vmax_d = np.percentile(log_de[de_img > 0], 99) if de_img.max() > 0 else 1
        vmin_d = np.percentile(log_de[de_img > 0], 1) if de_img.max() > 0 else 0

        # ── Row 1a: DE tumor fluence (linear) ───────────────────────────────
        ax = axes[0, idx]
        vmax_ld = de_img.max()
        ax.imshow(de_img, cmap="hot", origin="lower",
                  vmin=0, vmax=vmax_ld if vmax_ld > 0 else 1,
                  extent=[-40, 40, -40, 40])
        ax.set_title(f"θ={angle:+d}°\nDE source (linear)", fontsize=8)
        ax.set_xlabel("U (mm)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("V (mm)", fontsize=7)

        # ── Row 1b: DE tumor fluence (log) ─────────────────────────────────
        ax = axes[1, idx]
        ax.imshow(log_de, cmap="hot", origin="lower",
                  vmin=vmin_d, vmax=vmax_d,
                  extent=[-40, 40, -40, 40])
        ax.set_title(f"DE source (log)\nn={len(visible_idx)}", fontsize=8)
        ax.set_xlabel("U (mm)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("V (mm)", fontsize=7)

        # ── Row 2a: MCX JNII fluence (linear) ──────────────────────────────
        ax = axes[2, idx]
        ax.imshow(proj_jnii, cmap="hot", origin="lower",
                  vmin=0, vmax=proj_jnii.max() if proj_jnii.max() > 0 else 1,
                  extent=[-40, 40, -40, 40])
        ax.set_title(f"MCX JNII (linear)", fontsize=8)
        ax.set_xlabel("U (mm)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("V (mm)", fontsize=7)

        # ── Row 2b: MCX JNII fluence (log) ─────────────────────────────────
        ax = axes[3, idx]
        ax.imshow(log_mcx, cmap="hot", origin="lower",
                  vmin=vmin_m, vmax=vmax_m,
                  extent=[-40, 40, -40, 40])
        ax.set_title(f"MCX JNII (log)", fontsize=8)
        ax.set_xlabel("U (mm)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("V (mm)", fontsize=7)

        # ── Row 4: MCX proj.npz + visible nodes (log) ─────────────────────
        ax = axes[4, idx]
        ax.imshow(log_mcx, cmap="hot", origin="lower",
                  vmin=vmin_m, vmax=vmax_m,
                  extent=[-40, 40, -40, 40])
        ax.scatter(u_mm, v_mm, c="lime", s=4.0, alpha=0.9, edgecolors="none")
        ax.set_title(f"MCX proj.npz + visible\n(n={len(visible_idx)}, ε=0.5)", fontsize=8)
        ax.set_xlabel("U (mm)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("V (mm)", fontsize=7)

    plt.suptitle(
        "DE Source vs MCX Fluence vs Visible Nodes (tn=5000 + 3-Fix)\n"
        "Row 1a: DE Gaussian source (linear) | Row 1b: DE source (log)\n"
        "Row 2a: MCX JNII (linear) | Row 2b: MCX JNII (log)\n"
        "Row 3: MCX proj.npz (log) + lime=visible exterior nodes",
        fontsize=10,
    )
    plt.tight_layout()
    grid_path = OUTPUT_DIR / "de_vs_mcx_visibility_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {grid_path}")

    # ── Individual 4-panel figures ─────────────────────────────────────────
    for idx, angle in enumerate(angles):
        angle_str = str(angle)

        _, depth_map = project_volume_reference(
            mask_xyz.astype(np.uint8), angle,
            camera.camera_distance_mm, camera.fov_mm, camera.detector_resolution,
            VOXEL_SIZE_MM, VOLUME_CENTER_WORLD,
        )
        proj_jnii, _ = project_volume_reference(
            fluence_xyz.astype(np.float32), angle,
            camera.camera_distance_mm, camera.fov_mm, camera.detector_resolution,
            VOXEL_SIZE_MM, VOLUME_CENTER_WORLD,
        )
        proj_mcx = proj_data[angle_str]
        visible_idx = compute_visible_indices(
            nodes, exterior_faces, camera, depth_map, angle, epsilon=0.5
        )
        de_img = project_nodes_to_image(
            de_fluence, visible_idx, nodes, camera, angle, res=256, fov=80.0
        )

        u_px, v_px, _ = camera.project_nodes_to_detector(nodes[visible_idx], angle)
        in_fov = (u_px >= 0) & (u_px < 256) & (v_px >= 0) & (v_px < 256)
        u_mm = (u_px[in_fov] / 256.0) * 80.0 - 40.0
        v_mm = (v_px[in_fov] / 256.0) * 80.0 - 40.0

        log_de = np.log1p(de_img)
        log_mcx = np.log1p(proj_mcx)
        vmax_m = np.percentile(log_mcx[proj_mcx > 0], 99) if proj_mcx.max() > 0 else 1
        vmin_m = np.percentile(log_mcx[proj_mcx > 0], 1) if proj_mcx.max() > 0 else 0
        vmax_d = np.percentile(log_de[de_img > 0], 99) if de_img.max() > 0 else 1
        vmin_d = np.percentile(log_de[de_img > 0], 1) if de_img.max() > 0 else 0

        fig, axes_fig = plt.subplots(1, 4, figsize=(24, 5.5))

        titles = [
            f"θ={angle:+d}° | DE Gaussian source (linear)",
            f"DE Gaussian source (log)\nn={len(visible_idx)} visible",
            f"MCX JNII (linear)",
            f"MCX JNII (log)",
        ]
        imgs = [
            de_img,
            log_de,
            proj_jnii,
            log_mcx,
        ]
        vmins = [0, vmin_d, 0, vmin_m]
        vmaxs = [de_img.max() if de_img.max() > 0 else 1, vmax_d,
                 proj_jnii.max() if proj_jnii.max() > 0 else 1, vmax_m]

        for ci, (title, img, vmin, vmax) in enumerate(zip(titles, imgs, vmins, vmaxs)):
            ax = axes_fig[ci]
            ax.imshow(img, cmap="hot", origin="lower", vmin=vmin, vmax=vmax,
                     extent=[-40, 40, -40, 40])
            ax.set_title(title)
            ax.set_xlabel("U (mm)")
            ax.set_ylabel("V (mm)")
            if ci == 0:
                plt.colorbar(ax.images[0], ax=ax, shrink=0.8, label="Intensity")

        plt.suptitle(f"θ={angle:+d}° — DE Source vs MCX (tn=5000 + 3-Fix)", fontsize=11)
        plt.tight_layout()
        fig_path = OUTPUT_DIR / f"de_vs_mcx_theta_{angle:+04d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
