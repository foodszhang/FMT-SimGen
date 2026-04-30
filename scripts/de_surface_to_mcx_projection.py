#!/usr/bin/env python3
"""
DE Surface → MCX Surface Interpolation → Projection Comparison
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import jdata as jd
from scipy import ndimage
from fmt_simgen.mcx_projection import project_volume_reference, rotation_matrix_y
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD

import matplotlib.pyplot as plt


def resolve_shared_mesh_path() -> Path:
    candidates = [
        Path("output/shared/digimouse_trunk_mesh.npz"),
        Path("output/shared/mesh.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No shared mesh file found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_de_surface_data(sample_dir: Path):
    mesh = np.load(resolve_shared_mesh_path())
    nodes = mesh["nodes"]
    surface_idx = mesh["surface_node_indices"]
    surface_nodes = nodes[surface_idx]
    b = np.load(sample_dir / "measurement_b.npy")
    return surface_nodes, b


def load_mcx_volume(sample_dir: Path) -> np.ndarray:
    jnii_path = list(sample_dir.glob("*.jnii"))[0]
    data = jd.load(str(jnii_path))
    nifti = data["NIFTIData"][:, :, :, 0, 0]
    vol = np.transpose(nifti, (2, 1, 0))
    return vol


def load_label_volume() -> np.ndarray:
    label_path = Path("output/shared/mcx_volume_trunk.bin")
    if not label_path.exists():
        return None
    vol = np.fromfile(label_path, dtype=np.uint8)
    if len(vol) == 104 * 200 * 190:
        vol = vol.reshape((104, 200, 190))
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def identify_box_surface_voxels(shape: tuple) -> np.ndarray:
    nx, ny, nz = shape
    surface = np.zeros(shape, dtype=bool)
    surface[0, :, :] = True
    surface[nx-1, :, :] = True
    surface[:, 0, :] = True
    surface[:, ny-1, :] = True
    surface[:, :, 0] = True
    surface[:, :, nz-1] = True
    return surface


def identify_body_surface_voxels(label_vol: np.ndarray) -> np.ndarray:
    body = label_vol > 0
    eroded = ndimage.binary_erosion(
        body,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    return body & (~eroded)


def interpolate_de_to_mcx_surface(
    de_nodes: np.ndarray,
    de_values: np.ndarray,
    surface_mask: np.ndarray,
    voxel_size: float = 0.2,
    radius: float = 3.0,
    min_neighbors: int = 1,
) -> np.ndarray:
    mcx_vol = np.zeros(surface_mask.shape, dtype=np.float32)
    vol_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    surface_indices = np.array(np.where(surface_mask)).T

    de_voxel_coords = np.zeros_like(de_nodes)
    de_voxel_coords[:, 0] = (de_nodes[:, 0] - vol_origin[0]) / voxel_size
    de_voxel_coords[:, 1] = (de_nodes[:, 1] - vol_origin[1]) / voxel_size
    de_voxel_coords[:, 2] = (de_nodes[:, 2] - vol_origin[2]) / voxel_size

    radius_vox = radius / voxel_size
    radius_sq = radius_vox ** 2

    for (sx, sy, sz) in surface_indices:
        dx = de_voxel_coords[:, 0] - sx
        dy = de_voxel_coords[:, 1] - sy
        dz = de_voxel_coords[:, 2] - sz
        dist_sq = dx*dx + dy*dy + dz*dz
        in_radius = dist_sq <= radius_sq
        n_neighbors = np.sum(in_radius)
        if n_neighbors >= min_neighbors:
            dist = np.sqrt(dist_sq[in_radius])
            weights = 1.0 / (dist ** 2 + 1e-6)
            mcx_vol[sx, sy, sz] = np.sum(weights * de_values[in_radius]) / np.sum(weights)
    return mcx_vol


def project_de_mesh_contour(
    surface_nodes: np.ndarray,
    angle_deg: float,
    camera_distance: float,
    fov_mm: float,
    detector_resolution: tuple,
    voxel_size: float = 0.2,
    volume_center_world: tuple = VOLUME_CENTER_WORLD,
) -> np.ndarray:
    width_pixels, height_pixels = detector_resolution
    width_phys = float(fov_mm)
    height_phys = float(fov_mm)
    pixel_to_phys_x = width_phys / width_pixels
    pixel_to_phys_y = height_phys / height_pixels
    half_w = width_phys / 2
    half_h = height_phys / 2

    pts = surface_nodes.copy().astype(np.float32)
    pts -= np.array(volume_center_world, dtype=np.float32)
    if angle_deg != 0:
        R = rotation_matrix_y(angle_deg)
        pts = pts @ R.T

    cam_x = pts[:, 0]
    cam_y = pts[:, 1]
    cam_z = pts[:, 2]
    depths = camera_distance - cam_z

    contour = np.zeros((height_pixels, width_pixels), dtype=np.float32)

    for i in range(len(cam_x)):
        px, py, dz = cam_x[i], cam_y[i], depths[i]
        if dz <= 0:
            continue
        pix_x = (px + half_w) / pixel_to_phys_x
        pix_y = (py + half_h) / pixel_to_phys_y
        hw = max(1.0, voxel_size / pixel_to_phys_x)
        hh = max(1.0, voxel_size / pixel_to_phys_y)
        x0 = max(0, int(pix_x - hw))
        x1 = min(width_pixels, int(pix_x + hw) + 1)
        y0 = max(0, int(pix_y - hh))
        y1 = min(height_pixels, int(pix_y + hh) + 1)
        contour[y0:y1, x0:x1] = 1.0

    return contour


def project_label_boundary(
    label_vol: np.ndarray,
    angle_deg: float,
    camera_distance: float,
    fov_mm: float,
    detector_resolution: tuple,
    voxel_size: float = 0.2,
    volume_center_world: tuple = VOLUME_CENTER_WORLD,
) -> np.ndarray:
    binary_vol = (label_vol > 0).astype(np.float32)
    proj, _ = project_volume_reference(
        binary_vol, angle_deg, camera_distance, fov_mm,
        detector_resolution, voxel_size, volume_center_world
    )
    return (proj > 0).astype(np.float32)


def make_log_display_same_scale(proj_de: np.ndarray, mcx_vol: np.ndarray) -> tuple:
    """DE and MCX both scaled to MCX's peak value, then log(1+x).

    Args:
        proj_de: DE projection (2D array)
        mcx_vol: Raw MCX fluence volume (for computing scale)
    """
    # Use raw MCX volume max for scaling
    mcx_max = mcx_vol.max()
    de_max = proj_de.max()

    if de_max > 0 and mcx_max > 0:
        # Scale DE so its peak matches MCX's peak
        scale = mcx_max / de_max
        de_scaled = proj_de * scale
    else:
        de_scaled = proj_de

    # Apply log1p to both
    log_de = np.log1p(de_scaled)
    log_mcx = np.log1p(mcx_vol)  # This will be projected later, but here we use raw vol

    return log_de, log_mcx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="sample_0000")
    parser.add_argument("--samples_dir", type=str, default="data/small_uniform_5samples/samples")
    parser.add_argument("--output_dir", type=str, default="output/verification")
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()

    sample_dir = Path(args.samples_dir) / args.sample
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.sample}...")

    # Load data
    surface_nodes, b = load_de_surface_data(sample_dir)
    print(f"DE surface: {len(surface_nodes)} nodes, b range [{b.min():.4f}, {b.max():.4f}]")

    mcx_vol = load_mcx_volume(sample_dir)
    print(f"MCX volume: shape={mcx_vol.shape}, max fluence={mcx_vol.max():.2f}")

    label_vol = load_label_volume()
    if label_vol is not None:
        print(f"Label volume: shape={label_vol.shape}, non-zero={np.count_nonzero(label_vol)}")
        surface_mask = identify_body_surface_voxels(label_vol)
    else:
        print("Label volume not found; fallback to box-boundary surface.")
        surface_mask = identify_box_surface_voxels(mcx_vol.shape)

    # Interpolate DE to MCX surface
    de_interp_vol = interpolate_de_to_mcx_surface(
        surface_nodes, b, surface_mask, voxel_size=0.2, radius=args.radius,
    )
    print(f"DE interpolated: max={de_interp_vol.max():.4f}")

    # Camera
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cam = TurntableCamera(cfg["view_config"])

    import json
    tp = json.load(open(sample_dir / "tumor_params.json"))
    n_foci = tp.get("num_foci", 0)
    depth_tier = tp.get("depth_tier", "unknown")

    angles = [-90, -60, -30, 0, 30, 60, 90]
    n = len(angles)

    fig, axes = plt.subplots(7, n, figsize=(3 * n, 21))

    for col, angle in enumerate(angles):
        # Row 0: DE interpolated projection
        proj_de, _ = project_volume_reference(
            de_interp_vol, float(angle), cam.camera_distance_mm, cam.fov_mm,
            cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
        )
        de_den = max(np.percentile(proj_de, 99.5), 1e-12)
        proj_de_norm = np.clip(proj_de / de_den, 0.0, 1.0)
        ax = axes[0, col]
        ax.imshow(proj_de_norm, cmap="hot", vmin=0, vmax=1.0)
        ax.set_title(f"{angle}°")
        ax.axis("off")

        # Row 1: MCX surface fluence projection (normalized)
        mcx_norm = mcx_vol / max(mcx_vol.max(), 1e-6)
        mcx_surface = mcx_norm * surface_mask.astype(np.float32)
        proj_mcx, _ = project_volume_reference(
            mcx_surface, float(angle), cam.camera_distance_mm, cam.fov_mm,
            cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
        )
        mcx_den = max(np.percentile(proj_mcx, 99.5), 1e-12)
        proj_mcx_norm = np.clip(proj_mcx / mcx_den, 0.0, 1.0)
        ax = axes[1, col]
        ax.imshow(proj_mcx_norm, cmap="hot", vmin=0, vmax=1.0)
        ax.axis("off")

        # Row 2: DE contour (mesh surface nodes projected)
        de_contour = project_de_mesh_contour(
            surface_nodes, float(angle), cam.camera_distance_mm, cam.fov_mm,
            cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
        )
        ax = axes[2, col]
        ax.imshow(proj_de_norm, cmap="hot", vmin=0, vmax=1.0)
        ax.imshow(de_contour, cmap="Oranges", alpha=0.8)
        ax.axis("off")

        # Row 3: MCX body boundary (label volume silhouette)
        if label_vol is not None:
            body_boundary = project_label_boundary(
                label_vol, float(angle), cam.camera_distance_mm, cam.fov_mm,
                cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
            )
        else:
            body_boundary = np.zeros((cam.detector_resolution[1], cam.detector_resolution[0]))
        ax = axes[3, col]
        ax.imshow(proj_mcx_norm, cmap="hot", vmin=0, vmax=1.0)
        ax.imshow(body_boundary, cmap="Oranges", alpha=0.8)
        ax.axis("off")

        # Row 4: DE log (on normalized projection)
        log_de = np.log1p(proj_de_norm)
        ax = axes[4, col]
        ax.imshow(log_de, cmap="hot")
        ax.axis("off")

        # Row 5: MCX log (on normalized surface projection)
        log_mcx = np.log1p(proj_mcx_norm)
        ax = axes[5, col]
        ax.imshow(log_mcx, cmap="hot")
        ax.axis("off")

        # Row 6: Difference (normalized DE - normalized MCX), masked by body silhouette
        ax = axes[6, col]
        diff = (proj_de_norm - proj_mcx_norm) * body_boundary
        v = max(abs(diff.min()), abs(diff.max()), 0.05)
        ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.axis("off")

    for row, label in enumerate([
        "DE interp→MCX surf",
        "MCX fluence (norm)",
        "DE contour (mesh nodes)",
        "MCX body silhouette",
        "DE log (scaled to MCX)",
        "MCX log (raw)",
        "Difference"
    ]):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    fig.suptitle(f"{args.sample}: {n_foci} foci, depth={depth_tier}, source=uniform", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"{args.sample}_de_mcx_surface.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
