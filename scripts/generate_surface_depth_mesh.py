#!/usr/bin/env python3
"""
Compute surface_depth from mesh geometry using triangle rasterization.
Uses scanline fill to properly cover the full body silhouette per angle.

This ensures every pixel within the body projection gets a finite depth,
not just the pixels where surface nodes happen to project.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.view_config import TurntableCamera


def rasterize_triangles(
    nodes_xyz: np.ndarray,
    triangles: np.ndarray,
    angle_deg: float,
    camera: TurntableCamera,
    fov_mm: float,
    detector_resolution: tuple[int, int],
    volume_center_world: tuple[float, float, float],
) -> np.ndarray:
    """Rasterize surface triangles into a depth map (z-buffer).

    For orthographic projection, each surface triangle projects to a 2D polygon.
    Uses scanline rasterization to fill all pixels inside the projected triangle.

    Returns
    -------
    depth_map [H×W]: camera-frame depth of frontmost triangle hit per pixel.
    """
    H, W = detector_resolution
    D = camera.camera_distance_mm
    c = np.asarray(volume_center_world, dtype=np.float64)

    # Shift + rotate all surface nodes
    θ = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(θ), np.sin(θ)

    p = nodes_xyz - c
    x_rot = p[:, 0] * cos_t + p[:, 2] * sin_t
    y_rot = p[:, 1]
    z_rot = -p[:, 0] * sin_t + p[:, 2] * cos_t

    depth_all = D - z_rot  # [N_all]

    half_fov = fov_mm / 2.0
    scale = W / fov_mm  # px per mm

    # Project all 3 vertices of each surface triangle
    def proj_tri(tri_idx):
        """Project a triangle, return 2D vertices + depth."""
        verts = tri_idx  # (3,) node indices
        pts_2d = np.zeros((3, 2), dtype=np.float64)
        d_vals = np.zeros(3, dtype=np.float64)
        for k, vi in enumerate(verts):
            u = x_rot[vi]
            v = y_rot[vi]
            pts_2d[k, 0] = (u + half_fov) * scale
            pts_2d[k, 1] = (v + half_fov) * scale
            d_vals[k] = depth_all[vi]
        return pts_2d, d_vals

    # Z-buffer: track minimum depth per pixel
    depth_map = np.full((H, W), np.inf, dtype=np.float64)

    for tri in triangles:
        pts_2d, d_vals = proj_tri(tri)

        # Bounding box of projected triangle (in pixels)
        x_min = int(np.clip(np.floor(pts_2d[:, 0].min()), 0, W - 1))
        x_max = int(np.clip(np.ceil(pts_2d[:, 0].max()), 0, W - 1))
        y_min = int(np.clip(np.floor(pts_2d[:, 1].min()), 0, H - 1))
        y_max = int(np.clip(np.ceil(pts_2d[:, 1].max()), 0, H - 1))

        if x_min > x_max or y_min > y_max:
            continue

        # For each pixel in bounding box, point-in-triangle test (using barycentric)
        # Sample center points
        for py in range(y_min, y_max + 1):
            for px in range(x_min, x_max + 1):
                # Pixel center
                pc = np.array([px + 0.5, py + 0.5])

                # Barycentric coordinates
                v0 = pts_2d[1] - pts_2d[0]
                v1 = pts_2d[2] - pts_2d[0]
                v2 = pc - pts_2d[0]

                d00 = v0 @ v0
                d01 = v0 @ v1
                d11 = v1 @ v1
                d20 = v2 @ v0
                d21 = v2 @ v1

                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-10:
                    continue

                w0 = (d11 * d20 - d01 * d21) / denom
                w1 = (d00 * d21 - d01 * d20) / denom
                w2 = 1.0 - w0 - w1

                if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                    # Point is inside triangle (or on edge)
                    # Interpolate depth via barycentric
                    d_interp = w0 * d_vals[0] + w1 * d_vals[1] + w2 * d_vals[2]
                    if d_interp < depth_map[py, px]:
                        depth_map[py, px] = d_interp

    return depth_map


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="data/uniform_trunk_v2_20260420_100948/samples")
    parser.add_argument("--mesh", default="output/shared/mesh.npz")
    parser.add_argument("--view_config", default="output/shared/view_config.json")
    parser.add_argument("--max_faces", type=int, default=0,
                        help="Max faces to rasterize (0=all). Use for debugging.")
    args = parser.parse_args()

    mesh_data = np.load(args.mesh)
    nodes = mesh_data["nodes"]
    sf = mesh_data["surface_faces"]

    view_cfg = json.load(open(args.view_config))
    camera = TurntableCamera(view_cfg)
    angles = view_cfg["angles"]

    if args.max_faces > 0:
        sf = sf[:args.max_faces]
        print(f"DEBUG: rasterizing only first {args.max_faces} faces")

    fov_mm = camera.fov_mm
    det_res = camera.detector_resolution
    vcw = tuple(camera.volume_center_world)

    samples_dir = Path(args.samples_dir)
    for sp in sorted(samples_dir.glob("sample_*")):
        out_path = sp / "surface_depth.npz"
        if out_path.exists():
            continue
        depths = {}
        for angle in angles:
            print(f"  {sp.name} angle={angle}...", end="", flush=True)
            sdepth = rasterize_triangles(
                nodes, sf, angle,
                camera=camera,
                fov_mm=fov_mm,
                detector_resolution=det_res,
                volume_center_world=vcw,
            )
            depths[str(angle)] = sdepth.astype(np.float32)
            print(f" finite={np.isfinite(sdepth).sum()}")
        np.savez_compressed(out_path, **depths)
        print(f"  {sp.name}: saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
