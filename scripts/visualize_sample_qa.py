#!/usr/bin/env python3
"""
Visualize sample quality for the unified-frame FMT-SimGen pipeline.
Produces 5 figures:
  1. Surface visibility (z-buffer, union of all angles)
  2. MCX fluence + depth projections (all angles, all samples)
  3. Tumor placement (3D mesh + foci + MCX bbox)
  4. Tumor on MCX projection overlay (corrected camera geometry)
  5. SUMMARY: unified view of mesh + visibility + MCX projection
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT = Path("output/visualizations"); OUT.mkdir(parents=True, exist_ok=True)
SHARED = Path("output/shared")

# ── Load shared data ───────────────────────────────────────────────────────────
mesh = np.load(SHARED / "mesh.npz")
nodes = mesh["nodes"]
sf = mesh["surface_faces"]

manifest = json.load(open(SHARED / "frame_manifest.json"))
gt_offset = np.array(manifest["voxel_grid_gt"]["offset_world_mm"])
gt_spacing = manifest["voxel_grid_gt"]["spacing_mm"]
gt_shape = manifest["voxel_grid_gt"]["shape"]
gt_bbox_min = gt_offset
gt_bbox_max = gt_offset + np.array(gt_shape) * gt_spacing

mcx_bbox_min = np.array([0.0, 0.0, 0.0])
mcx_bbox_max = np.array([38.0, 40.0, 20.8])

print(f"Mesh: {len(nodes)} nodes, {len(sf)} surface faces")
print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")
print(f"gt_voxels: X=[{gt_bbox_min[0]},{gt_bbox_max[0]}] Y=[{gt_bbox_min[1]},{gt_bbox_max[1]}] Z=[{gt_bbox_min[2]},{gt_bbox_max[2]}]")

# Find newest experiment
all_exps = [(p.stat().st_mtime, p) for p in Path("data").glob("*") if p.is_dir() and (p / "samples").exists()]
_, newest_exp = max(all_exps)
samples = sorted((newest_exp / "samples").glob("sample_*"))
print(f"\nSamples: {newest_exp.name} ({len(samples)} total)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TURNTABLE CAMERA GEOMETRY (matching MCX convention)
# Camera at (D*sinθ, 0, D*cosθ), looks toward origin
# forward = (-sinθ, 0, -cosθ), up = world +Z = (0,0,1)
# right = up × forward, cam_up = forward × right
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.view_config import TurntableCamera

view_cfg = json.load(open(SHARED / "view_config.json"))
camera = TurntableCamera(view_cfg)
angles = view_cfg["angles"]
CAMERA_DIST = 200.0  # mm, matches mcx_projection.py


def get_camera_frame(angle_deg):
    """Get camera position, forward, right, up vectors for an angle.
    Matches MCX convention: camera at D*(sinθ, 0, cosθ), looks toward origin.
    """
    θ = np.deg2rad(angle_deg)
    cam_pos = np.array([CAMERA_DIST * np.sin(θ), 0.0, CAMERA_DIST * np.cos(θ)])
    forward = np.array([-np.sin(θ), 0.0, -np.cos(θ)])  # toward origin
    up = np.array([0.0, 0.0, 1.0])  # world +Z (dorsal)
    right = np.cross(up, forward)
    right /= (np.linalg.norm(right) + 1e-10)
    cam_up = np.cross(forward, right)
    return cam_pos, forward, right, cam_up


def get_ventral_camera_frame(cam_z=-50.0):
    """Camera from BELOW looking UP toward +Z (ventral view).
    Camera at (cx, cy, cam_z) looking toward (cx, cy, 0).
    Image axes: right = +X (world right), up = +Y (world anterior).
    """
    cx, cy = 19.0, 20.0  # approximate center of MCX volume
    cam_pos = np.array([cx, cy, cam_z])
    forward = np.array([0.0, 0.0, 1.0])   # from cam toward origin (+Z)
    right = np.array([1.0, 0.0, 0.0])     # world +X
    cam_up = np.array([0.0, 1.0, 0.0])    # world +Y (anterior on top)
    return cam_pos, forward, right, cam_up


def project_to_detector(focus_pos, angle_deg, fov_mm=80.0, detector=256):
    """Project a 3D point onto the detector at given angle.
    Returns (pixel_x, pixel_y) or None if behind camera or out of FOV.
    """
    cam_pos, forward, right, cam_up = get_camera_frame(angle_deg)
    rel = focus_pos - cam_pos
    depth = np.dot(rel, forward)
    if depth < 0:
        return None, depth  # behind camera
    u = np.dot(rel, right)
    v = np.dot(rel, cam_up)
    half_fov = fov_mm / 2.0
    if abs(u) > half_fov or abs(v) > half_fov:
        return None, depth  # outside FOV
    px = int(np.clip((u + half_fov) / fov_mm * detector, 0, detector - 1))
    py = int(np.clip((v + half_fov) / fov_mm * detector, 0, detector - 1))
    return (px, py), depth


def compute_zbuffer_visibility(nodes_all, surf_coords, surf_normals, angle_deg,
                               fov_mm=80.0, detector=256):
    """Compute z-buffer surface visibility at given angle (no self-occlusion).
    Returns visible_mask for surf_coords.
    """
    cam_pos, forward, right, cam_up = get_camera_frame(angle_deg)
    z_center = camera.z_center
    if camera.pose == "prone":
        platform_occl = surf_coords[:, 2] < z_center
    else:
        platform_occl = surf_coords[:, 2] > z_center

    rel = surf_coords - cam_pos
    depth = np.dot(rel, forward)
    u = np.dot(rel, right)
    v = np.dot(rel, cam_up)

    half_fov = fov_mm / 2.0
    in_fov = (np.abs(u) <= half_fov) & (np.abs(v) <= half_fov)
    valid = (~platform_occl) & in_fov & (depth > 0)

    u_px = ((u + half_fov) / fov_mm * detector).astype(int)
    v_px = ((v + half_fov) / fov_mm * detector).astype(int)
    u_px = np.clip(u_px, 0, detector - 1)
    v_px = np.clip(v_px, 0, detector - 1)

    depth_buf = np.full((detector, detector), -np.inf, dtype=np.float64)
    visible = np.zeros(len(surf_coords), dtype=bool)

    for i in np.where(valid)[0]:
        d = depth[i]
        px, py = u_px[i], v_px[i]
        if d > depth_buf[py, px]:
            depth_buf[py, px] = d
            visible[i] = True
    return visible


# Precompute visibility for all angles using MCX depth_map (self-occlusion)
unique_surf = np.unique(sf)
surf_coords = nodes[unique_surf]
surf_normals = camera.compute_surface_normals(nodes, sf)
visible_per_angle = {}

# Load depth_map from sample_0000 (representative fluence geometry)
sp0 = samples[0]
proj0 = np.load(sp0 / "proj.npz") if (sp0 / "proj.npz").exists() else None

for angle in angles:
    if proj0 is not None and f"depth_{angle}" in proj0:
        depth_map = proj0[f"depth_{angle}"]
        vis = camera.get_visible_surface_nodes_from_mcx_depth(
            nodes, surf_normals, depth_map, angle, depth_tolerance_mm=0.5
        )
        # Map all-node indices → surface-coord indices
        # unique_surf[i_surf] = node_index, so invert: node_index → i_surf
        node_to_surf_idx = np.full(len(nodes), -1, dtype=int)
        node_to_surf_idx[unique_surf] = np.arange(len(unique_surf))
        vis_surf_idx = node_to_surf_idx[vis]
        vis_mask = np.zeros(len(surf_coords), dtype=bool)
        vis_mask[vis_surf_idx[vis_surf_idx >= 0]] = True
        visible_per_angle[angle] = vis_mask
    else:
        # Fallback if no MCX depth_map
        vis_mask = compute_zbuffer_visibility(nodes, surf_coords, surf_normals, angle)
        visible_per_angle[angle] = vis_mask
    print(f"  Angle {angle:4d}°: {visible_per_angle[angle].sum()}/{len(surf_coords)} visible")

# Ventral view: camera from BELOW looking UP (+Z direction)
# This simulates "imaging from abdomen" — shows which dorsal nodes are visible from below
ventral_cam_pos, ventral_forward, ventral_right, ventral_cam_up = get_ventral_camera_frame()
VENTRAL_FOV = 80.0
VENTRAL_DET = 256
z_center = camera.z_center  # platform_z_center ≈ 4.0

rel_v = surf_coords - ventral_cam_pos
depth_v = np.dot(rel_v, ventral_forward)   # positive = in front of camera
u_v = np.dot(rel_v, ventral_right)
v_v = np.dot(rel_v, ventral_cam_up)

half_fov = VENTRAL_FOV / 2.0
in_fov_v = (np.abs(u_v) <= half_fov) & (np.abs(v_v) <= half_fov)

# Platform occlusion: for prone, platform is BELOW mouse (low Z)
# Ventral surface (Z≈0) is ON the platform → occluded
platform_occl_v = surf_coords[:, 2] < z_center

# Front-facing: normals must point toward +Z (upward) to be seen from below
# surf_normals is (n_all_nodes, 3), unique_surf gives surface node indices
front_facing_v = surf_normals[unique_surf, 2] > 0  # +Z normal = dorsal-facing

valid_v = (~platform_occl_v) & in_fov_v & (depth_v > 0) & front_facing_v

# Z-buffer
u_v_px = ((u_v + half_fov) / VENTRAL_FOV * VENTRAL_DET).astype(int)
v_v_px = ((v_v + half_fov) / VENTRAL_FOV * VENTRAL_DET).astype(int)
u_v_px = np.clip(u_v_px, 0, VENTRAL_DET - 1)
v_v_px = np.clip(v_v_px, 0, VENTRAL_DET - 1)

depth_buf_v = np.full((VENTRAL_DET, VENTRAL_DET), -np.inf, dtype=np.float64)
ventral_visible = np.zeros(len(surf_coords), dtype=bool)
for i in np.where(valid_v)[0]:
    d = depth_v[i]
    px, py = u_v_px[i], v_v_px[i]
    if d > depth_buf_v[py, px]:
        depth_buf_v[py, px] = d
        ventral_visible[i] = True

print(f"  Ventral view: {ventral_visible.sum()}/{len(surf_coords)} visible "
      f"(front-facing dorsal surface, platform-occluded ventral)")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: SURFACE VISIBILITY — z-buffer + union
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] Surface visibility...")

# Union of visible nodes across all angles
union_visible = np.zeros(len(surf_coords), dtype=bool)
for v in visible_per_angle.values():
    union_visible |= v
print(f"  Union of all angles: {union_visible.sum()}/{len(surf_coords)} ({100*union_visible.mean():.1f}%)")

fig = plt.figure(figsize=(20, 14))

# 3D views at 4 angles + union
for idx, angle in enumerate(sorted(angles)[:4]):
    ax = fig.add_subplot(4, 7, idx + 1, projection='3d')
    v = visible_per_angle[angle]
    ax.scatter(surf_coords[~v, 0], surf_coords[~v, 1], surf_coords[~v, 2],
               s=0.5, c="gray", alpha=0.2)
    sc = ax.scatter(surf_coords[v, 0], surf_coords[v, 1], surf_coords[v, 2],
                    s=2, c=surf_coords[v, 2], cmap="plasma", alpha=0.6)
    ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D {angle}°\n{v.sum()} visible", fontsize=9)

# Union 3D
ax_union = fig.add_subplot(4, 7, 5, projection='3d')
ax_union.scatter(surf_coords[~union_visible, 0], surf_coords[~union_visible, 1], surf_coords[~union_visible, 2],
                 s=0.5, c="gray", alpha=0.2)
sc = ax_union.scatter(surf_coords[union_visible, 0], surf_coords[union_visible, 1], surf_coords[union_visible, 2],
                      s=2, c=surf_coords[union_visible, 2], cmap="plasma", alpha=0.6)
ax_union.set_xlim(0, 38); ax_union.set_ylim(0, 40); ax_union.set_zlim(0, 21)
ax_union.set_xlabel("X"); ax_union.set_ylabel("Y"); ax_union.set_zlabel("Z")
ax_union.set_title(f"3D UNION\n{union_visible.sum()} visible", fontsize=9)

# XY detector projections for 4 angles + union
for idx, angle in enumerate(sorted(angles)[:4]):
    ax = fig.add_subplot(4, 7, 7 + idx + 1)
    v = visible_per_angle[angle]
    cam_pos, forward, right, cam_up = get_camera_frame(angle)
    rel = surf_coords - cam_pos
    u_all = np.dot(rel, right)
    v_all = np.dot(rel, cam_up)
    depth_all = np.dot(rel, forward)
    half_fov = 40.0
    detector = 256

    valid = (np.abs(u_all) <= half_fov) & (np.abs(v_all) <= half_fov) & (depth_all > 0)
    ax.scatter(u_all[~valid], v_all[~valid], s=0.3, c="gray", alpha=0.1)
    ax.scatter(u_all[valid & ~v], v_all[valid & ~v], s=0.5, c="gray", alpha=0.3)
    sc = ax.scatter(u_all[v], v_all[v], s=2, c=depth_all[v], cmap="plasma", alpha=0.8)
    ax.set_xlim(-half_fov, half_fov); ax.set_ylim(-half_fov, half_fov)
    ax.set_xlabel("U (mm)"); ax.set_ylabel("V (mm)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Detector {angle}°", fontsize=9)

# Union detector
ax = fig.add_subplot(4, 7, 12)
ax.scatter(u_all[~union_visible], v_all[~union_visible], s=0.3, c="gray", alpha=0.1)
ax.scatter(u_all[union_visible], v_all[union_visible], s=2, c=depth_all[union_visible], cmap="plasma", alpha=0.8)
ax.set_xlim(-half_fov, half_fov); ax.set_ylim(-half_fov, half_fov)
ax.set_xlabel("U (mm)"); ax.set_ylabel("V (mm)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_title("Detector UNION", fontsize=9)

# Bar chart
ax_bar = fig.add_subplot(4, 1, 4)
n_vis = [visible_per_angle[a].sum() for a in sorted(angles)]
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(angles)))
ax_bar.bar([str(a) for a in sorted(angles)], n_vis, color=colors, edgecolor="black")
ax_bar.axhline(union_visible.sum(), color="red", lw=2, linestyle="--", label=f"Union={union_visible.sum()}")
ax_bar.set_xlabel("Angle (°)"); ax_bar.set_ylabel("Visible surface nodes")
ax_bar.set_title("Z-buffer visible nodes per angle (red dashed = union across all angles)")
for i, (a, n) in enumerate(zip(sorted(angles), n_vis)):
    ax_bar.text(i, n + 5, str(n), ha="center", fontsize=8)
ax_bar.set_ylim(0, max(max(n_vis), union_visible.sum()) * 1.2)
ax_bar.legend()

fig.suptitle("[1] Surface Visibility (z-buffer orthographic, no self-occlusion)\n"
             "plasma color = depth (mm from camera) | gray = occluded | 3D colored by Z height",
             fontsize=12)
fig.tight_layout()
fig.savefig(str(OUT / "check1_surface_visibility.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check1_surface_visibility.png'}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: MCX FLUENCE + DEPTH — all angles, all samples
# ═══════════════════════════════════════════════════════════════════════════════
print("[2] MCX projections...")

n_samples = min(8, len(samples))
n_cols = n_samples

# Fluence
fig, axes = plt.subplots(7, n_cols, figsize=(5 * n_cols, 35))
for col, sp in enumerate(samples[:n_samples]):
    proj = np.load(sp / "proj.npz") if (sp / "proj.npz").exists() else None
    for row, angle in enumerate(sorted(angles)):
        ax = axes[row, col]
        if proj is not None and str(angle) in proj:
            img = proj[str(angle)]
            vmax = np.percentile(img[img > 0], 99) if img.max() > 0 else 1
            im = ax.imshow(img, cmap="hot", origin="lower", vmin=0, vmax=vmax)
            ax.set_title(f"{sp.name}\n{angle}° max={img.max():.0f}", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=7)
            ax.set_title(f"{sp.name}\n{angle}° — N/A", fontsize=7, color="red")
        ax.axis("off")
        if col == n_samples - 1:
            ax.text(1.02, 0.5, f"{angle}°", transform=ax.transAxes, fontsize=7, va="center")

fig.suptitle(f"[2] MCX Fluence Projections — {n_samples} samples × 7 angles\n"
             f"color=fluence (log scale implicit in hot colormap)", fontsize=12)
fig.tight_layout()
fig.savefig(str(OUT / "check2_mcx_fluence.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check2_mcx_fluence.png'}")
plt.close()

# Depth maps
fig, axes = plt.subplots(7, n_cols, figsize=(5 * n_cols, 35))
for col, sp in enumerate(samples[:n_samples]):
    proj = np.load(sp / "proj.npz") if (sp / "proj.npz").exists() else None
    for row, angle in enumerate(sorted(angles)):
        ax = axes[row, col]
        if proj is not None and f"depth_{angle}" in proj:
            d = proj[f"depth_{angle}"]
            d_cap = np.where(d < CAMERA_DIST, d, CAMERA_DIST)
            im = ax.imshow(d_cap, cmap="coolwarm", origin="lower", vmin=180, vmax=200)
            ax.set_title(f"{sp.name} depth\n{angle}°", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=7)
        ax.axis("off")
        if col == n_samples - 1:
            ax.text(1.02, 0.5, f"{angle}°", transform=ax.transAxes, fontsize=7, va="center")

fig.suptitle(f"[2b] MCX Depth Maps (mm, inf→{CAMERA_DIST} capped) — cool=warm shallow, warm=deep\n"
             f"{n_samples} samples × 7 angles", fontsize=12)
fig.tight_layout()
fig.savefig(str(OUT / "check2b_mcx_depth.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check2b_mcx_depth.png'}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: TUMOR PLACEMENT — 3D mesh + foci + MCX bbox
# ═══════════════════════════════════════════════════════════════════════════════
print("[3] Tumor placement...")

tumor_data = []
for sp in samples:
    tp = json.load(open(sp / "tumor_params.json"))
    foci_list = tp.get("foci", [])
    centers = [f["center"] for f in foci_list]
    radii = [f.get("radius") or max(f.get("rx", 0), f.get("ry", 0), f.get("rz", 0)) for f in foci_list]
    foci_in_mcx = [bool(np.all(np.array(c) - r >= mcx_bbox_min) and np.all(np.array(c) + r <= mcx_bbox_max))
                    for c, r in zip(centers, radii)]
    tumor_data.append({"id": sp.name, "centers": centers, "radii": radii,
                       "all_in_mcx": all(foci_in_mcx), "n_in_mcx": sum(foci_in_mcx), "n_total": len(centers)})

n_plot = min(8, len(tumor_data))
fig = plt.figure(figsize=(20, 5 * ((n_plot + 3) // 4)))
step_sf = max(1, len(sf) // 3000)

cx, cy, cz = 19.0, 20.0, 10.4
hx, hy, hz = 19.0, 20.0, 10.4
bbox_corners = np.array([
    [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
    [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
])
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

for idx, td in enumerate(tumor_data[:n_plot]):
    ax = fig.add_subplot(((n_plot + 3)//4), 4, idx + 1, projection='3d')
    for tri in sf[::step_sf]:
        pts = nodes[tri]; pts_c = np.vstack([pts, pts[0]])
        ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], "b-", linewidth=0.15, alpha=0.3)
    for i, j in edges:
        ax.plot(*bbox_corners[[i, j]].T, "r-", linewidth=1.5)
    for c, r in zip(td["centers"], td["radii"]):
        c = np.array(c)
        ok = bool(np.all(c - r >= mcx_bbox_min) and np.all(c + r <= mcx_bbox_max))
        color = "lime" if ok else "red"
        u, v = np.linspace(0, 2*np.pi, 16), np.linspace(0, np.pi, 10)
        sx = c[0] + r*np.outer(np.cos(u), np.sin(v))
        sy = c[1] + r*np.outer(np.sin(u), np.sin(v))
        sz = c[2] + r*np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(sx, sy, sz, color=color, alpha=0.25, linewidth=0)
        ax.scatter([c[0]], [c[1]], [c[2]], s=60, c=color, marker="o")
    ax.set_xlim(0, 38); ax.set_ylim(0, 40); ax.set_zlim(0, 21)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    status = "✓" if td["all_in_mcx"] else "✗"
    ax.set_title(f"{td['id']} {status} {td['n_in_mcx']}/{td['n_total']} foci in MCX", fontsize=10,
                 color="green" if td["all_in_mcx"] else "red")

fig.suptitle("[3] Tumor Placement — mesh(blue), MCX bbox(red wireframe), foci(sphere, lime=inside, red=outside)\n"
             "Foci extent shown at 1σ radius", fontsize=12)
fig.tight_layout()
fig.savefig(str(OUT / "check3_tumor_placement.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check3_tumor_placement.png'}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: TUMOR ON MCX PROJECTION — corrected camera geometry
# ═══════════════════════════════════════════════════════════════════════════════
print("[4] Tumor on MCX projection overlay...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for row, sp in enumerate(samples[:2]):
    tp = json.load(open(sp / "tumor_params.json"))
    proj = np.load(sp / "proj.npz") if (sp / "proj.npz").exists() else None
    foci_list = tp.get("foci", [])
    foci_centers = [np.array(f["center"]) for f in foci_list]
    foci_radii = [f.get("radius") or max(f.get("rx", 0), f.get("ry", 0), f.get("rz", 0)) for f in foci_list]

    for col, angle in enumerate([-90, 0, 90]):
        ax = axes[row, col]
        if proj is not None and str(angle) in proj:
            img = proj[str(angle)]
            vmax = np.percentile(img[img > 0], 99) if img.max() > 0 else 1
            ax.imshow(img, cmap="hot", origin="lower", vmin=0, vmax=vmax)

            for fi, (fc, fr) in enumerate(zip(foci_centers, foci_radii)):
                pix, depth = project_to_detector(fc, angle)
                if pix is not None:
                    px, py = pix
                    circle = plt.Circle((px, py), radius=12, fill=False,
                                       color="cyan", linewidth=2.5, zorder=5)
                    ax.add_patch(circle)
                    ax.scatter([px], [py], s=100, c="cyan", marker="o", zorder=6)
                    ax.text(px + 14, py + 14, f"F{fi} θ={angle}°\nd={depth:.0f}mm",
                           color="yellow", fontsize=8,
                           bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
            ax.set_title(f"{sp.name} {angle}°\nmax={img.max():.0f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No proj.npz", transform=ax.transAxes, ha="center", color="red")
        ax.axis("off")

fig.suptitle("[4] MCX Fluence + Tumor Focus Overlay (corrected camera geometry)\n"
             "Cyan circle = projected tumor center, label = depth from camera\n"
             "Camera convention: at θ, camera at D*(sinθ,0,cosθ) looking toward origin",
             fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT / "check4_tumor_on_mcx.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check4_tumor_on_mcx.png'}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: SUMMARY — final overview of the complete pipeline result
# ═══════════════════════════════════════════════════════════════════════════════
print("[5] Generating summary figure...")

fig = plt.figure(figsize=(24, 18))

# Row 1: Mesh + visible nodes (union) — 3 projections
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
ax1.scatter(surf_coords[~union_visible, 0], surf_coords[~union_visible, 1], surf_coords[~union_visible, 2],
            s=0.5, c="gray", alpha=0.3, label="Not visible")
ax1.scatter(surf_coords[union_visible, 0], surf_coords[union_visible, 1], surf_coords[union_visible, 2],
            s=1.5, c=surf_coords[union_visible, 2], cmap="plasma", alpha=0.7, label="Visible")
for i, j in edges:
    ax1.plot(*bbox_corners[[i, j]].T, "r-", linewidth=1)
ax1.set_xlim(0, 38); ax1.set_ylim(0, 40); ax1.set_zlim(0, 21)
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_title(f"Mesh + Visible (union)\n{union_visible.sum()}/{len(surf_coords)} surface nodes visible", fontsize=10)

# XY projection
ax2 = fig.add_subplot(3, 4, 2)
ax2.scatter(nodes[:, 0], nodes[:, 1], s=0.5, c="lightblue", alpha=0.3, label="All nodes")
ax2.scatter(surf_coords[~union_visible, 0], surf_coords[~union_visible, 1], s=0.5, c="gray", alpha=0.3)
ax2.scatter(surf_coords[union_visible, 0], surf_coords[union_visible, 1], s=1, c="red", alpha=0.7)
ax2.axvline(0, color="red", lw=0.8); ax2.axvline(38, color="red", lw=0.8)
ax2.axhline(0, color="red", lw=0.8); ax2.axhline(40, color="red", lw=0.8)
ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)")
ax2.set_title("XY Projection\nMCX bbox (red)", fontsize=10)
ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)

# XZ projection
ax3 = fig.add_subplot(3, 4, 3)
ax3.scatter(nodes[:, 0], nodes[:, 2], s=0.5, c="lightblue", alpha=0.3)
ax3.scatter(surf_coords[union_visible, 0], surf_coords[union_visible, 2], s=1, c="red", alpha=0.7)
ax3.axvline(0, color="red", lw=0.8); ax3.axvline(38, color="red", lw=0.8)
ax3.axhline(0, color="red", lw=0.8); ax3.axhline(20.8, color="red", lw=0.8)
ax3.set_xlabel("X (mm)"); ax3.set_ylabel("Z (mm)")
ax3.set_title("XZ Projection", fontsize=10)
ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# Ventral detector view: camera from BELOW looking UP (+Z)
# Shows XY detector plane — dorsal surface (Z~20) is visible, ventral surface (Z~0) is occluded by platform
ax4 = fig.add_subplot(3, 4, 4)
half_fov = VENTRAL_FOV / 2.0
# XY detector: X→pixel_x (right), Y→pixel_y (up)
det_x = (u_v + half_fov) / VENTRAL_FOV * VENTRAL_DET
det_y = (v_v + half_fov) / VENTRAL_FOV * VENTRAL_DET
valid_viz = in_fov_v & (depth_v > 0)
ax4.scatter(det_x[valid_viz & ~ventral_visible],
            det_y[valid_viz & ~ventral_visible],
            s=0.5, c="gray", alpha=0.2, label="Behind (self-occ)")
ax4.scatter(det_x[valid_viz & ventral_visible],
            det_y[valid_viz & ventral_visible],
            s=1.5, c=depth_v[valid_viz & ventral_visible],
            cmap="plasma", alpha=0.8, vmin=0, vmax=70)
ax4.set_xlim(0, VENTRAL_DET); ax4.set_ylim(0, VENTRAL_DET)
ax4.set_xlabel("Detector X (pixel)"); ax4.set_ylabel("Detector Y (pixel)")
ax4.set_title(f"Ventral View (below→up)\n{ventral_visible.sum()}/{len(surf_coords)} nodes visible\n"
              f"↑ dorsal lit, ↓ ventral dark (platform-occluded)", fontsize=9)
ax4.set_aspect("equal"); ax4.grid(True, alpha=0.2)
ax4.invert_yaxis()  # pixel 0 at bottom

# Row 2: MCX fluence (first sample, all 7 angles)
sp0 = samples[0]
proj0 = np.load(sp0 / "proj.npz") if (sp0 / "proj.npz").exists() else None
for col, angle in enumerate(sorted(angles)):
    ax = fig.add_subplot(3, 7, 7 + col + 1)
    if proj0 is not None and str(angle) in proj0:
        img = proj0[str(angle)]
        vmax = np.percentile(img[img > 0], 99) if img.max() > 0 else 1
        ax.imshow(img, cmap="hot", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(f"{angle}°\nmax={img.max():.0f}", fontsize=8)
    else:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
        ax.set_title(f"{angle}°", fontsize=8)
    ax.axis("off")
fig.text(0.5, 0.62, f"Row 2: MCX Fluence Projections — {sp0.name} (all 7 angles)", fontsize=11, ha="center")

# Row 3: Summary table + per-angle stats
ax_table = fig.add_subplot(3, 1, 3)
ax_table.axis("off")

# Stats table
table_data = []
for angle in sorted(angles):
    n = visible_per_angle[angle].sum()
    frac = 100 * n / len(surf_coords)
    table_data.append([f"{angle}°", str(n), f"{frac:.1f}%"])

# Add union row
table_data.append(["UNION", str(union_visible.sum()), f"{100*union_visible.sum()/len(surf_coords):.1f}%"])
# Add ventral view row
table_data.append(["VENTRAL", str(ventral_visible.sum()), f"{100*ventral_visible.sum()/len(surf_coords):.1f}%"])

tbl = ax_table.table(
    cellText=table_data,
    colLabels=["Angle", "Visible Nodes", "% of Surface"],
    loc="center",
    cellLoc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
colors = ["lightblue"]*7 + ["lightgreen", "orange"]  # 7 angles, union, ventral
for i in range(len(table_data)):
    for j in range(3):
        tbl[(i+1, j)].set_facecolor(colors[i] if i < len(colors) else "lightblue")

# Add text summary
summary_text = (
    f"SAMPLES: {len(samples)} ({newest_exp.name})\n"
    f"MESH: {len(nodes)} nodes | SURFACE: {len(surf_coords)} nodes\n"
    f"UNION VISIBLE: {union_visible.sum()}/{len(surf_coords)} ({100*union_visible.sum()/len(surf_coords):.1f}%)\n"
    f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8] mm\n"
    f"gt_voxels: X=[{gt_bbox_min[0]},{gt_bbox_max[0]}] Y=[{gt_bbox_min[1]},{gt_bbox_max[1]}] Z=[{gt_bbox_min[2]},{gt_bbox_max[2]}] mm\n"
)
for i, td in enumerate(tumor_data):
    c = np.array(td["centers"][0]) if td["centers"] else np.zeros(3)
    r = td["radii"][0] if td["radii"] else 0
    summary_text += f"\n{td['id']}: foci={td['n_in_mcx']}/{td['n_total']} in MCX [{'✓' if td['all_in_mcx'] else '✗'}] center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) r={r:.1f}mm"

ax_table.text(0.02, 0.95, summary_text, transform=ax_table.transAxes,
              fontsize=10, va="top", family="monospace",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle(
    "[5] SUMMARY — Complete FMT-SimGen Pipeline Result\n"
    "Row 1: Mesh + visible nodes (union, plasma=visible, gray=occluded, red=MCX bbox)\n"
    "Row 2: MCX fluence projections (all 7 angles for first sample)\n"
    "Row 3: Visibility stats table + sample summary",
    fontsize=12
)
fig.tight_layout()
fig.savefig(str(OUT / "check5_summary.png"), dpi=150)
print(f"[✓] Saved {OUT / 'check5_summary.png'}")
plt.close()


# ── Print terminal summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"Experiment: {newest_exp.name}")
print(f"Mesh: {len(nodes)} nodes, {len(sf)} surface faces")
print(f"MCX bbox: X=[0,38] Y=[0,40] Z=[0,20.8]")
print(f"gt_voxels: X=[{gt_bbox_min[0]:.1f},{gt_bbox_max[0]:.1f}] Y=[{gt_bbox_min[1]:.1f},{gt_bbox_max[1]:.1f}] Z=[{gt_bbox_min[2]:.1f},{gt_bbox_max[2]:.1f}]")
print()
print("Z-buffer surface visibility (no self-occlusion):")
for angle in sorted(angles):
    n = visible_per_angle[angle].sum()
    print(f"  {angle:4d}°: {n}/{len(surf_coords)} ({100*n/len(surf_coords):.1f}%)")
print(f"  UNION: {union_visible.sum()}/{len(surf_coords)} ({100*union_visible.sum()/len(surf_coords):.1f}%)")
print()
print("Tumor placement:")
for td in tumor_data:
    status = "✓" if td["all_in_mcx"] else "✗"
    if td["centers"]:
        c = np.array(td["centers"][0])
        r = td["radii"][0]
        print(f"  {td['id']}: {td['n_in_mcx']}/{td['n_total']} foci in MCX [{status}] center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) r={r:.1f}mm")
    else:
        print(f"  {td['id']}: no foci")
print()
print("Output files:")
for f in sorted(OUT.glob("check*.png")):
    print(f"  {f.name}")
