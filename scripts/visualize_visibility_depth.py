#!/usr/bin/env python3
"""Visualize FEM surface mesh with measurement_b for sample_XXXX.

- Gray dots = nodes with meas_b == 0 (no signal)
- Colored dots = nodes with meas_b > 0 (signal), color = meas_b intensity
- For each node: show its triangle edge connections
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import json
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Load data
sample_name = "sample_0000"
sample_dir = Path(f"data/uniform_trunk_v2_roi38_multi/samples/{sample_name}")
mesh = np.load("output/shared/mesh.npz", allow_pickle=True)
nodes = mesh["nodes"].astype(np.float64)
surface_faces = mesh["surface_faces"]
surface_idx = mesh["surface_node_indices"]

meas_b = np.load(sample_dir / "measurement_b.npy")

with open(sample_dir / "tumor_params.json") as f:
    tp = json.load(f)
foci = tp["foci"]

print(f"Surface mesh: {len(surface_faces)} triangles, {len(surface_idx)} nodes")
print(f"measurement_b: shape={meas_b.shape}, >0: {np.count_nonzero(meas_b > 0)}/{len(meas_b)}")

focus_centers = [np.array(f["center"]) for f in foci]

# Surface node positions in surface_idx order
surf_nodes = nodes[surface_idx]  # [3900, 3]
orig_to_surf = {ni: i for i, ni in enumerate(surface_idx)}
surf_faces = np.array([[orig_to_surf[fi], orig_to_surf[fs], orig_to_surf[ft]]
                        for fi, fs, ft in surface_faces])

# Classify triangles by signal
tri_has_signal = np.array([
    (meas_b[orig_to_surf[fi]] > 0) or (meas_b[orig_to_surf[fs]] > 0) or (meas_b[orig_to_surf[ft]] > 0)
    for fi, fs, ft in surface_faces
], dtype=bool)
tri_vals = np.array([
    (meas_b[orig_to_surf[fi]] + meas_b[orig_to_surf[fs]] + meas_b[orig_to_surf[ft]]) / 3.0
    for fi, fs, ft in surface_faces
])

# Build line segments for mesh wireframe
def get_face_edges(faces):
    """Get list of (3, 3) line segments for all faces."""
    lines = []
    for fi, fs, ft in faces:
        lines.append([[surf_nodes[fi, 0], surf_nodes[fs, 0], surf_nodes[ft, 0]],
                      [surf_nodes[fi, 1], surf_nodes[fs, 1], surf_nodes[ft, 1]],
                      [surf_nodes[fi, 2], surf_nodes[fs, 2], surf_nodes[ft, 2]]])
    return lines

# ============================================================
# Figure 1: 4 3D views
# ============================================================
fig = plt.figure(figsize=(16, 14))

view_angles = [(20, 30), (20, 120), (20, -60), (20, -150)]
import matplotlib.cm as cm
cmap = cm.inferno
norm = plt.Normalize(0, meas_b.max())

for ax_idx, (elev, azim) in enumerate(view_angles):
    ax = fig.add_subplot(2, 2, ax_idx + 1, projection='3d')

    # --- No-signal triangles: gray wireframe (very faint) ---
    no_sig_faces = surf_faces[~tri_has_signal]
    for fi, fs, ft in no_sig_faces:
        xs = [surf_nodes[fi, 0], surf_nodes[fs, 0], surf_nodes[ft, 0], surf_nodes[fi, 0]]
        ys = [surf_nodes[fi, 1], surf_nodes[fs, 1], surf_nodes[ft, 1], surf_nodes[fi, 1]]
        zs = [surf_nodes[fi, 2], surf_nodes[fs, 2], surf_nodes[ft, 2], surf_nodes[fi, 2]]
        ax.plot(xs, ys, zs, color='lightgray', lw=0.1, alpha=0.12)

    # --- Signal triangles: filled polygons via Poly3DCollection ---
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sig_faces = surf_faces[tri_has_signal]
    sig_vals = tri_vals[tri_has_signal]
    triangles = []
    colors = []
    for i, (fi, fs, ft) in enumerate(sig_faces):
        v = sig_vals[i]
        if v <= 0:
            continue
        c = cmap(min(v / meas_b.max(), 1.0))
        triangles.append([
            [surf_nodes[fi, 0], surf_nodes[fi, 1], surf_nodes[fi, 2]],
            [surf_nodes[fs, 0], surf_nodes[fs, 1], surf_nodes[fs, 2]],
            [surf_nodes[ft, 0], surf_nodes[ft, 1], surf_nodes[ft, 2]],
        ])
        colors.append(c)

    if triangles:
        poly_coll = Poly3DCollection(triangles, facecolors=colors, alpha=0.85, edgecolors='none')
        ax.add_collection3d(poly_coll)

    # Mark foci
    for fc in focus_centers:
        ax.scatter([fc[0]], [fc[1]], [fc[2]],
                   color='cyan', s=200, marker='*', edgecolors='white', linewidth=0.5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'elev={elev} azim={azim} | gray=no-signal, color=signal')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=0.8)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=fig.axes, shrink=0.4, label='measurement_b', orientation='horizontal', pad=0.06)
fig.suptitle(f'{sample_name} | {np.count_nonzero(meas_b>0)}/{len(meas_b)} nodes with signal | {len(foci)} foci', fontsize=12)
plt.tight_layout()
plt.savefig('output/visualization/sample_0000_mesh_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved output/visualization/sample_0000_mesh_3d.png")

# ============================================================
# Figure 2: Orthographic projection (detector view) — 7 angles
# ============================================================
from fmt_simgen.view_config import TurntableCamera
with open("output/shared/view_config.json") as f:
    vc = json.load(f)
camera = TurntableCamera(vc)
angles = sorted(vc.get("angles", [-90, -60, -30, 0, 30, 60, 90]))

fig2, axes2 = plt.subplots(2, len(angles), figsize=(28, 8))

for col, angle in enumerate(angles):
    u_px, v_px, depths = camera.project_nodes_to_detector(surf_nodes, angle)

    # Row 0: all nodes scattered
    ax = axes2[0, col]
    sc = ax.scatter(u_px, v_px, c=meas_b, cmap='inferno', s=5, alpha=0.8, vmin=0, vmax=meas_b.max())
    ax.set_xlabel('u (px)')
    ax.set_ylabel('v (px)')
    ax.set_title(f'{angle}')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Row 1: gray dots = no signal, colored dots = signal
    ax = axes2[1, col]
    zero_mask = meas_b == 0
    nonzero_mask = ~zero_mask
    if zero_mask.sum() > 0:
        ax.scatter(u_px[zero_mask], v_px[zero_mask],
                  c='lightgray', s=3, alpha=0.5, marker='o',
                  label=f'b=0 ({zero_mask.sum()})')
    if nonzero_mask.sum() > 0:
        ax.scatter(u_px[nonzero_mask], v_px[nonzero_mask],
                  c=meas_b[nonzero_mask], cmap='inferno', s=5, alpha=0.8,
                  vmin=0, vmax=meas_b.max(),
                  label=f'b>0 ({nonzero_mask.sum()})')
    ax.set_xlabel('u (px)')
    ax.set_ylabel('v (px)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='upper right')

fig2.text(0.01, 0.75, 'meas_b (all nodes)', fontsize=9, va='center', rotation=90)
fig2.text(0.01, 0.25, 'gray=b=0, color=b>0', fontsize=9, va='center', rotation=90)
fig2.colorbar(sc, ax=axes2, shrink=0.6, label='measurement_b', orientation='horizontal', pad=0.03)
fig2.suptitle(f'{sample_name} | foci={[round(c,1) for c in focus_centers[0]]}', fontsize=11)
plt.tight_layout(rect=[0.03, 0, 1, 0.95])
plt.savefig('output/visualization/sample_0000_mesh_projected.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved output/visualization/sample_0000_mesh_projected.png")

print(f"\nSummary:")
print(f"  meas_b > 0: {np.count_nonzero(meas_b > 0)} / {len(meas_b)}")
print(f"  meas_b == 0: {(meas_b == 0).sum()} / {len(meas_b)}")
print(f"  Foci: {[f['center'] for f in foci]}")
