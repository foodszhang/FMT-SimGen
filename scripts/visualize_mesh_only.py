#!/usr/bin/env python3
"""Generate 3D mesh visualization for the corrected trunk crop mesh."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)

# Load mesh (now in trunk-local, cropped to atlas Y=[30,70] => trunk-local Y=[0,40])
mesh = np.load("output/shared/mesh.npz")
nodes = mesh["nodes"]  # trunk-local mm
sf = mesh["surface_faces"]
elements = mesh["elements"]

print(f"Mesh: {len(nodes)} nodes, {len(elements)} tets, {len(sf)} surface faces")
print(f"Trunk-local Y: [{nodes[:,1].min():.2f}, {nodes[:,1].max():.2f}]")

# ── 3D surface mesh ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Subsample for speed
step = max(1, len(sf) // 5000)
for tri in sf[::step]:
    pts = nodes[tri]
    pts_c = np.vstack([pts, pts[0]])
    ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], "b-", linewidth=0.3, alpha=0.5)

ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=1, c="gray", alpha=0.15)

# MCX volume bbox wireframe (trunk-local)
cx, cy, cz = 19.0, 20.0, 10.4
hx, hy, hz = 19.0, 20.0, 10.4
bbox_corners = np.array([
    [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
    [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
])
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
for i,j in edges:
    ax.plot(*bbox_corners[[i,j]].T, "r-", linewidth=2)

ax.set_xlabel("X (mm) [Left→Right]", fontsize=10)
ax.set_ylabel("Y (mm) [Anterior→Posterior]", fontsize=10)
ax.set_zlabel("Z (mm) [Inferior→Superior]", fontsize=10)
ax.set_title(f"Trunk-only Mesh (unified frame)\n{len(nodes)} nodes | Red box = MCX volume (38×40×20.8mm)",
             fontsize=12)
fig.tight_layout()
fig.savefig(str(out / "mesh_3d_v2.png"), dpi=150)
print(f"[✓] Saved {out / 'mesh_3d_v2.png'}")
plt.close()

# ── 3 orthographic projections ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
proj_pairs = [
    (0, 1, "X (mm)", "Y (mm)"),
    (0, 2, "X (mm)", "Z (mm)"),
    (1, 2, "Y (mm)", "Z (mm)"),
]
for ax, (d1, d2, xl, yl) in zip(axes, proj_pairs):
    ax.scatter(nodes[:, d1], nodes[:, d2], s=2, c="lightblue", alpha=0.6)
    # MCX bbox
    if d1 == 0 and d2 == 1:
        ax.axvline(0, color="red", lw=1); ax.axvline(38, color="red", lw=1)
        ax.axhline(0, color="red", lw=1); ax.axhline(40, color="red", lw=1)
        ax.text(19, 42, "MCX bbox", color="red", fontsize=8, ha="center")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Trunk Mesh Orthographic Projections (crop Y=[30,70]atlas → trunk-local Y=[0,40])", fontsize=12)
fig.tight_layout()
fig.savefig(str(out / "mesh_projections_v2.png"), dpi=150)
print(f"[✓] Saved {out / 'mesh_projections_v2.png'}")
plt.close()

# ── Y histogram ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(nodes[:, 1], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0, color="green", lw=2, label="Y=0 (trunk-local origin)")
ax.axvline(40, color="orange", lw=2, label="Y=40 (MCX bbox top)")
ax.set_xlabel("Y (trunk-local mm)")
ax.set_ylabel("Node count")
ax.set_title(f"Node Y distribution — {len(nodes)} nodes | atlas Y=[30, 70]mm")
ax.legend()
fig.tight_layout()
fig.savefig(str(out / "mesh_y_hist_v2.png"), dpi=150)
print(f"[✓] Saved {out / 'mesh_y_hist_v2.png'}")
plt.close()

print("\nAll saved to output/visualizations/")
