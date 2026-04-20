#!/usr/bin/env python3
"""U4.5: Generate diagnostic figures for mesh outer hull."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.frame_contract import TRUNK_SIZE_MM


def load_mesh():
    m = np.load("output/shared/mesh.npz")
    return m["nodes"], m["elements"], m["surface_faces"], m["tissue_labels"]


def classify_faces(nodes, elements, faces, tissue_labels):
    """Classify each face as exterior or interior."""
    node_to_tets = defaultdict(set)
    for t_idx in range(len(elements)):
        for n in elements[t_idx]:
            node_to_tets[n].add(t_idx)

    exterior_mask = np.zeros(len(faces), dtype=bool)
    for f_idx, f in enumerate(faces):
        s = set(f)
        candidates = node_to_tets[f[0]] & node_to_tets[f[1]] & node_to_tets[f[2]]
        adjacent = [t for t in candidates if len(s & set(elements[t])) >= 3]
        if len(adjacent) <= 1:
            exterior_mask[f_idx] = True
    return exterior_mask


def plot_axial_slice(nodes, elements, faces, tissue_labels, z_mm=10.0):
    """Figure A: Axial slice at z=z_mm showing tet cross-section."""
    # Element centers
    elem_centers = nodes[elements[:, :4]].mean(axis=1)

    # Filter elements with center near z=z_mm (within 1mm)
    z_mask = np.abs(elem_centers[:, 2] - z_mm) < 1.0
    z_elems = elements[z_mask]
    z_labels = tissue_labels[z_mask]
    z_centers = elem_centers[z_mask]

    print(f"Slice z={z_mm}: {z_mask.sum()} elements")

    # Color map for tissue labels
    label_colors = {
        1: "lightgray", 2: "red", 3: "orange", 4: "yellow", 5: "gold",
        6: "green", 7: "cyan", 8: "blue", 9: "purple", 10: "magenta",
        11: "pink", 12: "brown",
    }
    label_names = {
        1: "soft tissue", 2: "bone", 3: "muscle", 4: "fat",
        5: "skin", 6: "lung", 7: "heart", 8: "liver",
        9: "kidney", 10: "spleen", 11: "stomach", 12: "intestine",
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot element centers as colored dots
    for label in sorted(np.unique(z_labels)):
        mask = z_labels == label
        ax.scatter(
            z_centers[mask, 0], z_centers[mask, 1],
            s=50, c=label_colors.get(int(label), "gray"),
            label=f"label {int(label)}", alpha=0.8,
        )

    ax.set_xlim(0, 38)
    ax.set_ylim(0, 40)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Axial Slice at Z={z_mm:.1f} mm — Tet Centers Colored by Tissue Label\n"
                 f"({mask.sum()} elements in slice)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Draw body outline
    circle = plt.Circle((19, 20), 17, fill=False, color="black", linewidth=2, linestyle="--")
    ax.add_patch(circle)

    fig.tight_layout()
    out = Path("output/visualizations/mesh_slice_z10.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_outer_hull(nodes, faces, exterior_mask):
    """Figure B: Outer hull wireframe only."""
    outer_faces = faces[exterior_mask]

    print(f"Outer hull faces: {len(outer_faces)} / {len(faces)} = "
          f"{exterior_mask.sum()/len(faces)*100:.1f}%")

    # Surface area
    def tri_area(pts):
        a,b,c = pts
        return 0.5 * np.linalg.norm(np.cross(b-a, c-a))
    outer_area = sum(tri_area(nodes[f]) for f in outer_faces)
    expected = 2 * (TRUNK_SIZE_MM[0]*TRUNK_SIZE_MM[1] +
                     TRUNK_SIZE_MM[0]*TRUNK_SIZE_MM[2] +
                     TRUNK_SIZE_MM[1]*TRUNK_SIZE_MM[2])
    print(f"Outer hull area: {outer_area:.1f} mm² (expected {expected:.1f} mm²)")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for performance
    step = max(1, len(outer_faces) // 8000)
    for tri in outer_faces[::step]:
        pts = nodes[tri]
        pts_c = np.vstack([pts, pts[0]])
        ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], "b-", linewidth=0.3, alpha=0.6)

    # MCX bbox wireframe
    cx, cy, cz = TRUNK_SIZE_MM / 2
    hx, hy, hz = TRUNK_SIZE_MM / 2
    bbox_corners = np.array([
        [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
        [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        ax.plot(*bbox_corners[[i,j]].T, "r-", linewidth=1.5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Outer Hull Wireframe (n={len(outer_faces)} faces)\n"
                 f"Area={outer_area:.0f} mm² (expected {expected:.0f} mm²)",
                 fontsize=11)
    fig.tight_layout()
    out = Path("output/visualizations/mesh_outer_hull.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def main():
    print("=== U4.5 Diagnostic Figures ===")
    nodes, elements, faces, tissue_labels = load_mesh()

    exterior_mask = classify_faces(nodes, elements, faces, tissue_labels)

    print(f"\n--- Figure A: Axial slice z=10mm ---")
    plot_axial_slice(nodes, elements, faces, tissue_labels, z_mm=10.0)

    print(f"\n--- Figure B: Outer hull wireframe ---")
    plot_outer_hull(nodes, faces, exterior_mask)

    print("\nDone.")


if __name__ == "__main__":
    main()
