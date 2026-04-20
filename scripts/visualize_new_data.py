#!/usr/bin/env python3
"""Visualize new unified-frame data: mesh, samples, projections."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD, TRUNK_SIZE_MM

# ── 1. Mesh 3D viewer (matplotlib) ───────────────────────────────────────────
def mesh_html():
    mesh = np.load("assets/mesh/mesh.npz")
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    sf = mesh["surface_faces"]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)

    # 3D surface mesh (subsampled)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(sf) // 6000)
    for tri in sf[::step]:
        pts = nodes[tri]
        pts_c = np.vstack([pts, pts[0]])
        ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], "b-", linewidth=0.3, alpha=0.6)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=0.5, c="gray", alpha=0.2)

    # MCX bbox wireframe
    cx, cy, cz = VOLUME_CENTER_WORLD
    hx, hy, hz = TRUNK_SIZE_MM / 2.0
    bbox_corners = np.array([
        [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
        [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        ax.plot(*bbox_corners[[i,j]].T, "r-", linewidth=2)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title(f"Mesh (trunk-local) — {len(nodes)} nodes, {len(elements)} tets\nRed=MCX bbox")
    fig.tight_layout()
    fig.savefig(str(out / "mesh_3d.png"), dpi=150)
    print(f"[✓] Saved {out / 'mesh_3d.png'}")
    plt.close()

    # ── Node Y histogram ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nodes[:, 1], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(-25.2, color="r", label="Y min = -25.2")
    ax.axvline(42.0, color="g", label="Y max = 42.0")
    ax.set_xlabel("Y (trunk-local mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Node Y distribution — {len(nodes)} nodes")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(str(out / "mesh_y_hist.png"), dpi=150)
    print(f"[✓] Saved {out / 'mesh_y_hist.png'}")
    plt.close()

# ── 2. Sample scatter + gt_nodes ─────────────────────────────────────────────
def sample_html():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    samples = sorted(Path("data/default/samples").glob("sample_*"))[:8]
    out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, sp in enumerate(samples):
        sample_id = sp.name
        tp = json.load(open(sp / "tumor_params.json"))
        gt_npy = np.load(sp / "gt_nodes.npy")
        b = np.load(sp / "measurement_b.npy")

        ax = axes[idx]
        n_foci = len(tp.get("centers", []))
        n_foci_text = str(n_foci)
        radius = tp.get("foci_radius_mm", "N/A")
        gt_max = float(gt_npy.max())
        b_max = float(b.max())

        ax.set_title(f"{sample_id}", fontsize=8, fontweight="bold")
        ax.axis("off")

        foci_str = "\n".join([
            f"{sample_id}",
            f"n_foci: {n_foci}",
            f"radius: {radius}mm",
            f"gt.max: {gt_max:.4f}",
            f"b.max: {b_max:.3f}",
        ])
        ax.text(0.1, 0.5, foci_str, fontsize=7, transform=ax.transAxes, va="center")

    fig.suptitle("Sample overview (first 8)", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(out / "samples_overview.png"), dpi=150)
    print(f"[✓] Saved {out / 'samples_overview.png'}")
    plt.close()

# ── 3. Mesh vs MCX bbox overlap ─────────────────────────────────────────────
def mesh_mcx_overlap():
    import matplotlib.pyplot as plt
    mesh = np.load("assets/mesh/mesh.npz")
    nodes = mesh["nodes"]
    sf = mesh["surface_faces"]

    mcx_bbox_min = np.array([0, 0, 0])
    mcx_bbox_max = np.array([38, 40, 20.8])

    in_mcx = np.all((nodes >= mcx_bbox_min - 1) & (nodes <= mcx_bbox_max + 1), axis=1)
    ratio = in_mcx.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (dim1, dim2, xl, yl) in zip(axes, [
        (0, 1, "X (mm)", "Y (mm)"),
        (0, 2, "X (mm)", "Z (mm)"),
        (1, 2, "Y (mm)", "Z (mm)"),
    ]):
        ax.scatter(nodes[~in_mcx, dim1], nodes[~in_mcx, dim2], s=0.5, c="gray", alpha=0.3, label="Outside MCX")
        ax.scatter(nodes[in_mcx, dim1], nodes[in_mcx, dim2], s=0.5, c="red", alpha=0.5, label="Inside MCX")
        ax.axvline(mcx_bbox_min[dim1], color="green", lw=1)
        ax.axvline(mcx_bbox_max[dim1], color="green", lw=1)
        ax.axhline(mcx_bbox_min[dim2], color="green", lw=1)
        ax.axhline(mcx_bbox_max[dim2], color="green", lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}\n{ratio*100:.1f}% inside MCX bbox")
        ax.legend(fontsize=7)
    fig.suptitle(f"Mesh nodes vs MCX bbox — {ratio*100:.1f}% inside | Red box=MCX volume", fontsize=12)
    fig.tight_layout()
    out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / "mesh_mcx_overlap.png"), dpi=150)
    print(f"[✓] Saved {out / 'mesh_mcx_overlap.png'}")
    plt.close()

# ── 4. Frame manifest summary ─────────────────────────────────────────────────
def manifest_summary():
    out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)
    m = json.load(open("assets/mesh/frame_manifest.json"))

    lines = ["## Frame Manifest Summary", "---"]
    lines.append(f"`world_frame`: **{m['world_frame']}**")
    lines.append(f"`atlas_to_world_offset_mm`: `{m['atlas_to_world_offset_mm']}`")
    lines.append(f"")
    lines.append(f"**MCX volume**:")
    lines.append(f"  - shape_xyz: `{m['mcx_volume']['shape_xyz']}`")
    lines.append(f"  - voxel_size_mm: `{m['mcx_volume']['voxel_size_mm']}`")
    lines.append(f"  - bbox_world_mm: `{m['mcx_volume']['bbox_world_mm']}`")
    lines.append(f"")
    lines.append(f"**FEM mesh**:")
    lines.append(f"  - frame: `{m['fem_mesh']['frame']}`")
    lines.append(f"  - n_nodes: `{m['fem_mesh']['n_nodes']}`")
    lines.append(f"  - bbox_world_mm: `{m['fem_mesh']['bbox_world_mm']}`")
    lines.append(f"")
    lines.append(f"**gt_voxels**:")
    lines.append(f"  - shape: `{m['voxel_grid_gt']['shape']}`")
    lines.append(f"  - spacing_mm: `{m['voxel_grid_gt']['spacing_mm']}`")
    lines.append(f"  - offset_world_mm: `{m['voxel_grid_gt']['offset_world_mm']}`")
    lines.append(f"  - frame: `{m['voxel_grid_gt']['frame']}`")

    txt = "\n".join(lines)
    open(out / "manifest_summary.txt", "w").write(txt)
    print(f"[✓] Saved {out / 'manifest_summary.txt'}")
    print(txt)

# ── 5. Sample statistics table ───────────────────────────────────────────────
def sample_stats():
    samples = sorted(Path("data/default/samples").glob("sample_*"))
    out = Path("output/visualizations"); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sp in samples:
        tp = json.load(open(sp / "tumor_params.json"))
        b = np.load(sp / "measurement_b.npy")
        gt_vals = np.load(sp / "gt_nodes.npy")
        rows.append({
            "id": sp.name,
            "n_foci": len(tp.get("centers", [])),
            "foci_radius": tp.get("foci_radius_mm", "?"),
            "b_max": float(b.max()) if b.size else 0,
            "gt_max": float(gt_vals.max()) if gt_vals.size else 0,
        })

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    tbl_data = [[r["id"], r["n_foci"], r["foci_radius"], f"{r['b_max']:.3f}", f"{r['gt_max']:.4f}"] for r in rows]
    col_labels = ["Sample", "N foci", "Radius", "b.max", "gt.max"]
    tbl = ax.table(cellText=tbl_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.2)
    fig.suptitle(f"Sample Statistics ({len(rows)} samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out / "sample_stats.png"), dpi=150, bbox_inches="tight")
    print(f"[✓] Saved {out / 'sample_stats.png'}")
    plt.close()

    # Also save as CSV
    import csv
    with open(out / "sample_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","n_foci","foci_radius","b_max","gt_max"])
        w.writeheader(); w.writerows(rows)
    print(f"[✓] Saved {out / 'sample_stats.csv'}")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating visualizations...")
    mesh_html()
    sample_html()
    mesh_mcx_overlap()
    manifest_summary()
    sample_stats()
    print("\nAll done → output/visualizations/")
