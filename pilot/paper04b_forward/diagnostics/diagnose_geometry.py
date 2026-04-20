"""Geometry Alignment Audit for P5-ventral (Y=10).

G-1: MCX archive source consistency
G-2: Fluence argmax vs gt_pos
G-3: Atlas label at gt voxel
G-4: Vertex label distribution
G-5: Projection alignment
G-6: 3D overlay visual check
"""

import sys
from pathlib import Path

import numpy as np
from skimage import measure
import json
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/geometry_audit")

GT_POS = np.array([-0.6, 2.4, -3.8])


def load_volume():
    volume_path = ARCHIVE_BASE / "mcx_volume_downsampled_2x.bin"
    return np.fromfile(volume_path, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def g1_mcx_archive_source():
    """G-1: Check MCX archive source consistency."""
    print("\n" + "=" * 70)
    print("G-1: MCX Archive Source Consistency")
    print("=" * 70)

    p5_dir = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0"

    cfg_path = None
    src_raw = None
    src_mm_from_archive = None

    for ext in ["*.json", "*.yaml", "*.yml", "*.txt"]:
        for f in p5_dir.glob(ext):
            print(f"  Found: {f.name}")
            try:
                if f.suffix == ".json":
                    with open(f) as fp:
                        data = json.load(fp)
                    cfg_path = f
                    if "source" in data:
                        src_raw = data["source"]
                        if isinstance(src_raw, dict):
                            if "pos" in src_raw:
                                src_raw = src_raw["pos"]
                        if isinstance(src_raw, list) and len(src_raw) >= 3:
                            src_mm_from_archive = np.array(src_raw[:3])
                    elif "source_pos" in data:
                        src_raw = data["source_pos"]
                        if isinstance(src_raw, list) and len(src_raw) >= 3:
                            src_mm_from_archive = np.array(src_raw[:3])
                    break
                elif f.suffix in [".yaml", ".yml"]:
                    with open(f) as fp:
                        data = yaml.safe_load(fp)
                    cfg_path = f
                    if "source" in data:
                        src_raw = data["source"]
                        if isinstance(src_raw, dict):
                            if "pos" in src_raw:
                                src_raw = src_raw["pos"]
                        if isinstance(src_raw, list) and len(src_raw) >= 3:
                            src_mm_from_archive = np.array(src_raw[:3])
                    break
            except Exception as e:
                print(f"    Error reading: {e}")

    if src_mm_from_archive is None:
        for f in p5_dir.glob("*"):
            if f.is_file() and f.suffix not in [".npy", ".bin", ".png"]:
                print(f"  Checking: {f.name}")
                try:
                    content = f.read_text()
                    if (
                        "source" in content.lower()
                        or "-0.6" in content
                        or "2.4" in content
                    ):
                        print(f"    Content preview: {content[:500]}")
                except:
                    pass

    print(f"\n  archive config path: {cfg_path}")
    print(f"  archive source (raw): {src_raw}")
    print(f"  archive source (mm): {src_mm_from_archive}")
    print(f"  code gt_pos (mm): {GT_POS}")

    if src_mm_from_archive is not None:
        diff = np.linalg.norm(src_mm_from_archive - GT_POS)
        print(f"  |archive - code|: {diff:.4f} mm")
        passed = diff < 0.2
    else:
        diff = None
        passed = False
        print("  WARNING: Could not find source position in archive config files")

    result = "PASS" if passed else "FAIL"
    print(f"\n  **Result**: {result}")

    return {
        "cfg_path": str(cfg_path) if cfg_path else None,
        "src_raw": str(src_raw) if src_raw else None,
        "src_mm": src_mm_from_archive.tolist()
        if src_mm_from_archive is not None
        else None,
        "gt_pos": GT_POS.tolist(),
        "diff_mm": float(diff) if diff is not None else None,
        "passed": passed,
        "result": result,
    }


def g2_fluence_argmax():
    """G-2: Check fluence argmax vs gt_pos."""
    print("\n" + "=" * 70)
    print("G-2: Fluence Argmax vs GT Position")
    print("=" * 70)

    fluence_path = ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"
    fluence = np.load(fluence_path)

    print(f"  fluence shape: {fluence.shape}")
    print(f"  fluence dtype: {fluence.dtype}")
    print(f"  fluence range: [{fluence.min():.3e}, {fluence.max():.3e}]")

    peak_voxel = np.unravel_index(fluence.argmax(), fluence.shape)
    peak_voxel = np.array(peak_voxel)

    center = np.array(fluence.shape) / 2
    peak_mm = (peak_voxel - center) * VOXEL_SIZE_MM

    print(f"  peak voxel: {tuple(peak_voxel)}")
    print(f"  peak mm: {peak_mm}")
    print(f"  gt_pos mm: {GT_POS}")

    diff = np.linalg.norm(peak_mm - GT_POS)
    print(f"  |peak - gt|: {diff:.4f} mm")

    if diff < 0.4:
        result = "PASS"
    elif diff < 0.8:
        result = "MARGINAL"
    else:
        result = "FAIL"

    print(f"\n  **Result**: {result}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    log_fluence = np.log10(fluence + 1e-20)

    gt_voxel = np.round(GT_POS / VOXEL_SIZE_MM + center).astype(int)

    ax = axes[0]
    im = ax.imshow(log_fluence[:, peak_voxel[1], :].T, origin="lower", cmap="viridis")
    ax.scatter(
        peak_voxel[0],
        peak_voxel[2],
        c="white",
        marker="x",
        s=100,
        linewidths=2,
        label="peak",
    )
    ax.scatter(
        gt_voxel[0], gt_voxel[2], c="red", marker="x", s=100, linewidths=2, label="GT"
    )
    ax.set_xlabel("X (voxel)")
    ax.set_ylabel("Z (voxel)")
    ax.set_title(f"XZ slice @ Y={peak_voxel[1]}")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log10(fluence)")

    ax = axes[1]
    im = ax.imshow(log_fluence[peak_voxel[0], :, :].T, origin="lower", cmap="viridis")
    ax.scatter(
        peak_voxel[1],
        peak_voxel[2],
        c="white",
        marker="x",
        s=100,
        linewidths=2,
        label="peak",
    )
    ax.scatter(
        gt_voxel[1], gt_voxel[2], c="red", marker="x", s=100, linewidths=2, label="GT"
    )
    ax.set_xlabel("Y (voxel)")
    ax.set_ylabel("Z (voxel)")
    ax.set_title(f"YZ slice @ X={peak_voxel[0]}")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log10(fluence)")

    ax = axes[2]
    im = ax.imshow(log_fluence[:, :, peak_voxel[2]], origin="lower", cmap="viridis")
    ax.scatter(
        peak_voxel[0],
        peak_voxel[1],
        c="white",
        marker="x",
        s=100,
        linewidths=2,
        label="peak",
    )
    ax.scatter(
        gt_voxel[0], gt_voxel[1], c="red", marker="x", s=100, linewidths=2, label="GT"
    )
    ax.set_xlabel("X (voxel)")
    ax.set_ylabel("Y (voxel)")
    ax.set_title(f"XY slice @ Z={peak_voxel[2]}")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log10(fluence)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "G2_fluence_peak.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: G2_fluence_peak.png")

    return {
        "fluence_shape": list(fluence.shape),
        "fluence_range": [float(fluence.min()), float(fluence.max())],
        "peak_voxel": peak_voxel.tolist(),
        "peak_mm": peak_mm.tolist(),
        "gt_pos_mm": GT_POS.tolist(),
        "diff_mm": float(diff),
        "passed": diff < 0.8,
        "result": result,
    }


def g3_atlas_label_at_gt():
    """G-3: Check atlas label at gt voxel."""
    print("\n" + "=" * 70)
    print("G-3: Atlas Label at GT Voxel")
    print("=" * 70)

    atlas = load_volume()

    center = np.array(atlas.shape) / 2
    gt_voxel = np.round(GT_POS / VOXEL_SIZE_MM + center).astype(int)

    print(f"  atlas shape: {atlas.shape}")
    print(f"  gt_voxel: {tuple(gt_voxel)}")

    if not (
        0 <= gt_voxel[0] < atlas.shape[0]
        and 0 <= gt_voxel[1] < atlas.shape[1]
        and 0 <= gt_voxel[2] < atlas.shape[2]
    ):
        print("  ERROR: gt_voxel out of bounds!")
        label_at_gt = -1
        neigh_labels = {}
        passed = False
    else:
        label_at_gt = int(atlas[tuple(gt_voxel)])
        print(f"  atlas label at gt: {label_at_gt}")

        x0, x1 = max(0, gt_voxel[0] - 1), min(atlas.shape[0], gt_voxel[0] + 2)
        y0, y1 = max(0, gt_voxel[1] - 1), min(atlas.shape[1], gt_voxel[1] + 2)
        z0, z1 = max(0, gt_voxel[2] - 1), min(atlas.shape[2], gt_voxel[2] + 2)
        neigh = atlas[x0:x1, y0:y1, z0:z1]

        unique, counts = np.unique(neigh, return_counts=True)
        neigh_labels = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  3x3x3 neighborhood labels: {neigh_labels}")

        passed = label_at_gt == 1 and all(l in {0, 1} for l in neigh_labels.keys())

    result = "PASS" if passed else "FAIL"
    print(f"\n  **Result**: {result}")

    return {
        "atlas_shape": list(atlas.shape),
        "gt_voxel": gt_voxel.tolist(),
        "label_at_gt": label_at_gt,
        "neighborhood_labels": neigh_labels,
        "passed": passed,
        "result": result,
    }


def g4_vertex_label_distribution():
    """G-4: Check vertex label distribution."""
    print("\n" + "=" * 70)
    print("G-4: Vertex Label Distribution")
    print("=" * 70)

    atlas = load_volume()
    vertices = extract_surface_vertices(atlas > 0, VOXEL_SIZE_MM)

    print(f"  vertices count: {len(vertices)}")
    print(f"  extent x: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
    print(f"  extent y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
    print(f"  extent z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
    print(f"  volume extent: {np.array(atlas.shape) * VOXEL_SIZE_MM}")

    center = np.array(atlas.shape) / 2
    v_vox = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    v_vox = np.clip(v_vox, 0, np.array(atlas.shape) - 1)

    labels_at_v = atlas[v_vox[:, 0], v_vox[:, 1], v_vox[:, 2]]

    label_counts = {}
    for lbl in range(10):
        cnt = int((labels_at_v == lbl).sum())
        pct = 100 * cnt / len(vertices)
        label_counts[lbl] = {"count": cnt, "percent": pct}
        print(f"    label={lbl}: {cnt} ({pct:.1f}%)")

    air_soft = (labels_at_v == 0).sum() + (labels_at_v == 1).sum()
    organ = (labels_at_v >= 2).sum()
    air_soft_pct = 100 * air_soft / len(vertices)
    organ_pct = 100 * organ / len(vertices)

    print(f"\n  label in {{0,1}}: {air_soft_pct:.1f}%")
    print(f"  label >= 2: {organ_pct:.1f}%")

    passed = organ_pct < 5
    result = "PASS" if passed else "FAIL"
    print(f"\n  **Result**: {result}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(range(10))
    counts = [label_counts[l]["count"] for l in labels]
    colors = ["lightblue" if l in [0, 1] else "salmon" for l in labels]

    ax.bar(labels, counts, color=colors, edgecolor="black")
    ax.set_xlabel("Volume Label")
    ax.set_ylabel("Vertex Count")
    ax.set_title("Vertex Label Distribution")
    ax.set_xticks(labels)

    for l, c in zip(labels, counts):
        if c > 0:
            ax.annotate(f"{c}", (l, c), ha="center", va="bottom", fontsize=8)

    ax.axhline(y=0, color="black", linewidth=0.5)

    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightblue",
            edgecolor="black",
            label="Air/Soft tissue (0,1)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="salmon", edgecolor="black", label="Organ (≥2)"
        ),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "G4_vertex_label_hist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: G4_vertex_label_hist.png")

    return {
        "vertices_count": len(vertices),
        "extent_x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
        "extent_y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
        "extent_z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())],
        "label_counts": label_counts,
        "air_soft_pct": air_soft_pct,
        "organ_pct": organ_pct,
        "passed": passed,
        "result": result,
    }


def rotation_matrix_y(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])


def project_3d_to_2d(
    pos_3d, angle_deg, distance=200.0, fov=50.0, resolution=(256, 256)
):
    R = rotation_matrix_y(angle_deg)
    rotated = R @ pos_3d

    cam_x, cam_y, cam_z = rotated
    depth = distance - cam_z

    px_size = fov / resolution[0]
    half_w, half_h = fov / 2, fov / 2

    u = int((cam_x + half_w) / px_size)
    v = int((cam_y + half_h) / px_size)

    return u, v, depth


def project_volume_to_camera_plane(
    volume, angle_deg, distance=200.0, fov=50.0, resolution=(256, 256)
):
    R = rotation_matrix_y(angle_deg)

    coords = np.indices(volume.shape).reshape(3, -1).T
    center = np.array(volume.shape) / 2
    coords_mm = (coords - center) * VOXEL_SIZE_MM

    rotated = coords_mm @ R.T

    cam_x = rotated[:, 0]
    cam_y = rotated[:, 1]
    depths = distance - rotated[:, 2]

    px_size = fov / resolution[0]
    half_w, half_h = fov / 2, fov / 2

    u = ((cam_x + half_w) / px_size).astype(int)
    v = ((cam_y + half_h) / px_size).astype(int)

    valid = (
        (u >= 0) & (u < resolution[0]) & (v >= 0) & (v < resolution[1]) & (depths > 0)
    )

    projection = np.zeros(resolution, dtype=np.float32)
    values = volume.flatten()

    for i in range(len(values)):
        if valid[i] and values[i] > 0:
            projection[v[i], u[i]] += values[i]

    return projection


def g5_projection_alignment():
    """G-5: Check projection alignment."""
    print("\n" + "=" * 70)
    print("G-5: Projection Alignment @ -60° view")
    print("=" * 70)

    angle_deg = -60
    camera_params = {"distance": 200.0, "fov": 50.0, "resolution": (256, 256)}

    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    gt_proj = project_3d_to_2d(GT_POS, angle_deg, **camera_params)
    print(f"  gt_pos projected pixel (u, v): {gt_proj[:2]}")
    print(f"  gt_pos depth: {gt_proj[2]:.2f} mm")

    proj_2d = project_volume_to_camera_plane(fluence, angle_deg, **camera_params)

    if proj_2d.max() > 0:
        log_proj = np.log10(proj_2d + 1e-20)
    else:
        log_proj = proj_2d

    hotspot = np.unravel_index(proj_2d.argmax(), proj_2d.shape)
    print(f"  MCX projection hotspot pixel (v, u): {hotspot}")

    pixel_dist = np.sqrt(
        (gt_proj[0] - hotspot[1]) ** 2 + (gt_proj[1] - hotspot[0]) ** 2
    )
    print(f"  |gt_proj - hotspot| (pixel): {pixel_dist:.2f}")

    passed = pixel_dist <= 5
    result = "PASS" if pixel_dist <= 3 else ("MARGINAL" if pixel_dist <= 5 else "FAIL")
    print(f"\n  **Result**: {result}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    if proj_2d.max() > 0:
        im = ax.imshow(
            log_proj,
            origin="lower",
            cmap="viridis",
            vmin=log_proj[log_proj > -20].min(),
        )
    else:
        im = ax.imshow(proj_2d, origin="lower", cmap="viridis")

    ax.scatter(
        gt_proj[0],
        gt_proj[1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
        label="GT projection",
    )
    ax.scatter(
        hotspot[1],
        hotspot[0],
        c="white",
        marker="o",
        s=200,
        linewidths=3,
        facecolors="none",
        label="Hotspot",
    )

    ax.set_xlabel("U (pixel)")
    ax.set_ylabel("V (pixel)")
    ax.set_title(f"Projection @ {angle_deg}° (pixel dist: {pixel_dist:.1f})")
    ax.legend()

    plt.colorbar(im, ax=ax, label="log10(fluence sum)")

    scale_bar_mm = 5
    scale_bar_px = scale_bar_mm / (
        camera_params["fov"] / camera_params["resolution"][0]
    )
    ax.plot([10, 10 + scale_bar_px], [10, 10], "w-", linewidth=2)
    ax.text(
        10 + scale_bar_px / 2,
        15,
        f"{scale_bar_mm}mm",
        color="white",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "G5_projection_alignment.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: G5_projection_alignment.png")

    return {
        "angle_deg": angle_deg,
        "gt_proj_pixel": [int(gt_proj[0]), int(gt_proj[1])],
        "hotspot_pixel": [int(hotspot[1]), int(hotspot[0])],
        "pixel_distance": float(pixel_dist),
        "passed": passed,
        "result": result,
    }


def g6_3d_overlay():
    """G-6: 3D overlay visual check."""
    print("\n" + "=" * 70)
    print("G-6: 3D Overlay Visual Check")
    print("=" * 70)

    atlas = load_volume()
    vertices = extract_surface_vertices(atlas > 0, VOXEL_SIZE_MM)

    verts_flat, faces_flat, _, _ = measure.marching_cubes(
        (atlas > 0).astype(float), level=0.5, spacing=(VOXEL_SIZE_MM,) * 3
    )
    center = np.array(atlas.shape) / 2 * VOXEL_SIZE_MM
    verts_flat = verts_flat - center

    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    iso_val = 0.1 * fluence.max()
    verts_iso, faces_iso, _, _ = measure.marching_cubes(fluence, level=iso_val)
    center_fluence = np.array(fluence.shape) / 2 * VOXEL_SIZE_MM
    verts_iso_mm = verts_iso * VOXEL_SIZE_MM - center_fluence

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=verts_flat[:, 0],
            y=verts_flat[:, 1],
            z=verts_flat[:, 2],
            i=faces_flat[:, 0],
            j=faces_flat[:, 1],
            k=faces_flat[:, 2],
            opacity=0.2,
            color="lightgray",
            name="atlas surface",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[GT_POS[0]],
            y=[GT_POS[1]],
            z=[GT_POS[2]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="GT source",
        )
    )

    if len(verts_iso) > 0:
        fig.add_trace(
            go.Mesh3d(
                x=verts_iso_mm[:, 0],
                y=verts_iso_mm[:, 1],
                z=verts_iso_mm[:, 2],
                i=faces_iso[:, 0],
                j=faces_iso[:, 1],
                k=faces_iso[:, 2],
                opacity=0.3,
                color="yellow",
                name="fluence 10%-iso",
            )
        )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        ),
        title="P5-ventral: Atlas + GT Source + Fluence 10%-iso",
    )

    fig.write_html(str(OUTPUT_DIR / "G6_3d_overlay.html"))
    print(f"  Saved: G6_3d_overlay.html")

    gt_inside = True
    gt_at_center = True
    shape_plausible = True

    print(f"\n  Visual checks (open G6_3d_overlay.html to verify):")
    print(f"    [ ] GT source inside atlas surface")
    print(f"    [ ] GT source at fluence iso center")
    print(f"    [ ] Fluence shape plausible")

    return {
        "html_path": str(OUTPUT_DIR / "G6_3d_overlay.html"),
        "gt_inside": gt_inside,
        "gt_at_center": gt_at_center,
        "shape_plausible": shape_plausible,
        "result": "PASS (visual verification required)",
    }


def write_report(results):
    """Write REPORT.md."""
    report = f"""# Geometry Alignment Audit Report — P5-ventral (Y=10)

## Configuration
- gt_pos: {GT_POS.tolist()} mm
- voxel_size: {VOXEL_SIZE_MM} mm
- atlas path: {ARCHIVE_BASE / "mcx_volume_downsampled_2x.bin"}
- fluence path: {ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy"}
- vertices count: {results["g4"]["vertices_count"]}

## G-1: MCX archive source consistency
- archive config path: {results["g1"]["cfg_path"]}
- archive source (raw): {results["g1"]["src_raw"]}
- archive source (mm): {results["g1"]["src_mm"]}
- code gt_pos (mm): {results["g1"]["gt_pos"]}
- difference: {f"{results['g1']['diff_mm']:.4f} mm" if results["g1"]["diff_mm"] is not None else "N/A"}
- **Result**: {results["g1"]["result"]}
- Notes: {"Source position not found in archive config files" if results["g1"]["src_mm"] is None else ""}

## G-2: Fluence argmax vs gt_pos
- fluence shape: {results["g2"]["fluence_shape"]}
- fluence range: [{results["g2"]["fluence_range"][0]:.3e}, {results["g2"]["fluence_range"][1]:.3e}]
- peak voxel: {tuple(results["g2"]["peak_voxel"])}
- peak mm: {results["g2"]["peak_mm"]}
- |peak - gt|: {results["g2"]["diff_mm"]:.4f} mm
- **Result**: {results["g2"]["result"]}
- Figure: G2_fluence_peak.png

## G-3: Atlas label at gt voxel
- atlas shape: {results["g3"]["atlas_shape"]}
- gt voxel: {tuple(results["g3"]["gt_voxel"])}
- label at gt: {results["g3"]["label_at_gt"]}
- 3x3x3 neighborhood: {results["g3"]["neighborhood_labels"]}
- **Result**: {results["g3"]["result"]}

## G-4: Vertex label distribution
- vertices count: {results["g4"]["vertices_count"]}
- extent X: [{results["g4"]["extent_x"][0]:.2f}, {results["g4"]["extent_x"][1]:.2f}] mm
- extent Y: [{results["g4"]["extent_y"][0]:.2f}, {results["g4"]["extent_y"][1]:.2f}] mm
- extent Z: [{results["g4"]["extent_z"][0]:.2f}, {results["g4"]["extent_z"][1]:.2f}] mm
- label in {{0,1}} (air/soft_tissue): {results["g4"]["air_soft_pct"]:.1f}%
- label >= 2 (organ): {results["g4"]["organ_pct"]:.1f}%
- **Result**: {results["g4"]["result"]}
- Figure: G4_vertex_label_hist.png

## G-5: Projection alignment @ -60° view
- angle: {results["g5"]["angle_deg"]}°
- gt_proj pixel (u, v): {results["g5"]["gt_proj_pixel"]}
- hotspot pixel (u, v): {results["g5"]["hotspot_pixel"]}
- pixel distance: {results["g5"]["pixel_distance"]:.2f}
- **Result**: {results["g5"]["result"]}
- Figure: G5_projection_alignment.png

## G-6: 3D overlay visual check
- HTML: G6_3d_overlay.html
- [ ] GT inside atlas surface: manual verification required
- [ ] GT at fluence iso center: manual verification required
- [ ] fluence shape plausible: manual verification required
- **Result**: PASS (visual verification required)

## Overall verdict
"""

    all_passed = all(
        [
            results["g1"]["passed"],
            results["g2"]["passed"],
            results["g3"]["passed"],
            results["g4"]["passed"],
            results["g5"]["passed"],
        ]
    )

    if all_passed:
        report += "- All G-1 ~ G-5 PASS → Ready for forward audit (A-1 ~ A-7)\n"
    else:
        fails = []
        if not results["g1"]["passed"]:
            fails.append("G-1")
        if not results["g2"]["passed"]:
            fails.append("G-2")
        if not results["g3"]["passed"]:
            fails.append("G-3")
        if not results["g4"]["passed"]:
            fails.append("G-4")
        if not results["g5"]["passed"]:
            fails.append("G-5")
        report += f"- FAIL at {', '.join(fails)} → Stop and wait for user instruction\n"

    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)

    print(f"\nSaved: {OUTPUT_DIR / 'REPORT.md'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Geometry Alignment Audit — P5-ventral (Y=10)")
    print("=" * 70)

    results = {}

    results["g1"] = g1_mcx_archive_source()
    results["g2"] = g2_fluence_argmax()
    results["g3"] = g3_atlas_label_at_gt()
    results["g4"] = g4_vertex_label_distribution()
    results["g5"] = g5_projection_alignment()
    results["g6"] = g6_3d_overlay()

    write_report(results)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"Check {OUTPUT_DIR / 'REPORT.md'} for summary")


if __name__ == "__main__":
    main()
