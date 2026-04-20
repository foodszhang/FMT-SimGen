#!/usr/bin/env python3
"""
U6.5: Single-sample pilot QA for sample_0000.

Generates comprehensive QA outputs in output/qa/sample_0000/:
  - U5 assertion log (from build_shared_assets)
  - Mesh × mcx_volume 3-slice overlay (axial/coronal/sagittal)
  - Tumor position dual-source verification (L2 check)
  - Visibility UNION snapshot @ ε=0.5
  - DE solution sanity check
  - Projection snapshots (requires MCX jnii)

Usage:
    uv run python scripts/qa_pilot_sample.py [--sample 0000]
"""
import sys
from pathlib import Path
import json
import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.frame_contract import (
    TRUNK_SIZE_MM, VOXEL_SIZE_MM, TRUNK_GRID_SHAPE,
    VOLUME_CENTER_WORLD, assert_in_trunk_bbox,
)
from fmt_simgen.view_config import TurntableCamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path("data/default/samples/sample_0000")
SHARED_DIR = Path("output/shared")
QA_DIR = Path("output/qa/sample_0000")
MCX_VOL_PATH = SHARED_DIR / "mcx_volume_trunk.bin"


def load_shared():
    """Load mesh and mcx_volume."""
    mesh = np.load(SHARED_DIR / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float64)
    elements = mesh["elements"]
    surface_faces = mesh["surface_faces"]
    tissue_labels = mesh["tissue_labels"]

    # MCX volume: ZYX order, uint8 labels
    mcx_raw = np.fromfile(MCX_VOL_PATH, dtype=np.uint8)
    mcx_xyz = mcx_raw.reshape((104, 200, 190)).transpose(2, 1, 0)  # XYZ uint8

    return nodes, elements, surface_faces, tissue_labels, mcx_xyz


def load_sample():
    """Load sample_0000 DE outputs."""
    tumor_params = json.load(open(SAMPLE_DIR / "tumor_params.json"))
    measurement_b = np.load(SAMPLE_DIR / "measurement_b.npy")
    gt_nodes = np.load(SAMPLE_DIR / "gt_nodes.npy")
    return tumor_params, measurement_b, gt_nodes


# ─── Gate 1: U5 assertion (replay) ────────────────────────────────────────────

def gate1_u5_assertion():
    """Re-run U5 frame consistency check."""
    logger.info("=" * 60)
    logger.info("Gate 1: U5 Frame Consistency Assertion")
    logger.info("=" * 60)

    nodes, _, _, _, mcx_xyz = load_shared()

    # MCX body voxel centers in trunk-local mm
    body_idx = np.argwhere(mcx_xyz > 0)
    mcx_body_mm = body_idx.astype(np.float64) * VOXEL_SIZE_MM

    for axis, name in enumerate("XYZ"):
        m_lo, m_hi = nodes[:, axis].min(), nodes[:, axis].max()
        v_lo, v_hi = mcx_body_mm[:, axis].min(), mcx_body_mm[:, axis].max()
        diff_lo = abs(m_lo - v_lo)
        diff_hi = abs(m_hi - v_hi)
        logger.info(
            f"  {name}: mesh [{m_lo:.2f}, {m_hi:.2f}] vs mcx [{v_lo:.2f}, {v_hi:.2f}] "
            f"(diff_lo={diff_lo:.2f}, diff_hi={diff_hi:.2f})"
        )
        assert diff_lo < 2.0 and diff_hi < 2.0, (
            f"U5 FAILED: {name} axis diff exceeds 2mm"
        )

    logger.info("✅ U5 assertion PASSED: mesh ↔ mcx_volume < 2mm per axis")
    return True


# ─── Gate 2: 3-slice overlay ─────────────────────────────────────────────────

def _plot_slice_with_mesh(ax, mcx_xyz, nodes, surface_faces, axis, slice_idx_mm,
                          title, cmap="viridis"):
    """Plot a 2D slice with mesh overlay."""
    axes_dict = {"X": 0, "Y": 1, "Z": 2}

    # Convert mm to voxel index (mcx_xyz is indexed by integers)
    slice_idx_vox = int(round(slice_idx_mm / VOXEL_SIZE_MM))

    # Transpose mcx_xyz to the right view order
    if axis == "X":
        # Coronal: view from front (Y-Z plane), x=slice_idx_mm
        vol = mcx_xyz[slice_idx_vox, :, :].T  # [Y, Z]
        y_lim = (0, 40); z_lim = (0, 20.8)
        ax.set_ylabel("Z (mm)"); ax.set_xlabel("Y (mm)")
        y_coords = np.arange(0, 40, VOXEL_SIZE_MM)[:vol.shape[0]]
        z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]
        ax.imshow(vol, origin="lower", cmap=cmap, aspect="auto",
                  extent=[y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]])
        # Overlay mesh nodes in this plane
        node_slice = nodes[np.abs(nodes[:, 0] - slice_idx_mm) < 2.0]
        ax.scatter(node_slice[:, 1], node_slice[:, 2], s=0.5, c="red", alpha=0.5)

    elif axis == "Y":
        # Axial: view from top (X-Z plane), y=slice_idx_mm
        vol = mcx_xyz[:, slice_idx_vox, :].T  # [X, Z]
        x_lim = (0, 38); z_lim = (0, 20.8)
        ax.set_ylabel("Z (mm)"); ax.set_xlabel("X (mm)")
        x_coords = np.arange(0, 38, VOXEL_SIZE_MM)[:vol.shape[0]]
        z_coords = np.arange(0, 20.8, VOXEL_SIZE_MM)[:vol.shape[1]]
        ax.imshow(vol, origin="lower", cmap=cmap, aspect="auto",
                  extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]])
        node_slice = nodes[np.abs(nodes[:, 1] - slice_idx_mm) < 2.0]
        ax.scatter(node_slice[:, 0], node_slice[:, 2], s=0.5, c="red", alpha=0.5)

    elif axis == "Z":
        # Sagittal: view from side (X-Y plane), z=slice_idx_mm
        vol = mcx_xyz[:, :, slice_idx_vox].T  # [X, Y]
        x_lim = (0, 38); y_lim = (0, 40)
        ax.set_ylabel("Y (mm)"); ax.set_xlabel("X (mm)")
        x_coords = np.arange(0, 38, VOXEL_SIZE_MM)[:vol.shape[0]]
        y_coords = np.arange(0, 40, VOXEL_SIZE_MM)[:vol.shape[1]]
        ax.imshow(vol, origin="lower", cmap=cmap, aspect="auto",
                  extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
        node_slice = nodes[np.abs(nodes[:, 2] - slice_idx_mm) < 2.0]
        ax.scatter(node_slice[:, 0], node_slice[:, 1], s=0.5, c="red", alpha=0.5)

    ax.set_title(title)


def gate2_mesh_mcx_overlay():
    """Gate 2: 3-slice overlay at z=10mm, y=20mm, x=19mm."""
    logger.info("Gate 2: Mesh × MCX Volume overlay")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    nodes, _, surface_faces, _, mcx_xyz = load_shared()

    slices = [
        ("Z", 10.0, "Axial (Z=10mm)"),
        ("Y", 20.0, "Coronal (Y=20mm)"),
        ("X", 19.0, "Sagittal (X=19mm)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (axis, mm, title) in enumerate(slices):
        _plot_slice_with_mesh(
            axes[i], mcx_xyz, nodes, surface_faces,
            axis, mm, title
        )
        axes[i].set_xlim(0, 40)
        axes[i].set_ylim(0, 21)
    fig.tight_layout()
    out = QA_DIR / "overlay_slices.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Saved {out}")
    plt.close()

    # Also save individual slices
    for axis, mm, name in slices:
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        _plot_slice_with_mesh(ax2, mcx_xyz, nodes, surface_faces, axis, mm, name)
        ax2.set_xlim(0, 40); ax2.set_ylim(0, 21)
        fig2.tight_layout()
        out2 = QA_DIR / f"overlay_{axis.lower()}={mm}.png"
        fig2.savefig(out2, dpi=150)
        plt.close()
        logger.info(f"Saved {out2}")

    logger.info("✅ Overlay figures saved")
    return True


# ─── Gate 3: Tumor dual-source L2 check ───────────────────────────────────────

def gate3_tumor_dual_source():
    """Gate 3: Verify tumor position in mesh vs MCX volume."""
    logger.info("Gate 3: Tumor dual-source L2 check")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    _, _, _, _, mcx_xyz = load_shared()
    tumor_params, _, _ = load_sample()

    voxel_size = VOXEL_SIZE_MM  # 0.2 mm

    # For each focus, find:
    # 1. MCX tumor voxel center (centroid of tumor voxels in mask)
    # 2. Mesh tumor tet center (centroid of tets with center within tumor radius)
    nodes, elements, _, _, _ = load_shared()
    elem_centers = nodes[elements[:, :4]].mean(axis=1)

    results = []
    for fi, focus in enumerate(tumor_params["foci"]):
        center = np.array(focus["center"])  # trunk-local mm
        rx, ry, rz = focus["rx"], focus["ry"], focus["rz"]
        radius = max(rx, ry, rz)  # approximate radius

        # MCX tumor mask (voxels within the ellipsoid)
        # center_voxel = center / voxel_size
        center_vox = center / voxel_size

        # Build ellipsoid mask in MCX volume
        x_vox = np.arange(mcx_xyz.shape[0])
        y_vox = np.arange(mcx_xyz.shape[1])
        z_vox = np.arange(mcx_xyz.shape[2])
        xv, yv, zv = np.meshgrid(x_vox, y_vox, z_vox, indexing='ij')

        # Ellipsoid check (in voxel space)
        dx = (xv - center_vox[0]) / (radius / voxel_size)
        dy = (yv - center_vox[1]) / (radius / voxel_size)
        dz = (zv - center_vox[2]) / (radius / voxel_size)
        ellip_mask = (dx**2 + dy**2 + dz**2) <= 1.0

        tumor_voxels = np.argwhere(ellip_mask & (mcx_xyz > 0))
        if len(tumor_voxels) > 0:
            mcx_tumor_center = tumor_voxels.mean(axis=0) * voxel_size
        else:
            mcx_tumor_center = None

        # Mesh tumor tets (tet center within radius of tumor center)
        dists = np.linalg.norm(elem_centers - center, axis=1)
        tumor_tets_mask = dists < radius
        if tumor_tets_mask.sum() > 0:
            mesh_tumor_center = elem_centers[tumor_tets_mask].mean(axis=0)
        else:
            mesh_tumor_center = None

        if mcx_tumor_center is not None and mesh_tumor_center is not None:
            l2 = np.linalg.norm(mcx_tumor_center - mesh_tumor_center)
        else:
            l2 = None

        results.append({
            "focus_idx": fi,
            "tumor_center_mm": center.tolist(),
            "mcx_tumor_center_mm": mcx_tumor_center.tolist() if mcx_tumor_center is not None else None,
            "mesh_tumor_center_mm": mesh_tumor_center.tolist() if mesh_tumor_center is not None else None,
            "l2_distance_mm": float(l2) if l2 is not None else None,
            "n_tumor_voxels": int(len(tumor_voxels)),
            "n_tumor_tets": int(tumor_tets_mask.sum()),
        })
        logger.info(
            f"  Focus {fi}: center={center}mm, "
            f"mcx={mcx_tumor_center if mcx_tumor_center is not None else 'N/A'}, "
            f"mesh={mesh_tumor_center.tolist() if mesh_tumor_center is not None else 'N/A'}, "
            f"L2={l2:.3f}mm" if l2 is not None else "L2=N/A"
        )

    # Save results
    with open(QA_DIR / "tumor_l2_check.json", "w") as f:
        json.dump(results, f, indent=2)

    # L2 check
    all_l2 = [r["l2_distance_mm"] for r in results if r["l2_distance_mm"] is not None]
    if all_l2:
        max_l2 = max(all_l2)
        logger.info(f"Tumor L2 distances: {all_l2}")
        logger.info(f"Max L2 = {max_l2:.3f}mm (threshold: 1mm)")
        if max_l2 < 1.0:
            logger.info("✅ Tumor dual-source PASSED: all L2 < 1mm")
            return True
        else:
            logger.error(f"❌ Tumor dual-source FAILED: max L2 = {max_l2:.3f}mm >= 1mm")
            return False
    else:
        logger.warning("⚠️  Could not compute L2 distances (tumor may be outside MCX volume)")
        return False


# ─── Gate 4: Visibility UNION snapshot ───────────────────────────────────────

def gate4_visibility_union():
    """Gate 4: Visibility UNION snapshot @ ε=0.5 using project_volume_reference."""
    logger.info("Gate 4: Visibility UNION snapshot @ ε=0.5")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    from fmt_simgen.view_config import get_visible_surface_nodes_from_mcx_depth

    nodes, _, surface_faces, _, mcx_xyz = load_shared()

    # Compute depth map using mcx_volume > 0 (tissue surface)
    # project_volume_reference(mcx_volume > 0, vcw=VOLUME_CENTER_WORLD, voxel=VOXEL_SIZE_MM)
    mask_xyz = mcx_xyz > 0

    # Get surface nodes visible from all 7 angles
    camera = TurntableCamera({"pose": "prone"})
    angles = camera.angles  # [-90, -60, -30, 0, 30, 60, 90]

    results = {}
    union_epsilon = 0.5
    union_visible = np.zeros(len(nodes), dtype=bool)

    logger.info(f"  ε = {union_epsilon}mm (bilateral diff tolerance)")
    for angle in angles:
        finite_px, visible_nodes, depth_map = get_visible_surface_nodes_from_mcx_depth(
            nodes, surface_faces,
            mask_xyz=mask_xyz,
            angle_deg=angle,
            voxel_size=VOXEL_SIZE_MM,
            volume_center_world=VOLUME_CENTER_WORLD,
            epsilon=union_epsilon,
        )
        results[angle] = {
            "finite_px": int(finite_px),
            "n_visible_nodes": int(visible_nodes.sum()),
            "depth_min": float(depth_map[depth_map > 0].min()) if (depth_map > 0).any() else 0,
            "depth_max": float(depth_map.max()),
        }
        union_visible |= visible_nodes
        logger.info(
            f"  Angle {angle:4d}°: finite_px={finite_px:6d}, "
            f"visible_nodes={visible_nodes.sum():5d}, "
            f"depth=[{results[angle]['depth_min']:.1f}, {results[angle]['depth_max']:.1f}]mm"
        )

    logger.info(f"  UNION @ ε={union_epsilon}mm: {union_visible.sum()} nodes")

    with open(QA_DIR / "visibility_union.json", "w") as f:
        json.dump({"epsilon_mm": union_epsilon, "union_count": int(union_visible.sum()),
                   "per_angle": results}, f, indent=2)

    target_lo, target_hi = 2500, 3500
    if target_lo <= union_visible.sum() <= target_hi:
        logger.info(f"✅ Visibility UNION PASSED: {union_visible.sum()} ∈ [{target_lo}, {target_hi}]")
        return True
    else:
        logger.error(f"❌ Visibility UNION FAILED: {union_visible.sum()} ∉ [{target_lo}, {target_hi}]")
        return False


# ─── Gate 5: DE solution sanity ───────────────────────────────────────────────

def gate5_de_solution_sanity():
    """Gate 5: DE forward solution sanity check."""
    logger.info("Gate 5: DE solution sanity check")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    _, measurement_b, gt_nodes = load_sample()
    tumor_params, _, _ = load_sample()

    b = measurement_b
    logger.info(f"  measurement_b: shape={b.shape}, min={b.min():.6f}, max={b.max():.6f}")

    # Check 1: Not exploding
    if b.max() > 1e6:
        logger.error(f"❌ DE FAILED: max={b.max():.2e} > 1e6 (exploding solution)")
        return False
    logger.info(f"  ✅ No explosion: max={b.max():.6f} < 1e6")

    # Check 2: Source position should have max value
    # Find node closest to tumor focus center
    nodes, _, _, _, _ = load_shared()
    focus0_center = np.array(tumor_params["foci"][0]["center"])
    # measurement_b corresponds to surface nodes (first len(b) nodes in mesh)
    surf_nodes = nodes[:len(b)]
    dists = np.linalg.norm(surf_nodes - focus0_center, axis=1)
    nearest_node = np.argmin(dists)
    source_val = b[nearest_node]
    max_val = b.max()
    logger.info(
        f"  Source node {nearest_node} (dist={dists[nearest_node]:.2f}mm): "
        f"b={source_val:.6f}, max={max_val:.6f}"
    )
    if source_val < max_val * 0.1:
        logger.warning(f"  ⚠️  Source value {source_val:.6f} << max {max_val:.6f}")
    else:
        logger.info(f"  ✅ Source at maximum region")

    # Check 3: Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(b, bins=50, edgecolor="black")
    axes[0].set_xlabel("Fluence"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"DE Solution Distribution\nmin={b.min():.4f}, max={b.max():.4f}")
    axes[0].axvline(source_val, color="red", linestyle="--", label=f"source @ node {nearest_node}")
    axes[0].legend()

    # Spatial distribution
    surf_nodes = nodes[:len(b)]  # measurement_b is for surface nodes
    sc = axes[1].scatter(surf_nodes[:, 0], surf_nodes[:, 1], c=b, s=5, cmap="hot")
    axes[1].set_xlabel("X (mm)"); axes[1].set_ylabel("Y (mm)")
    axes[1].set_title("DE Surface Fluence"); plt.colorbar(sc, ax=axes[1])

    fig.tight_layout()
    out = QA_DIR / "de_solution_slice.png"
    fig.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved {out}")

    logger.info("✅ DE solution sanity PASSED")
    return True


# ─── Gate 6: Projection snapshots ─────────────────────────────────────────────

def gate6_projection_snapshots():
    """Gate 6: Projection snapshots @ 0° and 90°."""
    logger.info("Gate 6: Projection snapshots")
    QA_DIR.mkdir(parents=True, exist_ok=True)

    proj_path = SAMPLE_DIR / "proj.npz"
    if not proj_path.exists():
        logger.warning("⚠️  proj.npz not found — MCX pipeline may not have completed yet")
        logger.warning("  Skipping projection snapshots (will retry after MCX completes)")
        return None  # Inconclusive

    p = np.load(proj_path)
    logger.info(f"  proj.npz keys: {list(p.keys())}")

    # Check depth range
    depth_0 = p.get("-90") if "-90" in p else p.get("0")
    depth_90 = p.get("90") if "90" in p else None

    results = {}
    for angle_str, depth_arr in p.items():
        valid = depth_arr > 0
        results[angle_str] = {
            "finite_px": int(valid.sum()),
            "depth_min": float(depth_arr[valid].min()) if valid.any() else 0,
            "depth_max": float(depth_arr.max()),
        }
        logger.info(
            f"  Angle {angle_str}: finite_px={valid.sum():6d}, "
            f"depth=[{results[angle_str]['depth_min']:.1f}, {results[angle_str]['depth_max']:.1f}]mm"
        )

    # Save 0° and 90° figures
    for angle_str in ["0", "90", "-90"]:
        if angle_str not in p:
            continue
        depth = p[angle_str]
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(depth, cmap="bone", origin="lower")
        ax.set_title(f"Projection @ {angle_str}°\nfinite_px={results[angle_str]['finite_px']}, "
                     f"depth=[{results[angle_str]['depth_min']:.1f}, {results[angle_str]['depth_max']:.1f}]mm")
        plt.colorbar(im, ax=ax, label="Depth (mm)")
        fig.tight_layout()
        out = QA_DIR / f"proj_{angle_str}deg.png"
        fig.savefig(out, dpi=150)
        plt.close()
        logger.info(f"Saved {out}")

    # Save per-angle results
    with open(QA_DIR / "projection_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    # Validate: depth ∈ [180, 215]mm and finite_px @ 0° ∈ [10000, 18000]
    angle_0_key = "0" if "0" in p else "-90"
    if angle_0_key in results:
        r = results[angle_0_key]
        d_ok = 180 <= r["depth_min"] and r["depth_max"] <= 215
        px_ok = 10000 <= r["finite_px"] <= 18000
        if d_ok and px_ok:
            logger.info(f"✅ Projection @ {angle_0_key}° PASSED: depth={r['depth_min']:.1f}-{r['depth_max']:.1f}mm, px={r['finite_px']}")
            return True
        else:
            logger.warning(f"⚠️  Projection @ {angle_0_key}° issues: depth_ok={d_ok}, px_ok={px_ok}")
            return False

    return None


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="U6.5 pilot QA")
    parser.add_argument("--sample", default="0000")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("U6.5 Pilot QA — sample_0000")
    logger.info("=" * 60)

    results = {}

    # Gate 1: U5 assertion
    try:
        results["gate1_u5"] = gate1_u5_assertion()
    except Exception as e:
        logger.error(f"❌ Gate 1 U5 FAILED: {e}")
        results["gate1_u5"] = False

    # Gate 2: Mesh × MCX overlay
    try:
        results["gate2_overlay"] = gate2_mesh_mcx_overlay()
    except Exception as e:
        logger.error(f"❌ Gate 2 overlay FAILED: {e}")
        import traceback; traceback.print_exc()
        results["gate2_overlay"] = False

    # Gate 3: Tumor dual-source
    try:
        results["gate3_tumor"] = gate3_tumor_dual_source()
    except Exception as e:
        logger.error(f"❌ Gate 3 tumor FAILED: {e}")
        import traceback; traceback.print_exc()
        results["gate3_tumor"] = False

    # Gate 4: Visibility UNION
    try:
        results["gate4_union"] = gate4_visibility_union()
    except Exception as e:
        logger.error(f"❌ Gate 4 visibility FAILED: {e}")
        import traceback; traceback.print_exc()
        results["gate4_union"] = False

    # Gate 5: DE solution sanity
    try:
        results["gate5_de"] = gate5_de_solution_sanity()
    except Exception as e:
        logger.error(f"❌ Gate 5 DE sanity FAILED: {e}")
        import traceback; traceback.print_exc()
        results["gate5_de"] = False

    # Gate 6: Projection snapshots (may be missing if MCX not done)
    try:
        results["gate6_proj"] = gate6_projection_snapshots()
    except Exception as e:
        logger.error(f"❌ Gate 6 projection FAILED: {e}")
        import traceback; traceback.print_exc()
        results["gate6_proj"] = False

    # Summary
    logger.info("=" * 60)
    logger.info("U6.5 Gate Summary")
    logger.info("=" * 60)
    for gate, result in results.items():
        icon = "✅" if result is True else ("⚠️" if result is None else "❌")
        logger.info(f"  {icon} {gate}: {result}")

    # Save summary
    with open(QA_DIR / "gate_summary.json", "w") as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)

    # Determine pass/fail
    blocking_gates = ["gate1_u5", "gate2_overlay", "gate3_tumor", "gate4_union", "gate5_de"]
    blocking_results = {k: v for k, v in results.items() if k in blocking_gates}
    if all(v is True for v in blocking_results.values()):
        logger.info("=" * 60)
        logger.info("✅ ALL BLOCKING GATES PASSED — Ready for U7")
        logger.info("=" * 60)
    else:
        failed = [k for k, v in blocking_results.items() if v is not True]
        logger.error(f"❌ BLOCKING GATES FAILED: {failed}")
        logger.error("Fix issues before proceeding to U7.")


if __name__ == "__main__":
    main()
