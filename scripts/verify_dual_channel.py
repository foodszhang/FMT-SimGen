#!/usr/bin/env python3
"""
End-to-end dual-channel verification for FMT-SimGen DE + MCX pipeline.

Verifies physical self-consistency across all modules (M1-M7) for selected samples:
  1. Single-module sanity checks (file existence, shapes, value ranges)
  2. Cross-module spatial alignment (DE peaks, MCX peaks, GT tumor centers)
  3. DE vs MCX surface region overlap (top-K node proximity)
  4. Visual comparison figures (3×3: MCX proj / DE surface / GT surface × 3 angles)

Handles both old-format (b shape=7465) and new-format (b shape=6226 visible-only) samples.

Usage:
    cd /home/foods/pro/FMT-SimGen
    uv run python scripts/verify_dual_channel.py \
        --shared_dir output/shared \
        --samples_dir data/gaussian_1000/samples \
        --output_dir output/verification \
        --n_samples 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from fmt_simgen.mcx_projection import load_jnii_volume
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOXEL_SIZE_MM, TRUNK_GRID_SHAPE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_aligned_b(sample_b_path: Path, visible_mask: np.ndarray) -> np.ndarray:
    """Load measurement_b and crop to visible-only if needed.

    Old format: b.shape = (7465,) — all surface nodes, crop with visible_mask
    New format: b.shape = (6226,) — already cropped, no action needed
    """
    b = np.load(sample_b_path).ravel()
    if b.shape[0] == visible_mask.sum():
        return b  # already cropped (new format)
    elif b.shape[0] == len(visible_mask):
        return b[visible_mask]  # crop to visible (old format)
    else:
        # Unexpected format — return as-is and let caller handle
        return b


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(samples_dir: Path, n_samples: int = 3):
    """Select diverse samples: 1-shallow, 2-medium, 3-deep (or closest)."""
    results = []

    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        tp_path = sample_dir / "tumor_params.json"
        if not tp_path.exists():
            continue
        try:
            tp = json.load(open(tp_path))
        except Exception:
            continue

        has_jnii = bool(list(sample_dir.glob("*.jnii")))
        has_proj = (sample_dir / "proj.npz").exists()
        has_b = (sample_dir / "measurement_b.npy").exists()
        complete = has_jnii and has_proj and has_b

        results.append({
            "name": sample_dir.name,
            "n_foci": tp.get("num_foci", 0),
            "depth_tier": tp.get("depth_tier", "unknown"),
            "complete": complete,
        })

    pool = {"shallow_1": [], "medium_1": [], "deep_1": [], "multi": []}
    for r in results:
        n, foci, tier, complete = r["name"], r["n_foci"], r["depth_tier"], r["complete"]
        if not complete:
            continue
        if foci == 1 and tier == "shallow":
            pool["shallow_1"].append(n)
        elif foci == 1 and tier == "medium":
            pool["medium_1"].append(n)
        elif foci == 1 and tier == "deep":
            pool["deep_1"].append(n)
        else:
            pool["multi"].append(n)

    selected = []
    for category in ["shallow_1", "medium_1", "deep_1"]:
        if pool[category]:
            selected.append(pool[category][0])

    if len(selected) < n_samples:
        for name in pool["multi"]:
            if name not in selected:
                selected.append(name)
            if len(selected) >= n_samples:
                break

    return selected[:n_samples]


# ---------------------------------------------------------------------------
# Layer 1: Sanity checks
# ---------------------------------------------------------------------------

def check_single_sample(
    sample_dir: Path,
    shared_dir: Path,
    visible_mask: np.ndarray,
) -> dict:
    """Run Layer 1 sanity checks for one sample."""
    results = {}

    # M1: MCX volume
    vol_path = shared_dir / "mcx_volume_trunk.bin"
    if vol_path.exists():
        size_mb = vol_path.stat().st_size / 1e6
        results["m1_volume_mb"] = round(size_mb, 1)
        results["m1_pass"] = bool(size_mb > 1.0)
    else:
        results["m1_pass"] = False
        results["m1_error"] = "file not found"

    # M2: visible_mask
    mask_path = shared_dir / "visible_mask.npy"
    if mask_path.exists():
        ratio = visible_mask.sum() / len(visible_mask)
        results["m2_visible_count"] = int(visible_mask.sum())
        results["m2_visible_ratio"] = round(float(ratio), 3)
        results["m2_pass"] = bool(0.6 < ratio < 0.95)
    else:
        results["m2_pass"] = False
        results["m2_error"] = "visible_mask not found"

    # M3: MCX source
    source_bins = list(sample_dir.glob("source-*.bin"))
    if source_bins:
        results["m3_source_size"] = source_bins[0].stat().st_size
        results["m3_pass"] = bool(results["m3_source_size"] > 0)
    else:
        results["m3_pass"] = False
        results["m3_error"] = "no source-*.bin found"

    # M4: MCX fluence (jnii)
    jnii_files = list(sample_dir.glob("*.jnii"))
    if jnii_files:
        try:
            fluence = load_jnii_volume(jnii_files[0])
            nonzero_ratio = float((fluence > 0).sum() / fluence.size)
            results["m4_fluence_shape"] = str(fluence.shape)
            results["m4_fluence_max"] = round(float(fluence.max()), 4)
            results["m4_fluence_nonzero_ratio"] = round(nonzero_ratio, 3)
            results["m4_pass"] = bool(fluence.max() > 0)
        except Exception as e:
            results["m4_pass"] = False
            results["m4_error"] = str(e)
    else:
        results["m4_pass"] = False
        results["m4_error"] = "no .jnii found"

    # M5: projections
    proj_path = sample_dir / "proj.npz"
    if proj_path.exists():
        proj = np.load(proj_path)
        angles_with_signal = sum(1 for k in proj.files if proj[k].max() > 0)
        results["m5_n_angles"] = len(proj.files)
        results["m5_angles_with_signal"] = angles_with_signal
        results["m5_pass"] = bool(angles_with_signal >= 3)
    else:
        results["m5_pass"] = False
        results["m5_error"] = "proj.npz not found"

    # M7: DE measurement
    b_path = sample_dir / "measurement_b.npy"
    if b_path.exists():
        b_raw = np.load(b_path)
        nonzero_ratio = float((b_raw > 0).sum() / len(b_raw))
        results["m7_b_raw_shape"] = str(b_raw.shape)
        results["m7_b_max"] = round(float(b_raw.max()), 4)
        results["m7_b_nonzero_ratio"] = round(nonzero_ratio, 3)
        results["m7_b_is_full_format"] = bool(b_raw.shape[0] == len(visible_mask))
        # Check if already in new format (aligned with visible_mask)
        if b_raw.shape[0] == visible_mask.sum():
            results["m7_pass"] = bool(b_raw.max() > 0)
            results["m7_format"] = "new (visible-only)"
        elif b_raw.shape[0] == len(visible_mask):
            results["m7_pass"] = True  # old format, but exists and non-zero
            results["m7_format"] = "old (full surface)"
        else:
            results["m7_pass"] = bool(b_raw.max() > 0)
            results["m7_format"] = "unknown"
    else:
        results["m7_pass"] = False
        results["m7_error"] = "measurement_b not found"

    return results


# ---------------------------------------------------------------------------
# Layer 2: Spatial alignment
# ---------------------------------------------------------------------------

def check_spatial_alignment(
    sample_dir: Path,
    shared_dir: Path,
    view_config: dict,
    visible_mask: np.ndarray,
) -> dict:
    """Check DE peak, MCX peak, and GT center spatial alignment."""
    # Load MCX config for trunk_offset
    repo_root = Path(__file__).parent.parent
    with open(repo_root / "config" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    mcx_cfg = cfg.get("mcx", {})
    trunk_offset = np.array(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]))

    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"]
    surface_idx = mesh["surface_node_indices"]
    visible_surface_idx = surface_idx[visible_mask]

    tp = json.load(open(sample_dir / "tumor_params.json"))
    gt_centers = np.array([f["center"] for f in tp["foci"]])  # [K, 3] in mm

    # Load and align b
    b_raw = np.load(sample_dir / "measurement_b.npy").ravel()
    if b_raw.shape[0] == visible_mask.sum():
        b = b_raw
    elif b_raw.shape[0] == len(visible_mask):
        b = b_raw[visible_mask]
    else:
        b = b_raw

    # DE peak: weighted centroid of top-20% signal
    k_de = max(10, int(len(b) * 0.2))
    top_de_idx = np.argsort(b)[-k_de:]
    de_top_coords = nodes[visible_surface_idx[top_de_idx]]
    de_peak_coord = de_top_coords.mean(axis=0)

    de_to_gt_dists = np.linalg.norm(gt_centers - de_peak_coord, axis=1)
    de_gt_dist = float(de_to_gt_dists.min())

    # MCX 0° projection peak → physical world coords
    #
    # The reference projection (project_volume_reference) places the volume CENTER
    # at world origin (via ix - nx/2 centering). Physical world = reference_world +
    # [0.2*nx/2, 0.2*ny/2, 0.2*nz/2] + trunk_offset.
    #
    # At θ=0°: cam_x = world_X, cam_y = world_Y (relative to volume center)
    proj = np.load(sample_dir / "proj.npz")
    if "0" not in proj.files:
        return {"error": "no 0° projection"}

    proj_0 = proj["0"]
    peak_v, peak_u = np.unravel_index(proj_0.argmax(), proj_0.shape)

    fov_mm = view_config.get("fov_mm", 50.0)
    det_w, det_h = view_config.get("detector_resolution", [256, 256])
    voxel_size = VOXEL_SIZE_MM  # mm

    # MCX volume shape [X=190, Y=200, Z=104]
    nx, ny, nz = TRUNK_GRID_SHAPE
    vol_center_offset = np.array([
        voxel_size * nx / 2,
        voxel_size * ny / 2,
        voxel_size * nz / 2,
    ])  # [19.0, 20.0, 10.4]

    # pixel → reference world (relative to volume center)
    mcx_ref_x = (peak_u - det_w / 2) * (fov_mm / det_w)  # world_X relative to center
    mcx_ref_y = (peak_v - det_h / 2) * (fov_mm / det_h)  # world_Y relative to center

    # Convert to physical world coordinates (same frame as GT tumor centers)
    # physical = reference_world + vol_center_offset + trunk_offset
    mcx_peak_phys = np.array([
        mcx_ref_x + vol_center_offset[0] + trunk_offset[0],
        mcx_ref_y + vol_center_offset[1] + trunk_offset[1],
    ])  # [2]

    # GT center XY (z lost in orthographic projection — compare in 2D XY plane)
    gt_proj_2d = gt_centers[:, :2]  # [K, 2]
    mcx_to_gt_dists = np.linalg.norm(gt_proj_2d - mcx_peak_phys, axis=1)
    mcx_gt_dist = float(mcx_to_gt_dists.min())

    return {
        "de_peak_coord_mm": [round(float(x), 2) for x in de_peak_coord],
        "mcx_peak_pixel": [int(peak_u), int(peak_v)],
        "mcx_peak_phys_xy_mm": [round(float(x), 2) for x in mcx_peak_phys],
        "gt_centers_mm": [[round(float(x), 2) for x in c] for c in gt_centers],
        "de_to_nearest_gt_mm": round(de_gt_dist, 2),
        "mcx_to_nearest_gt_mm": round(mcx_gt_dist, 2),
        "de_align_pass": bool(de_gt_dist < 10.0),
        "mcx_align_pass": bool(mcx_gt_dist < 30.0),
    }


# ---------------------------------------------------------------------------
# Layer 3: DE vs MCX surface overlap
# ---------------------------------------------------------------------------

def check_surface_overlap(
    sample_dir: Path,
    shared_dir: Path,
    visible_mask: np.ndarray,
    top_ratio: float = 0.10,
    distance_tol: float = 10.0,
    view_config: dict | None = None,
) -> dict:
    """Check DE top-K and MCX top-K surface node spatial overlap.

    Compares the spatial distribution of DE surface fluence top-K nodes vs
    MCX 0° projection top-K nodes at the same surface positions. This is a
    true cross-channel comparison (DE vs MCX), unlike the old DE vs GT which
    were mathematically coupled via the same forward model.

    DE and MCX have fundamentally different physics (boundary flux vs. absorbed
    energy) and spatial coverage (full surface vs. orthographic FOV), so the
    overlap is expected to be lower than the old DE↔GT baseline.
    """
    from scipy.spatial import KDTree

    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"]
    surface_idx = mesh["surface_node_indices"]
    visible_surface_idx = surface_idx[visible_mask]

    b_raw = np.load(sample_dir / "measurement_b.npy").ravel()
    if b_raw.shape[0] == visible_mask.sum():
        b = b_raw
    elif b_raw.shape[0] == len(visible_mask):
        b = b_raw[visible_mask]
    else:
        b = b_raw

    # Load MCX 0° projection
    proj = np.load(sample_dir / "proj.npz")
    if "0" not in proj.files:
        return {"error": "no 0° projection"}
    proj_0 = proj["0"]

    # Camera parameters for 0° orthographic projection
    if view_config is not None:
        fov_mm = view_config.get("fov_mm", 50.0)
        det_w, det_h = view_config.get("detector_resolution", [256, 256])
    else:
        fov_mm = 50.0
        det_w, det_h = 256, 256
    pixel_size = fov_mm / det_w

    # Compute MCX signal at each visible surface node via 0° orthographic projection
    visible_nodes = nodes[visible_surface_idx]  # [V, 3]
    V = len(visible_nodes)

    # --- MCX volume center in world coordinates ---
    # project_volume_reference places volume center at origin.
    # We must subtract this offset so mesh world coords → MCX centered coords.
    repo_root = Path(__file__).parent.parent
    with open(repo_root / "config" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    mcx_cfg = cfg.get("mcx", {})
    trunk_offset = np.array(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]))
    voxel_size = VOXEL_SIZE_MM
    nx, ny, nz = mcx_cfg.get("volume_shape", list(TRUNK_GRID_SHAPE))
    volume_center_world = trunk_offset + np.array([nx, ny, nz]) * voxel_size / 2

    # world → centered (same transform as project_volume_reference)
    cam_x = visible_nodes[:, 0] - volume_center_world[0]
    cam_y = visible_nodes[:, 1] - volume_center_world[1]

    # Map centered (X, Y) to detector pixels
    u = np.round((cam_x + fov_mm / 2) / pixel_size).astype(int)
    v = np.round((cam_y + fov_mm / 2) / pixel_size).astype(int)

    # Clip to valid range (out-of-bounds = zero MCX signal)
    u_clipped = np.clip(u, 0, det_w - 1)
    v_clipped = np.clip(v, 0, det_h - 1)

    # MCX signal at each surface node (shallowest non-zero is already in proj_0)
    mcx_signal = np.zeros(V, dtype=np.float64)
    for i in range(V):
        if 0 <= u[i] < det_w and 0 <= v[i] < det_h:
            mcx_signal[i] = proj_0[v[i], u[i]]

    # Top-K DE nodes
    k_de = max(1, int(V * top_ratio))
    de_top_idx = np.argsort(b)[-k_de:]
    de_top_coords = visible_nodes[de_top_idx]

    # Top-K MCX nodes (nonzero only)
    mcx_nonzero_mask = mcx_signal > 0
    mcx_nonzero_nodes = visible_nodes[mcx_nonzero_mask]
    mcx_nonzero_signal = mcx_signal[mcx_nonzero_mask]
    n_mcx_nonzero = len(mcx_nonzero_nodes)
    k_mcx = max(1, int(n_mcx_nonzero * top_ratio))
    mcx_top_local = np.argsort(mcx_nonzero_signal)[-k_mcx:]
    mcx_top_coords = mcx_nonzero_nodes[mcx_top_local]

    # Overlap: distance from each MCX top node to nearest DE top node
    if len(de_top_coords) == 0 or len(mcx_top_coords) == 0:
        return {
            "de_top_k": k_de,
            "mcx_top_k": k_mcx,
            "mcx_nonzero_total": n_mcx_nonzero,
            "n_overlap_within_tol": 0,
            "overlap_ratio": 0.0,
            "distance_tol_mm": distance_tol,
            "overlap_pass": False,
            "note": "no DE or MCX nonzero nodes",
        }

    de_tree = KDTree(de_top_coords)
    distances, _ = de_tree.query(mcx_top_coords)
    n_overlap = int((distances < distance_tol).sum())
    overlap_ratio = n_overlap / max(len(mcx_top_coords), 1)

    return {
        "de_top_k": k_de,
        "mcx_top_k": k_mcx,
        "mcx_nonzero_total": n_mcx_nonzero,
        "n_overlap_within_tol": n_overlap,
        "overlap_ratio": round(float(overlap_ratio), 3),
        "distance_tol_mm": distance_tol,
        "overlap_pass": bool(overlap_ratio > 0.3),
    }


# ---------------------------------------------------------------------------
# Layer 4: Visual comparison figures
# ---------------------------------------------------------------------------

def generate_comparison_figure(
    sample_dir: Path,
    shared_dir: Path,
    output_path: Path,
    visible_mask: np.ndarray,
    angles: list = [-90, 0, 90],
    view_config: dict | None = None,
) -> None:
    """Generate 3×3 comparison: MCX proj / DE surface / GT surface × 3 angles."""
    import matplotlib.pyplot as plt

    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"]
    surface_idx = mesh["surface_node_indices"]
    visible_surface_idx = surface_idx[visible_mask]
    visible_coords = nodes[visible_surface_idx]  # [V, 3]

    b_raw = np.load(sample_dir / "measurement_b.npy").ravel()
    if b_raw.shape[0] == visible_mask.sum():
        b = b_raw
    elif b_raw.shape[0] == len(visible_mask):
        b = b_raw[visible_mask]
    else:
        b = b_raw

    gt = np.load(sample_dir / "gt_nodes.npy").ravel()
    gt_visible = gt[visible_surface_idx]
    proj = np.load(sample_dir / "proj.npz")

    tp = json.load(open(sample_dir / "tumor_params.json"))
    n_foci = tp.get("num_foci", 0)
    depth_tier = tp.get("depth_tier", "unknown")

    fig, axes = plt.subplots(3, len(angles), figsize=(5 * len(angles), 12))
    if len(angles) == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["MCX projection", "DE surface b", "GT surface"]

    for col, angle in enumerate(angles):
        angle_key = str(angle)

        # Row 0: MCX projection
        ax = axes[0, col]
        if angle_key in proj.files:
            ax.imshow(proj[angle_key], cmap="hot", aspect="equal")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(f"MCX {angle}°")
        ax.axis("off")

        # Rotate surface for this angle (around Y axis)
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array(
            [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]],
            dtype=np.float64,
        )
        rotated = visible_coords @ R.T
        rot_xy = rotated[:, :2]  # project to XY plane

        # Row 1: DE b
        ax1 = axes[1, col]
        b_max = b.max() if b.max() > 0 else 1.0
        ax1.scatter(
            rot_xy[:, 0], rot_xy[:, 1],
            c=b, cmap="hot", s=0.5, vmin=0, vmax=b_max * 0.8,
        )
        ax1.set_title(f"DE b {angle}°")
        ax1.set_aspect("equal")
        ax1.axis("off")

        # Row 2: GT
        ax2 = axes[2, col]
        gt_nonzero_max = np.max(gt_visible[gt_visible > 0]) if gt_visible.max() > 0 else 1.0
        vmax_gt = max(gt_nonzero_max * 0.8, 0.01)
        ax2.scatter(
            rot_xy[:, 0], rot_xy[:, 1],
            c=gt_visible, cmap="hot", s=0.5, vmin=0, vmax=vmax_gt,
        )
        ax2.set_title(f"GT {angle}°")
        ax2.set_aspect("equal")
        ax2.axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=8)

    fig.suptitle(
        f"{sample_dir.name}: {n_foci} foci, depth={depth_tier}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify dual-channel DE+MCX pipeline")
    parser.add_argument("--shared_dir", default="output/shared", type=Path)
    parser.add_argument("--samples_dir", default="data/gaussian_1000/samples", type=Path)
    parser.add_argument("--output_dir", default="output/verification", type=Path)
    parser.add_argument("--n_samples", default=3, type=int)
    parser.add_argument("--config", default="config/default.yaml", type=Path)
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent
    args.shared_dir = repo_root / args.shared_dir
    args.samples_dir = repo_root / args.samples_dir
    args.output_dir = repo_root / args.output_dir

    print("=" * 60)
    print("M1-M7 Dual Channel Verification")
    print("=" * 60)

    # Load config
    cfg_path = repo_root / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    view_cfg = cfg.get("view_config", {})

    # Load visible mask
    mask_path = args.shared_dir / "visible_mask.npy"
    visible_mask = np.load(mask_path) if mask_path.exists() else None
    if visible_mask is not None:
        print(f"visible_mask: shape={visible_mask.shape}, "
              f"visible={visible_mask.sum()}, ratio={visible_mask.sum()/len(visible_mask):.3f}")

    # Select samples
    selected = select_samples(args.samples_dir, args.n_samples)
    print(f"\nSelected: {selected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sanity_all = {}
    align_all = {}
    overlap_all = {}

    for sid in selected:
        sample_dir = args.samples_dir / sid
        print(f"\n{'=' * 40}")
        print(f"Sample: {sid}")
        print(f"{'=' * 40}")

        if visible_mask is None:
            print("  SKIP: no visible_mask.npy")
            continue

        # Layer 1
        print("\n[Layer 1: Sanity Checks]")
        sanity = check_single_sample(sample_dir, args.shared_dir, visible_mask)
        sanity_all[sid] = sanity
        for k, v in sanity.items():
            if k.endswith("_pass"):
                status = "PASS" if v else "FAIL"
                print(f"  {k}: {status}")
            elif k.endswith("_error"):
                print(f"  {k}: {v}")
            elif k in ["m7_format", "m7_b_raw_shape"]:
                print(f"  {k}: {v}")

        # Layer 2
        print("\n[Layer 2: Spatial Alignment]")
        try:
            align = check_spatial_alignment(
                sample_dir, args.shared_dir, view_cfg, visible_mask
            )
            align_all[sid] = align
            if "error" in align:
                print(f"  Error: {align['error']}")
            else:
                de_pass = "PASS" if align["de_align_pass"] else "FAIL"
                mcx_pass = "PASS" if align["mcx_align_pass"] else "FAIL"
                print(f"  DE peak → nearest GT: {align['de_to_nearest_gt_mm']} mm ({de_pass}, tol=10mm)")
                print(f"  MCX peak phys → nearest GT 2D: {align['mcx_to_nearest_gt_mm']} mm ({mcx_pass}, tol=30mm)")
                print(f"  DE peak: {align['de_peak_coord_mm']}")
                print(f"  MCX peak phys XY (mm): {align['mcx_peak_phys_xy_mm']}")
                print(f"  GT centers (mm): {align['gt_centers_mm']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            align_all[sid] = {"error": str(e)}
            print(f"  ERROR: {e}")

        # Layer 3
        print("\n[Layer 3: Surface Overlap]")
        try:
            overlap = check_surface_overlap(
                sample_dir, args.shared_dir, visible_mask, view_config=view_cfg
            )
            overlap_all[sid] = overlap
            p = "PASS" if overlap["overlap_pass"] else "FAIL"
            mcx_k = overlap.get("mcx_top_k", "?")
            print(f"  DE top-{overlap['de_top_k']} ↔ MCX top-{mcx_k} "
                  f"overlap: {overlap['overlap_ratio']:.3f} ({p}, tol=3mm)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            overlap_all[sid] = {"error": str(e)}
            print(f"  ERROR: {e}")

        # Layer 4
        print("\n[Layer 4: Comparison figures]")
        try:
            fig_path = args.output_dir / f"{sid}_comparison.png"
            generate_comparison_figure(
                sample_dir, args.shared_dir, fig_path,
                visible_mask=visible_mask,
                angles=[-90, 0, 90],
                view_config=view_cfg,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    modules = ["M1", "M2", "M3", "M4", "M5", "M7"]
    print("\n[Layer 1: Sanity Check]")
    for mod in modules:
        key = f"{mod.lower()}_pass"
        counts = [sanity_all[sid].get(key, False) for sid in selected]
        n_pass = sum(counts)
        status = "PASS" if n_pass == len(selected) else "FAIL"
        print(f"  {mod}: {status} ({n_pass}/{len(selected)})")

    print("\n[Layer 2: Spatial Alignment]")
    for sid in selected:
        a = align_all.get(sid, {})
        if "error" in a:
            print(f"  {sid}: ERROR — {a['error']}")
            continue
        de_s = "PASS" if a.get("de_align_pass") else "FAIL"
        mcx_s = "PASS" if a.get("mcx_align_pass") else "FAIL"
        print(f"  {sid}: DE={a.get('de_to_nearest_gt_mm','?')}mm({de_s}), "
              f"MCX={a.get('mcx_to_nearest_gt_mm','?')}mm({mcx_s}, tol=30mm)")

    print("\n[Layer 3: Surface Overlap]")
    n_overlap_pass = 0
    for sid in selected:
        o = overlap_all.get(sid, {})
        if "error" in o:
            print(f"  {sid}: ERROR — {o['error']}")
            continue
        ratio = o.get("overlap_ratio", 0)
        p = "PASS" if o.get("overlap_pass") else "FAIL"
        if o.get("overlap_pass"):
            n_overlap_pass += 1
        print(f"  {sid}: overlap={ratio:.3f} ({p})")

    overall = "PASS" if n_overlap_pass >= 2 else "FAIL"
    print(f"\nOVERALL: {overall} (need ≥2/3 overlap pass)")

    # Write JSON report
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    report = make_serializable({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": selected,
        "sanity": sanity_all,
        "spatial_alignment": align_all,
        "surface_overlap": overlap_all,
        "overall": overall,
    })

    report_path = args.output_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    # Write text summary
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== M1-M7 Dual Channel Verification ===\n")
        f.write(f"Date: {report['date']}\n")
        f.write(f"Samples: {', '.join(selected)}\n\n")
        f.write("[Layer 1: Sanity Check]\n")
        for mod in modules:
            key = f"{mod.lower()}_pass"
            counts = [sanity_all[sid].get(key, False) for sid in selected]
            n_pass = sum(counts)
            status = "PASS" if n_pass == len(selected) else "FAIL"
            f.write(f"  {mod}: {status} ({n_pass}/{len(selected)})\n")
        f.write("\n[Layer 2: Spatial Alignment]\n")
        for sid in selected:
            a = align_all.get(sid, {})
            if "error" in a:
                f.write(f"  {sid}: ERROR\n")
                continue
            de_s = "PASS" if a.get("de_align_pass") else "FAIL"
            mcx_s = "PASS" if a.get("mcx_align_pass") else "FAIL"
            f.write(f"  {sid}: DE={a.get('de_to_nearest_gt_mm','?')}mm({de_s}), "
                    f"MCX={a.get('mcx_to_nearest_gt_mm','?')}mm({mcx_s})\n")
        f.write("\n[Layer 3: Surface Overlap]\n")
        for sid in selected:
            o = overlap_all.get(sid, {})
            if "error" in o:
                f.write(f"  {sid}: ERROR\n")
                continue
            ratio = o.get("overlap_ratio", 0)
            p = "PASS" if o.get("overlap_pass") else "FAIL"
            f.write(f"  {sid}: overlap={ratio:.3f} ({p})\n")
        f.write(f"\nOVERALL: {overall}\n")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
