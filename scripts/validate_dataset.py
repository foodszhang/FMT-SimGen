#!/usr/bin/env python3
"""
Validate FMT-SimGen dataset: integrity, quality statistics, and visualization.

Usage:
    python scripts/validate_dataset.py \
        --data_dir data/gaussian_1000 \
        --shared_dir output/shared

    python scripts/validate_dataset.py \
        --data_dir data/uniform_1000 \
        --shared_dir output/shared

    # Skip 3D rendering (faster)
    python scripts/validate_dataset.py \
        --data_dir data/gaussian_1000 \
        --shared_dir output/shared \
        --skip_3d
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Matplotlib (lazy import in functions for agg backend)
# Setup matplotlib for non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Validate FMT-SimGen dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., data/gaussian_1000)",
    )
    parser.add_argument(
        "--shared_dir",
        type=str,
        default="output/shared",
        help="Path to shared mesh directory (default: output/shared)",
    )
    parser.add_argument(
        "--skip_3d",
        action="store_true",
        help="Skip 3D rendering (faster validation)",
    )
    parser.add_argument(
        "--skip_uniform_check",
        action="store_true",
        help="Skip uniform source binary-value check",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Part 1: Integrity Check
# ---------------------------------------------------------------------------

def check_integrity(data_dir: Path, shared_dir: Path) -> dict[str, Any]:
    """Run integrity checks and return results dict."""
    results: dict[str, Any] = {
        "passed": True,
        "checks": [],
        "errors": [],
    }

    samples_dir = data_dir / "samples"
    splits_dir = data_dir / "splits"
    manifest_path = data_dir / "dataset_manifest.json"

    # 1. Check manifest exists
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            num_declared = manifest.get("num_samples", "UNKNOWN")
            results["checks"].append(f"[✓] Manifest exists ({num_declared} samples declared)")
        except json.JSONDecodeError as e:
            results["checks"].append(f"[✗] Manifest parse error: {e}")
            results["passed"] = False
    else:
        results["checks"].append("[✗] Manifest not found")
        results["passed"] = False
        return results

    # 2. Check samples directory
    if not samples_dir.exists():
        results["checks"].append("[✗] samples/ directory not found")
        results["passed"] = False
        return results

    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    results["checks"].append(f"[✓] Found {len(sample_dirs)} sample directories")

    if num_declared != "UNKNOWN" and len(sample_dirs) != num_declared:
        results["checks"].append(f"[✗] Sample count mismatch: {len(sample_dirs)} found vs {num_declared} declared")
        results["passed"] = False

    # 3. Check each sample has required files
    required_files = ["measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy", "tumor_params.json"]
    missing_files: list[str] = []
    bad_shapes: list[str] = []
    bad_json: list[str] = []

    # Expected shapes from manifest or defaults
    n_nodes = manifest.get("mesh_nodes", 11236)
    n_surface = manifest.get("mesh_surface_nodes", 7465)
    # gt_voxels shape might not be in manifest, use a sample to determine
    gt_voxels_shape = None

    for sample_dir in sample_dirs:
        sid = sample_dir.name
        for fname in required_files:
            fpath = sample_dir / fname
            if not fpath.exists():
                missing_files.append(f"{sid}: {fname} missing")
                continue

        # Check tumor_params.json parseability
        try:
            with open(sample_dir / "tumor_params.json") as f:
                params = json.load(f)
        except json.JSONDecodeError:
            bad_json.append(sid)

        # Check shapes on first sample with valid files
        if gt_voxels_shape is None and (sample_dir / "gt_voxels.npy").exists():
            try:
                voxels = np.load(sample_dir / "gt_voxels.npy")
                gt_voxels_shape = tuple(voxels.shape)
            except Exception:
                pass

    if missing_files:
        results["checks"].append(f"[✗] Missing files: {len(missing_files)}")
        for m in missing_files[:5]:
            results["checks"].append(f"      {m}")
        if len(missing_files) > 5:
            results["checks"].append(f"      ... and {len(missing_files) - 5} more")
        results["passed"] = False
    else:
        results["checks"].append("[✓] All samples have 4 required files")

    if bad_json:
        results["checks"].append(f"[✗] {len(bad_json)} samples have invalid JSON: {bad_json[:3]}")
        results["passed"] = False

    # 4. Check split files
    train_path = splits_dir / "train.txt"
    val_path = splits_dir / "val.txt"

    if train_path.exists() and val_path.exists():
        with open(train_path) as f:
            train_samples = [line.strip() for line in f if line.strip()]
        with open(val_path) as f:
            val_samples = [line.strip() for line in f if line.strip()]

        train_set = set(train_samples)
        val_set = set(val_samples)
        overlap = train_set & val_set

        results["checks"].append(f"[✓] Splits: {len(train_samples)} train + {len(val_samples)} val = {len(train_samples) + len(val_samples)} total")

        if overlap:
            results["checks"].append(f"[✗] Train/val overlap: {len(overlap)} samples")
            results["passed"] = False

        # Check all split IDs are in samples/
        sample_ids = {d.name for d in sample_dirs}
        train_not_found = train_set - sample_ids
        val_not_found = val_set - sample_ids

        if train_not_found:
            results["checks"].append(f"[✗] {len(train_not_found)} train IDs not in samples/: {list(train_not_found)[:3]}")
            results["passed"] = False
        if val_not_found:
            results["checks"].append(f"[✗] {len(val_not_found)} val IDs not in samples/: {list(val_not_found)[:3]}")
            results["passed"] = False
    else:
        results["checks"].append("[✗] Split files not found")
        results["passed"] = False

    # 5. Check shared assets
    if shared_dir.exists():
        mesh_path = shared_dir / "mesh.npz"
        if mesh_path.exists():
            try:
                mesh = np.load(mesh_path)
                mesh_nodes = mesh["nodes"]
                results["checks"].append(f"[✓] Shared mesh: {mesh_nodes.shape[0]} nodes")
            except Exception as e:
                results["checks"].append(f"[✗] Shared mesh load error: {e}")
                results["passed"] = False
        else:
            results["checks"].append("[✗] mesh.npz not found in shared_dir")
            results["passed"] = False
    else:
        results["checks"].append(f"[✗] Shared dir not found: {shared_dir}")
        results["passed"] = False

    return results


# ---------------------------------------------------------------------------
# Part 2: Quality Statistics
# ---------------------------------------------------------------------------

def compute_quality_stats(data_dir: Path, shared_dir: Path) -> tuple[list[dict], dict]:
    """
    Compute per-sample quality metrics and return list of stats + anomaly summary.
    """
    samples_dir = data_dir / "samples"
    manifest_path = data_dir / "dataset_manifest.json"

    # Load manifest for source_type
    source_type = "gaussian"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        source_type = manifest.get("source_type", "gaussian")

    # Load mesh for surface faces (for 3D viz)
    mesh_nodes = None
    surface_faces = None
    if shared_dir.exists():
        mesh_path = shared_dir / "mesh.npz"
        if mesh_path.exists():
            mesh = np.load(mesh_path)
            mesh_nodes = mesh["nodes"]
            surface_faces = mesh["surface_faces"]

    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    all_stats: list[dict] = []
    anomaly_counts = {
        "INVALID": [],   # gt_max < 0.01
        "WEAK": [],      # gt_max < 0.1
        "NO_SIGNAL": [], # b_max < 1e-6
        "SPARSE": [],    # gt_nonzero_count < 10
    }

    for sample_dir in sample_dirs:
        sid = sample_dir.name

        try:
            b = np.load(sample_dir / "measurement_b.npy")
            gt_nodes = np.load(sample_dir / "gt_nodes.npy")
            gt_voxels = np.load(sample_dir / "gt_voxels.npy")
            with open(sample_dir / "tumor_params.json") as f:
                params = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {sid}: {e}")
            continue

        # Per-sample metrics
        b_max = float(np.max(np.abs(b)))
        b_mean = float(np.mean(np.abs(b)))
        b_nonzero_frac = float(np.count_nonzero(b) / len(b))

        gt_max = float(np.max(gt_nodes))
        gt_mean = float(np.mean(gt_nodes))
        gt_nonzero_count = int(np.count_nonzero(gt_nodes))
        gt_nonzero_frac = float(gt_nonzero_count / len(gt_nodes))

        gt_voxels_nonzero_frac = float(np.count_nonzero(gt_voxels) / gt_voxels.size)

        num_foci = params.get("num_foci", len(params.get("foci", [])))
        depth_tier = params.get("depth_tier")  # may be None for older samples
        source_type_sample = params.get("source_type", source_type)

        # Extract foci info
        foci_centers = []
        foci_radii = []
        for focus in params.get("foci", []):
            c = focus.get("center", [0, 0, 0])
            foci_centers.append(c)
            # radius for sphere, mean of rx/ry/rz for ellipsoid
            if focus.get("shape") == "sphere":
                r = focus.get("radius", 0)
            else:
                rx = focus.get("rx") or focus.get("params", {}).get("rx", 0)
                ry = focus.get("ry") or focus.get("params", {}).get("ry", 0)
                rz = focus.get("rz") or focus.get("params", {}).get("rz", 0)
                r = (rx + ry + rz) / 3 if all(v is not None for v in [rx, ry, rz]) else 0
            foci_radii.append(r)

        # Determine depth_tier if missing (estimate from depth_mm)
        if depth_tier is None:
            depth_mm = params.get("depth_mm")
            if depth_mm is not None:
                if depth_mm < 3.5:
                    depth_tier = "shallow"
                elif depth_mm < 6.0:
                    depth_tier = "medium"
                else:
                    depth_tier = "deep"
            else:
                depth_tier = "unknown"

        # Anomaly detection
        flags: list[str] = []
        if gt_max < 0.01:
            flags.append("INVALID")
            anomaly_counts["INVALID"].append(sid)
        elif gt_max < 0.1:
            flags.append("WEAK")
            anomaly_counts["WEAK"].append(sid)
        if b_max < 1e-6:
            flags.append("NO_SIGNAL")
            anomaly_counts["NO_SIGNAL"].append(sid)
        if gt_nonzero_count < 10:
            flags.append("SPARSE")
            anomaly_counts["SPARSE"].append(sid)

        stat = {
            "sample_id": sid,
            "num_foci": num_foci,
            "depth_tier": depth_tier,
            "source_type": source_type_sample,
            "gt_max": gt_max,
            "gt_mean": gt_mean,
            "gt_nonzero_count": gt_nonzero_count,
            "gt_nonzero_frac": gt_nonzero_frac,
            "gt_voxels_nonzero_frac": gt_voxels_nonzero_frac,
            "b_max": b_max,
            "b_mean": b_mean,
            "b_nonzero_frac": b_nonzero_frac,
            "anomaly_flags": ",".join(flags) if flags else "",
            "foci_centers": foci_centers,
            "foci_radii": foci_radii,
        }
        all_stats.append(stat)

    # Group statistics
    group_stats = compute_group_stats(all_stats)

    # Anomaly summary
    anomaly_summary = {
        "INVALID (gt_max < 0.01)": anomaly_counts["INVALID"],
        "WEAK (gt_max < 0.1)": anomaly_counts["WEAK"],
        "NO_SIGNAL (b_max < 1e-6)": anomaly_counts["NO_SIGNAL"],
        "SPARSE (gt_nonzero_count < 10)": anomaly_counts["SPARSE"],
    }

    return all_stats, group_stats, anomaly_summary


def compute_group_stats(stats: list[dict]) -> dict:
    """Compute grouped statistics by num_foci, depth_tier, and cross-group."""
    groups: dict = {}

    # By num_foci
    for nf in [1, 2, 3]:
        sub = [s for s in stats if s["num_foci"] == nf]
        if sub:
            groups[f"num_foci_{nf}"] = {
                "count": len(sub),
                "gt_max_mean": np.mean([s["gt_max"] for s in sub]),
                "gt_max_std": np.std([s["gt_max"] for s in sub]),
                "gt_max_min": np.min([s["gt_max"] for s in sub]),
                "gt_max_max": np.max([s["gt_max"] for s in sub]),
                "b_max_mean": np.mean([s["b_max"] for s in sub]),
                "b_max_std": np.std([s["b_max"] for s in sub]),
                "gt_nonzero_frac_mean": np.mean([s["gt_nonzero_frac"] for s in sub]),
                "gt_nonzero_frac_std": np.std([s["gt_nonzero_frac"] for s in sub]),
            }

    # By depth_tier
    for tier in ["shallow", "medium", "deep", "unknown"]:
        sub = [s for s in stats if s["depth_tier"] == tier]
        if sub:
            groups[f"depth_{tier}"] = {
                "count": len(sub),
                "gt_max_mean": np.mean([s["gt_max"] for s in sub]),
                "gt_max_std": np.std([s["gt_max"] for s in sub]),
                "b_max_mean": np.mean([s["b_max"] for s in sub]),
                "b_max_std": np.std([s["b_max"] for s in sub]),
            }

    # Cross-group: num_foci × depth_tier
    for nf in [1, 2, 3]:
        for tier in ["shallow", "medium", "deep", "unknown"]:
            sub = [s for s in stats if s["num_foci"] == nf and s["depth_tier"] == tier]
            if sub:
                key = f"cross_{nf}f_{tier}"
                groups[key] = {
                    "count": len(sub),
                    "gt_max_mean": np.mean([s["gt_max"] for s in sub]),
                    "gt_max_std": np.std([s["gt_max"] for s in sub]),
                    "b_max_mean": np.mean([s["b_max"] for s in sub]),
                    "b_max_std": np.std([s["b_max"] for s in sub]),
                }

    return groups


def print_quality_report(all_stats: list[dict], group_stats: dict, anomaly_summary: dict):
    """Print quality statistics to console."""
    print("\n" + "=" * 60)
    print("=== Quality Statistics ===")
    print(f"Total samples: {len(all_stats)}")

    # By num_foci
    print("\n── By num_foci ──")
    for nf in [1, 2, 3]:
        key = f"num_foci_{nf}"
        if key in group_stats:
            g = group_stats[key]
            pct = g["count"] / len(all_stats) * 100
            print(f"  {nf}-foci: {g['count']:4d} ({pct:5.1f}%)  "
                  f"gt_max: {g['gt_max_mean']:.3f}±{g['gt_max_std']:.3f} "
                  f"[{g['gt_max_min']:.3f}, {g['gt_max_max']:.3f}]  "
                  f"b_max: {g['b_max_mean']:.3f}±{g['b_max_std']:.3f}")

    # By depth_tier
    print("\n── By depth_tier ──")
    for tier in ["shallow", "medium", "deep", "unknown"]:
        key = f"depth_{tier}"
        if key in group_stats:
            g = group_stats[key]
            pct = g["count"] / len(all_stats) * 100
            print(f"  {tier:7s}: {g['count']:4d} ({pct:5.1f}%)  "
                  f"gt_max: {g['gt_max_mean']:.3f}±{g['gt_max_std']:.3f}  "
                  f"b_max: {g['b_max_mean']:.3f}±{g['b_max_std']:.3f}")

    # Cross-group
    print("\n── Cross-group (foci × depth) ──")
    for nf in [1, 2, 3]:
        for tier in ["shallow", "medium", "deep"]:
            key = f"cross_{nf}f_{tier}"
            if key in group_stats:
                g = group_stats[key]
                if g["count"] > 0:
                    print(f"  {nf}-foci × {tier:7s}: {g['count']:4d} samples  "
                          f"gt_max: {g['gt_max_mean']:.3f}±{g['gt_max_std']:.3f}  "
                          f"b_max: {g['b_max_mean']:.3f}±{g['b_max_std']:.3f}")

    # Anomalies
    print("\n── Anomalies ──")
    any_anomalies = False
    for label, samples in anomaly_summary.items():
        if samples:
            any_anomalies = True
            print(f"  {label}: {len(samples)} samples → {', '.join(samples[:5])}"
                  f"{' ...' if len(samples) > 5 else ''}")

    if not any_anomalies:
        print("  (none)")

    # Quality filter summary
    usable = [s for s in all_stats if s["gt_max"] >= 0.1]
    strong = [s for s in all_stats if s["gt_max"] >= 0.3]
    print(f"\n── Quality Filter Summary ──")
    print(f"  Usable samples (gt_max >= 0.1): {len(usable)}/{len(all_stats)} ({len(usable)/len(all_stats)*100:.1f}%)")
    print(f"  Strong samples (gt_max >= 0.3): {len(strong)}/{len(all_stats)} ({len(strong)/len(all_stats)*100:.1f}%)")


def save_statistics_csv(stats: list[dict], output_path: Path):
    """Save per-sample statistics to CSV."""
    # Flatten for CSV (exclude complex nested fields)
    rows = []
    for s in stats:
        rows.append({
            "sample_id": s["sample_id"],
            "num_foci": s["num_foci"],
            "depth_tier": s["depth_tier"],
            "source_type": s["source_type"],
            "gt_max": s["gt_max"],
            "gt_mean": s["gt_mean"],
            "gt_nonzero_count": s["gt_nonzero_count"],
            "gt_nonzero_frac": s["gt_nonzero_frac"],
            "gt_voxels_nonzero_frac": s["gt_voxels_nonzero_frac"],
            "b_max": s["b_max"],
            "b_mean": s["b_mean"],
            "b_nonzero_frac": s["b_nonzero_frac"],
            "anomaly_flags": s["anomaly_flags"],
        })

    fieldnames = ["sample_id", "num_foci", "depth_tier", "source_type",
                  "gt_max", "gt_mean", "gt_nonzero_count", "gt_nonzero_frac",
                  "gt_voxels_nonzero_frac", "b_max", "b_mean", "b_nonzero_frac",
                  "anomaly_flags"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved sample statistics to {output_path}")


# ---------------------------------------------------------------------------
# Part 3: Visualization
# ---------------------------------------------------------------------------

def generate_figures(
    data_dir: Path,
    shared_dir: Path,
    all_stats: list[dict],
    group_stats: dict,
    skip_3d: bool = False,
    skip_uniform_check: bool = False,
):
    """Generate all visualization figures."""
    figures_dir = data_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Matplotlib style: academic paper
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    logger.info("Generating figure 1: dataset_overview.png")
    plot_dataset_overview(all_stats, figures_dir)

    logger.info("Generating figure 2: group_quality.png")
    plot_group_quality(all_stats, group_stats, figures_dir)

    if not skip_3d:
        logger.info("Generating figure 3: sample_examples.png (3D rendering)")
        plot_sample_examples(data_dir, shared_dir, all_stats, figures_dir)
    else:
        logger.info("Skipping 3D rendering (--skip_3d)")

    logger.info("Generating figure 4: voxel_slices.png")
    plot_voxel_slices(data_dir, all_stats, figures_dir)

    if not skip_uniform_check:
        # Check if this is a uniform dataset
        samples_dir = data_dir / "samples"
        if all_stats and all_stats[0].get("source_type") == "uniform":
            logger.info("Generating figure 5: uniform_binary_check.png")
            plot_uniform_binary_check(all_stats, samples_dir, figures_dir)
        else:
            logger.info("Skipping uniform binary check (not a uniform source dataset)")


def plot_dataset_overview(stats: list[dict], figures_dir: Path):
    """Figure 1: Dataset distribution overview (2×3 subplots)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Dataset Overview", fontweight="bold", fontsize=13)

    num_foci_vals = [s["num_foci"] for s in stats]
    depth_vals = [s["depth_tier"] for s in stats]
    gt_max_vals = [s["gt_max"] for s in stats]
    b_max_vals = [s["b_max"] for s in stats]
    gt_nonzero_frac_vals = [s["gt_nonzero_frac"] for s in stats]

    # (a) num_foci distribution
    ax = axes[0, 0]
    counts = [sum(1 for v in num_foci_vals if v == n) for n in [1, 2, 3]]
    bars = ax.bar(["1", "2", "3"], counts, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xlabel("Number of Foci")
    ax.set_ylabel("Count")
    ax.set_title("(a) num_foci Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9)

    # (b) depth_tier distribution
    ax = axes[0, 1]
    tiers = ["shallow", "medium", "deep", "unknown"]
    counts = [sum(1 for v in depth_vals if v == t) for t in tiers]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    bars = ax.bar(tiers, counts, color=colors)
    ax.set_xlabel("Depth Tier")
    ax.set_ylabel("Count")
    ax.set_title("(b) depth_tier Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) gt_max histogram
    ax = axes[0, 2]
    ax.hist(gt_max_vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1.2, label="threshold=0.1")
    ax.set_xlabel("gt_max")
    ax.set_ylabel("Count")
    ax.set_title("(c) gt_max Distribution")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) b_max histogram
    ax = axes[1, 0]
    ax.hist(b_max_vals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("b_max")
    ax.set_ylabel("Count")
    ax.set_title("(d) b_max Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (e) gt_nonzero_frac histogram
    ax = axes[1, 1]
    ax.hist(gt_nonzero_frac_vals, bins=40, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("gt_nonzero_frac")
    ax.set_ylabel("Count")
    ax.set_title("(e) gt_nonzero_frac Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (f) num_foci × depth_tier heatmap
    ax = axes[1, 2]
    tiers_clean = ["shallow", "medium", "deep"]
    heatmap_data = np.zeros((3, 3))
    for i, nf in enumerate([1, 2, 3]):
        for j, tier in enumerate(tiers_clean):
            heatmap_data[i, j] = sum(
                1 for s in stats if s["num_foci"] == nf and s["depth_tier"] == tier
            )
    im = ax.imshow(heatmap_data, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(tiers_clean)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["1", "2", "3"])
    ax.set_xlabel("Depth Tier")
    ax.set_ylabel("Number of Foci")
    ax.set_title("(f) foci × depth Heatmap")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Annotate cells
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(heatmap_data[i, j])}",
                    ha="center", va="center", color="black" if heatmap_data[i, j] < 200 else "white",
                    fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "dataset_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_quality(stats: list[dict], group_stats: dict, figures_dir: Path):
    """Figure 2: Group quality comparison (2×2 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("Group Quality Comparison", fontweight="bold", fontsize=13)

    # Prepare data
    data_by_foci = {nf: [s["gt_max"] for s in stats if s["num_foci"] == nf] for nf in [1, 2, 3]}
    data_by_depth = {tier: [s["gt_max"] for s in stats if s["depth_tier"] == tier]
                     for tier in ["shallow", "medium", "deep"]}
    data_b_by_depth = {tier: [s["b_max"] for s in stats if s["depth_tier"] == tier]
                       for tier in ["shallow", "medium", "deep"]}

    # (a) gt_max boxplot by num_foci
    ax = axes[0, 0]
    bp = ax.boxplot([data_by_foci[1], data_by_foci[2], data_by_foci[3]],
                    tick_labels=["1-foci", "2-foci", "3-foci"], patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Number of Foci")
    ax.set_ylabel("gt_max")
    ax.set_title("(a) gt_max by num_foci")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # (b) gt_max boxplot by depth_tier
    ax = axes[0, 1]
    depths = ["shallow", "medium", "deep"]
    valid_data = [data_by_depth[t] for t in depths]
    bp = ax.boxplot(valid_data, tick_labels=depths, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Depth Tier")
    ax.set_ylabel("gt_max")
    ax.set_title("(b) gt_max by depth_tier")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # (c) b_max boxplot by depth_tier
    ax = axes[1, 0]
    valid_data = [data_b_by_depth[t] for t in depths]
    bp = ax.boxplot(valid_data, tick_labels=depths, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Depth Tier")
    ax.set_ylabel("b_max")
    ax.set_title("(c) b_max by depth_tier")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # (d) gt_max vs b_max scatter colored by num_foci
    ax = axes[1, 1]
    colors_map = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}
    for nf in [1, 2, 3]:
        subset = [s for s in stats if s["num_foci"] == nf]
        ax.scatter([s["gt_max"] for s in subset], [s["b_max"] for s in subset],
                   c=colors_map[nf], label=f"{nf}-foci", alpha=0.5, s=15, edgecolors="none")
    ax.set_xlabel("gt_max")
    ax.set_ylabel("b_max")
    ax.set_title("(d) gt_max vs b_max")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "group_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_examples(
    data_dir: Path,
    shared_dir: Path,
    all_stats: list[dict],
    figures_dir: Path,
):
    """Figure 3: 6 representative samples with 3D rendering."""
    try:
        import pyvista as pv
        pv.set_plot_theme("document")
    except ImportError:
        logger.warning("PyVista not installed, skipping 3D visualization")
        return

    samples_dir = data_dir / "samples"

    # Load mesh
    mesh_path = shared_dir / "mesh.npz"
    if not mesh_path.exists():
        logger.warning(f"Mesh not found at {mesh_path}, skipping 3D visualization")
        return

    mesh_data = np.load(mesh_path)
    mesh_nodes = mesh_data["nodes"]
    surface_faces = mesh_data["surface_faces"]
    surface_node_indices = mesh_data["surface_node_indices"]

    # Build surface mesh for PyVista
    faces_pv = np.column_stack([
        np.full(len(surface_faces), 3), surface_faces
    ]).ravel()

    # Select 6 representative samples
    # Groups: 1f-shallow, 1f-deep, 2f-medium, 3f-shallow, 3f-medium, 3f-deep
    targets = [
        (1, "shallow"),
        (1, "deep"),
        (2, "medium"),
        (3, "shallow"),
        (3, "medium"),
        (3, "deep"),
    ]

    selected = []
    for nf, tier in targets:
        candidates = [s for s in all_stats if s["num_foci"] == nf and s["depth_tier"] == tier]
        if candidates:
            # Pick sample with median gt_max
            median_gt = sorted(candidates, key=lambda x: x["gt_max"])[len(candidates) // 2]
            selected.append(median_gt)
        else:
            # Fallback: any sample with this foci count
            fallback = [s for s in all_stats if s["num_foci"] == nf]
            if fallback:
                selected.append(fallback[0])

    n_rows = len(selected)
    n_cols = 2

    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        off_screen=True,
        window_size=(1600, 800 * n_rows // 2),
    )

    for row_idx, stat in enumerate(selected):
        sid = stat["sample_id"]
        sample_dir = samples_dir / sid

        try:
            gt_nodes = np.load(sample_dir / "gt_nodes.npy")
            b = np.load(sample_dir / "measurement_b.npy")
            with open(sample_dir / "tumor_params.json") as f:
                tumor_params = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {sid} for 3D viz: {e}")
            continue

        # Left subplot: GT on mesh
        plotter.subplot(row_idx, 0)

        # Semi-transparent background mesh
        background_mesh = pv.PolyData(mesh_nodes, faces_pv)
        plotter.add_mesh(
            background_mesh,
            color="lightgray",
            opacity=0.15,
            show_edges=False,
            reset_camera=False,
        )

        # Tumor region (gt > 0.05)
        tumor_mask = gt_nodes > 0.05
        if tumor_mask.any():
            tumor_cloud = pv.PolyData(mesh_nodes[tumor_mask])
            tumor_cloud.point_data["gt"] = gt_nodes[tumor_mask]
            plotter.add_mesh(
                tumor_cloud,
                scalars="gt",
                cmap="jet",
                point_size=4,
                render_points_as_spheres=True,
                clim=[0, max(gt_nodes.max(), 0.01)],
                reset_camera=False,
            )

        # Mark foci centers
        for focus in tumor_params.get("foci", []):
            center = np.array(focus["center"])
            sphere = pv.Sphere(radius=0.5, center=center)
            plotter.add_mesh(sphere, color="red", opacity=0.8, reset_camera=False)

        label = f"GT ({sid})\nmax={gt_nodes.max():.3f}"
        plotter.add_text(label, position="upper_left", font_size=9)

        # Right subplot: Measurement on surface
        plotter.subplot(row_idx, 1)

        # Broadcast b to all mesh nodes (b only has values for surface nodes)
        b_full = np.zeros(len(mesh_nodes), dtype=b.dtype)
        b_full[surface_node_indices] = b

        surface_mesh = pv.PolyData(mesh_nodes, faces_pv)
        surface_mesh.point_data["b"] = b_full
        plotter.add_mesh(
            surface_mesh,
            scalars="b",
            cmap="hot",
            clim=[0, max(b.max(), 1e-6)],
            reset_camera=False,
        )
        plotter.add_text(
            f"Measurement\nmax={b.max():.4f}",
            position="upper_left",
            font_size=9,
        )

    plotter.link_views()
    plotter.screenshot(figures_dir / "sample_examples.png", return_img=False)
    plotter.close()


def plot_voxel_slices(data_dir: Path, all_stats: list[dict], figures_dir: Path):
    """Figure 4: gt_voxels orthogonal slices for 3 samples."""
    import matplotlib.pyplot as plt

    samples_dir = data_dir / "samples"

    # Select 3 samples: one per 1/2/3 foci
    selected = []
    for nf in [1, 2, 3]:
        candidates = [s for s in all_stats if s["num_foci"] == nf]
        if candidates:
            # Pick sample with highest gt_max for visibility
            best = sorted(candidates, key=lambda x: x["gt_max"], reverse=True)[0]
            selected.append(best)

    if not selected:
        logger.warning("No valid samples found for voxel slices")
        return

    n_samples = len(selected)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row_idx, stat in enumerate(selected):
        sid = stat["sample_id"]
        sample_dir = samples_dir / sid

        try:
            gt_voxels = np.load(sample_dir / "gt_voxels.npy")
            with open(sample_dir / "tumor_params.json") as f:
                tumor_params = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {sid} for voxel slices: {e}")
            continue

        # Determine voxel spacing/offset from config or assume default
        # Use config from manifest
        voxel_spacing = 0.2  # mm
        offset = np.array([0.0, 0.0, 0.0])  # will be overridden if available

        # Get first focus center
        foci = tumor_params.get("foci", [])
        if not foci:
            continue
        center = np.array(foci[0].get("center", [0, 0, 0]))

        # Voxel coordinates of center
        voxel_center = (center / voxel_spacing).astype(int)
        # Clamp to valid range
        voxel_center = np.clip(voxel_center, [0, 0, 0], np.array(gt_voxels.shape) - 1)

        # Axial (Z slice)
        ax = axes[row_idx, 0]
        sl = gt_voxels[:, :, voxel_center[2]]
        im = ax.imshow(sl.T, cmap="jet", origin="lower", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{sid} Axial (Z={voxel_center[2]})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Coronal (Y slice)
        ax = axes[row_idx, 1]
        sl = gt_voxels[:, voxel_center[1], :]
        im = ax.imshow(sl.T, cmap="jet", origin="lower", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{sid} Coronal (Y={voxel_center[1]})")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Sagittal (X slice)
        ax = axes[row_idx, 2]
        sl = gt_voxels[voxel_center[0], :, :]
        im = ax.imshow(sl.T, cmap="jet", origin="lower", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{sid} Sagittal (X={voxel_center[0]})")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("gt_voxels Orthogonal Slices", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(figures_dir / "voxel_slices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uniform_binary_check(stats: list[dict], samples_dir: Path, figures_dir: Path):
    """Figure 5: Uniform source GT value distribution (should be binary 0/1)."""
    import matplotlib.pyplot as plt

    all_gt_values: list[float] = []
    for stat in stats:
        sid = stat["sample_id"]
        sample_dir = samples_dir / sid
        try:
            gt_nodes = np.load(sample_dir / "gt_nodes.npy")
            # Only non-zero values
            nonzero = gt_nodes[gt_nodes > 0]
            all_gt_values.extend(nonzero.tolist())
        except Exception:
            continue

    if not all_gt_values:
        logger.warning("No gt_nodes data found for uniform check")
        return

    all_gt_values = np.array(all_gt_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_gt_values, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.2, label="threshold=0.5")
    ax.set_xlabel("GT Node Value")
    ax.set_ylabel("Count")
    ax.set_title("Uniform Source: GT Value Distribution (nonzero only)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Check for intermediate values
    intermediate = np.sum((all_gt_values > 0.01) & (all_gt_values < 0.99))
    if intermediate > 0:
        ax.text(
            0.5, 0.95,
            f"WARNING: {intermediate} intermediate values detected!",
            transform=ax.transAxes,
            color="red",
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

    plt.tight_layout()
    fig.savefig(figures_dir / "uniform_binary_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    shared_dir = Path(args.shared_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("=== Integrity Check ===")
    print("=" * 60)

    integrity = check_integrity(data_dir, shared_dir)
    for check in integrity["checks"]:
        print(check)

    if not integrity["passed"]:
        print("\n[✗] Integrity check FAILED. Fix errors before proceeding.")
        sys.exit(1)

    print("\n[✓] Integrity check PASSED")

    # Part 2: Quality Statistics
    print("\n" + "=" * 60)
    print("=== Computing Quality Statistics ===")
    print("=" * 60)

    all_stats, group_stats, anomaly_summary = compute_quality_stats(data_dir, shared_dir)
    print_quality_report(all_stats, group_stats, anomaly_summary)

    # Save CSV
    csv_path = data_dir / "sample_statistics.csv"
    save_statistics_csv(all_stats, csv_path)

    # Part 3: Visualization
    print("\n" + "=" * 60)
    print("=== Generating Figures ===")
    print("=" * 60)

    generate_figures(
        data_dir,
        shared_dir,
        all_stats,
        group_stats,
        skip_3d=args.skip_3d,
        skip_uniform_check=args.skip_uniform_check,
    )

    print("\n" + "=" * 60)
    print("=== Validation Complete ===")
    print(f"CSV: {csv_path}")
    print(f"Figures: {data_dir / 'figures'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
