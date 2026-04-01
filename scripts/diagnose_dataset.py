#!/usr/bin/env python3
"""
Dataset Diagnosis Script

Analyzes the 50-sample FMT dataset for two issues:
1. Multi-focus samples have foci too far apart
2. Surface signals are too sparse

Usage:
    python scripts/diagnose_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json


def load_mesh():
    """Load mesh data."""
    mesh_paths = ["assets/mesh/mesh.npz", "output/shared/mesh.npz"]
    for path in mesh_paths:
        if Path(path).exists():
            mesh_data = np.load(path, allow_pickle=True)
            print(f"  Mesh loaded from: {path}")
            return {
                "nodes": mesh_data["nodes"],
                "elements": mesh_data["elements"],
            }
    raise FileNotFoundError("No mesh file found")


def diagnose_focus_distances():
    """Diagnose inter-focus distances for multi-focus samples."""
    print("=" * 70)
    print("1. INTER-FOCUS DISTANCE STATISTICS (Multi-focus only)")
    print("=" * 70)

    sample_dirs = sorted(Path("data").glob("sample_*"))
    all_distances = []
    multi_foci_samples = 0

    for sd in sample_dirs:
        with open(sd / "tumor_params.json") as f:
            tp = json.load(f)

        if tp["num_foci"] < 2:
            continue

        multi_foci_samples += 1
        centers = [focus["center"] for focus in tp["foci"]]

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                all_distances.append(dist)

    all_distances = np.array(all_distances)

    print(f"Multi-focus samples: {multi_foci_samples}")
    print(f"Total focus pairs: {len(all_distances)}")
    print(f"Distance (mm):")
    print(f"  min:  {all_distances.min():.2f}")
    print(f"  max:  {all_distances.max():.2f}")
    print(f"  mean: {all_distances.mean():.2f}")
    print(f"  std:  {all_distances.std():.2f}")
    print(f"  P25:  {np.percentile(all_distances, 25):.2f}")
    print(f"  P50:  {np.percentile(all_distances, 50):.2f}")
    print(f"  P75:  {np.percentile(all_distances, 75):.2f}")
    print()
    print(f"Pairs with distance < 5mm:  {np.sum(all_distances < 5)}")
    print(f"Pairs with distance < 10mm: {np.sum(all_distances < 10)}")
    print(f"Pairs with distance < 15mm: {np.sum(all_distances < 15)}")
    print(f"Pairs with distance < 20mm: {np.sum(all_distances < 20)}")

    return all_distances


def get_base_radius(focus):
    """Get base radius from focus params (handles sphere and ellipsoid)."""
    if "radius" in focus["params"]:
        return focus["params"]["radius"]
    elif "ry" in focus["params"]:
        return focus["params"]["ry"]
    else:
        return 0.5


def diagnose_node_sampling(mesh_nodes):
    """Diagnose node-level GT sampling with 3σ and 4σ cutoffs."""
    print()
    print("=" * 70)
    print("2. NODE-LEVEL GT SAMPLING DIAGNOSTIC (Gaussian, sigma=radius)")
    print("=" * 70)

    sample_dirs = sorted(Path("data").glob("sample_*"))

    total_foci = 0
    zero_node_3sigma_count = 0
    zero_node_4sigma_count = 0
    nodes_per_focus_3sigma = []
    nodes_per_focus_4sigma = []

    for sd in sample_dirs:
        with open(sd / "tumor_params.json") as f:
            tp = json.load(f)

        for focus in tp["foci"]:
            total_foci += 1
            center = np.array(focus["center"])
            base_radius = get_base_radius(focus)
            sigma = base_radius
            cutoff_3sigma = 3.0 * sigma
            cutoff_4sigma = 4.0 * sigma

            dists = np.linalg.norm(mesh_nodes - center, axis=1)
            nodes_3sigma = np.sum(dists <= cutoff_3sigma)
            nodes_4sigma = np.sum(dists <= cutoff_4sigma)

            nodes_per_focus_3sigma.append(nodes_3sigma)
            nodes_per_focus_4sigma.append(nodes_4sigma)

            if nodes_3sigma == 0:
                zero_node_3sigma_count += 1
            if nodes_4sigma == 0:
                zero_node_4sigma_count += 1

    nodes_per_focus_3sigma = np.array(nodes_per_focus_3sigma)
    nodes_per_focus_4sigma = np.array(nodes_per_focus_4sigma)

    print(f"Total foci analyzed: {total_foci}")
    print()
    print(f"At 3σ cutoff (Gaussian truncated at 3σ):")
    print(
        f"  Foci with 0 nodes: {zero_node_3sigma_count} ({zero_node_3sigma_count / total_foci * 100:.1f}%)"
    )
    print(
        f"  Nodes per focus: min={nodes_per_focus_3sigma.min()}, max={nodes_per_focus_3sigma.max()}, mean={nodes_per_focus_3sigma.mean():.1f}"
    )
    print()
    print(f"At 4σ cutoff (Gaussian truncated at 4σ):")
    print(
        f"  Foci with 0 nodes: {zero_node_4sigma_count} ({zero_node_4sigma_count / total_foci * 100:.1f}%)"
    )
    print(
        f"  Nodes per focus: min={nodes_per_focus_4sigma.min()}, max={nodes_per_focus_4sigma.max()}, mean={nodes_per_focus_4sigma.mean():.1f}"
    )

    return nodes_per_focus_4sigma


def diagnose_surface_signals():
    """Diagnose surface signal characteristics."""
    print()
    print("=" * 70)
    print("3. SURFACE SIGNAL DIAGNOSTIC")
    print("=" * 70)

    sample_dirs = sorted(Path("data").glob("sample_*"))
    signals = []

    for sd in sample_dirs:
        meas_b = np.load(sd / "measurement_b.npy")
        with open(sd / "tumor_params.json") as f:
            tp = json.load(f)

        nonzero_count = np.count_nonzero(meas_b)
        signals.append(
            {
                "sample": sd.name,
                "nonzero": nonzero_count,
                "max": meas_b.max(),
                "sum": meas_b.sum(),
                "nonzero_ratio": nonzero_count / len(meas_b),
                "num_foci": tp["num_foci"],
                "centers": [focus["center"] for focus in tp["foci"]],
                "radii": [get_base_radius(focus) for focus in tp["foci"]],
            }
        )

    signals_sorted = sorted(signals, key=lambda x: x["sum"])

    print(f"All samples ({len(signals)}):")
    print(f"{'Sample':<15} {'Nonzero':>10} {'Max':>8} {'Sum':>12} {'Nonzero%':>10}")
    print("-" * 60)
    for s in signals_sorted[:5]:
        print(
            f"{s['sample']:<15} {s['nonzero']:>10} {s['max']:>8.4f} {s['sum']:>12.6f} {s['nonzero_ratio'] * 100:>9.1f}%"
        )
    print("  ...")
    for s in signals_sorted[-5:]:
        print(
            f"{s['sample']:<15} {s['nonzero']:>10} {s['max']:>8.4f} {s['sum']:>12.6f} {s['nonzero_ratio'] * 100:>9.1f}%"
        )

    print()
    print("=" * 70)
    print("LOWEST 5 SUM SAMPLES - DETAILED")
    print("=" * 70)
    for s in signals_sorted[:5]:
        print(f"\n{s['sample']}:")
        print(f"  sum={s['sum']:.6f}, max={s['max']:.4f}, nonzero={s['nonzero']}")
        for i, (c, r) in enumerate(zip(s["centers"], s["radii"])):
            print(
                f"  Focus {i}: center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), radius={r:.2f}"
            )

    print()
    print("=" * 70)
    print("HIGHEST 5 SUM SAMPLES - DETAILED")
    print("=" * 70)
    for s in signals_sorted[-5:]:
        print(f"\n{s['sample']}:")
        print(f"  sum={s['sum']:.6f}, max={s['max']:.4f}, nonzero={s['nonzero']}")
        for i, (c, r) in enumerate(zip(s["centers"], s["radii"])):
            print(
                f"  Focus {i}: center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), radius={r:.2f}"
            )


def diagnose_mesh_spacing(mesh_nodes, mesh_elements):
    """Diagnose mesh element edge lengths."""
    print()
    print("=" * 70)
    print("4. MESH NODE SPACING STATISTICS")
    print("=" * 70)

    elem = mesh_elements[:, :4]
    edge_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    edge_lengths = []
    for e in elem:
        for i, j in edge_indices:
            dist = np.linalg.norm(mesh_nodes[e[i]] - mesh_nodes[e[j]])
            edge_lengths.append(dist)

    edge_lengths = np.array(edge_lengths)

    print(f"Total edges analyzed: {len(edge_lengths)}")
    print(f"Edge length (mm):")
    print(f"  min:   {edge_lengths.min():.4f}")
    print(f"  max:   {edge_lengths.max():.4f}")
    print(f"  mean:  {edge_lengths.mean():.4f}")
    print(f"  median: {np.median(edge_lengths):.4f}")
    print(f"  std:   {edge_lengths.std():.4f}")

    print()
    print("Tumor placement region analysis:")
    print("  Trunk region: Y in [30, 70], Dorsal Z>15 or Lateral X<8 or X>28")

    trunk_mask = (mesh_nodes[:, 1] >= 30) & (mesh_nodes[:, 1] <= 70)
    dorsal_mask = mesh_nodes[:, 2] > 15
    lateral_mask = (mesh_nodes[:, 0] < 8) | (mesh_nodes[:, 0] > 28)

    tumor_region_mask = trunk_mask & (dorsal_mask | lateral_mask)
    tumor_nodes = mesh_nodes[tumor_region_mask]

    print(
        f"\n  Nodes in tumor region: {len(tumor_nodes)} ({len(tumor_nodes) / len(mesh_nodes) * 100:.1f}%)"
    )

    if len(tumor_nodes) > 1:
        from scipy.spatial import KDTree

        tree = KDTree(tumor_nodes)
        distances, _ = tree.query(tumor_nodes, k=2)
        nn_distances = distances[:, 1]

        print(f"  Nearest-neighbor distances in tumor region:")
        print(f"    mean:  {nn_distances.mean():.4f}")
        print(f"    median: {np.median(nn_distances):.4f}")
        print(f"    min:   {nn_distances.min():.4f}")
        print(f"    max:   {nn_distances.max():.4f}")


def main():
    print("=" * 70)
    print("FMT-SimGen DATASET DIAGNOSIS")
    print("=" * 70)

    print("\nLoading mesh data...")
    mesh = load_mesh()
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    print(f"  Mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")

    diagnose_focus_distances()
    diagnose_node_sampling(nodes)
    diagnose_surface_signals()
    diagnose_mesh_spacing(nodes, elements)

    print()
    print("=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
