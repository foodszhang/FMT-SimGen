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
    mesh_data = np.load("output/shared/mesh.npz", allow_pickle=True)
    return {
        "nodes": mesh_data["nodes"],
        "elements": mesh_data["elements"],
    }


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


def diagnose_node_sampling(mesh_nodes):
    """Diagnose node-level GT sampling."""
    print()
    print("=" * 70)
    print("2. NODE-LEVEL GT SAMPLING DIAGNOSTIC")
    print("=" * 70)

    sample_dirs = sorted(Path("data").glob("sample_*"))
    all_nodes_per_focus = []

    zero_node_foci_count = 0
    total_foci = 0

    for sd in sample_dirs:
        with open(sd / "tumor_params.json") as f:
            tp = json.load(f)

        for focus in tp["foci"]:
            total_foci += 1
            center = np.array(focus["center"])
            radius = focus["params"].get("radius", 0.5)

            dists = np.linalg.norm(mesh_nodes - center, axis=1)
            nodes_inside = np.sum(dists <= radius)
            all_nodes_per_focus.append(nodes_inside)

            if nodes_inside == 0:
                zero_node_foci_count += 1

    all_nodes_per_focus = np.array(all_nodes_per_focus)

    print(f"Total foci analyzed: {total_foci}")
    print(f"Foci with 0 nodes inside: {zero_node_foci_count} ({zero_node_foci_count/total_foci*100:.1f}%)")
    print()
    print(f"Nodes per focus:")
    print(f"  min:  {all_nodes_per_focus.min()}")
    print(f"  max:  {all_nodes_per_focus.max()}")
    print(f"  mean: {all_nodes_per_focus.mean():.2f}")
    print(f"  median: {np.median(all_nodes_per_focus):.1f}")
    print(f"  P10:  {np.percentile(all_nodes_per_focus, 10):.1f}")
    print(f"  P25:  {np.percentile(all_nodes_per_focus, 25):.1f}")
    print(f"  P75:  {np.percentile(all_nodes_per_focus, 75):.1f}")
    print(f"  P90:  {np.percentile(all_nodes_per_focus, 90):.1f}")

    return all_nodes_per_focus


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
        signals.append({
            "sample": sd.name,
            "nonzero": nonzero_count,
            "max": meas_b.max(),
            "sum": meas_b.sum(),
            "nonzero_ratio": nonzero_count / len(meas_b),
            "num_foci": tp["num_foci"],
            "centers": [focus["center"] for focus in tp["foci"]],
            "radii": [focus["params"].get("radius", 0.5) for focus in tp["foci"]],
        })

    signals_sorted = sorted(signals, key=lambda x: x["sum"])

    print(f"All samples ({len(signals)}):")
    print(f"{'Sample':<15} {'Nonzero':>10} {'Max':>8} {'Sum':>12} {'Nonzero%':>10}")
    print("-" * 60)
    for s in signals_sorted[:5]:
        print(f"{s['sample']:<15} {s['nonzero']:>10} {s['max']:>8.4f} {s['sum']:>12.6f} {s['nonzero_ratio']*100:>9.1f}%")
    print("  ...")
    for s in signals_sorted[-5:]:
        print(f"{s['sample']:<15} {s['nonzero']:>10} {s['max']:>8.4f} {s['sum']:>12.6f} {s['nonzero_ratio']*100:>9.1f}%")

    print()
    print("=" * 70)
    print("LOWEST 5 SUM SAMPLES - DETAILED")
    print("=" * 70)
    for s in signals_sorted[:5]:
        print(f"\n{s['sample']}:")
        print(f"  sum={s['sum']:.6f}, max={s['max']:.4f}, nonzero={s['nonzero']}")
        for i, (c, r) in enumerate(zip(s["centers"], s["radii"])):
            print(f"  Focus {i}: center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), radius={r:.2f}")

    print()
    print("=" * 70)
    print("HIGHEST 5 SUM SAMPLES - DETAILED")
    print("=" * 70)
    for s in signals_sorted[-5:]:
        print(f"\n{s['sample']}:")
        print(f"  sum={s['sum']:.6f}, max={s['max']:.4f}, nonzero={s['nonzero']}")
        for i, (c, r) in enumerate(zip(s["centers"], s["radii"])):
            print(f"  Focus {i}: center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), radius={r:.2f}")


def diagnose_mesh_spacing(mesh_nodes, mesh_elements):
    """Diagnose mesh element edge lengths."""
    print()
    print("=" * 70)
    print("4. MESH NODE SPACING STATISTICS")
    print("=" * 70)

    elem = mesh_elements[:, :4]
    edge_indices = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

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

    print(f"\n  Nodes in tumor region: {len(tumor_nodes)} ({len(tumor_nodes)/len(mesh_nodes)*100:.1f}%)")

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
