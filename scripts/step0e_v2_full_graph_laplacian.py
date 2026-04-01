#!/usr/bin/env python3
"""
Step 0e-v2: Compute full-node Graph Laplacians (all 11535 nodes).

Computes:
1. Topological Laplacian (Lap_full) from tetrahedral element adjacency
2. Kernel Laplacians (n_Lap0..n_Lap3) at radii 1,2,3,4mm
3. kNN indices (k=32) for cross-attention

Usage:
    python scripts/step0e_v2_full_graph_laplacian.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def extract_edges_from_elements(elements: np.ndarray) -> np.ndarray:
    """Extract all unique edges from tetrahedral elements.

    Each tetrahedron has 4 choose 2 = 6 edges.
    elements: [M, 4] int array of node indices
    returns: [E, 2] int array of edge pairs (sorted)
    """
    M = elements.shape[0]
    edges = np.zeros((M * 6, 2), dtype=elements.dtype)
    combos = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for idx, (i, j) in enumerate(combos):
        edges[idx * M:(idx + 1) * M, 0] = elements[:, i]
        edges[idx * M:(idx + 1) * M, 1] = elements[:, j]
    # Sort each row so (i,j) with i <= j
    edges = np.sort(edges, axis=1)
    # Unique edges
    edges = np.unique(edges, axis=0)
    return edges  # [E, 2]


def build_sparse_adjacency(n_nodes: int, edges: np.ndarray) -> sp.spmatrix:
    """Build symmetric sparse adjacency matrix from edges."""
    n_edges = edges.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(2 * n_edges, dtype=np.float32)
    W = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    return W


def topological_laplacian(W: sp.spmatrix) -> sp.spmatrix:
    """Symmetric normalized topological Laplacian: L = D^{-1/2} (D-W) D^{-1/2}."""
    d = np.array(W.sum(axis=1)).flatten()  # degree vector
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = D_inv_sqrt @ (sp.diags(d) - W) @ D_inv_sqrt
    return L.tocsr()


def kernel_laplacian_radius(
    nodes: np.ndarray,
    r: float,
    sigma: float = None,
) -> sp.spmatrix:
    """Row-normalized Gaussian kernel Laplacian at given radius.

    For each node i, connect to all nodes j where ||xi - xj|| <= r.
    Weight: w_ij = exp(-||xi-xj||^2 / (2*sigma^2)), truncated at 3*sigma.
    Row-normalized: L_ij = w_ij / sum_k(w_ik).
    """
    n = nodes.shape[0]
    if sigma is None:
        sigma = r  # use r as sigma for truncation

    cutoff = 3.0 * sigma

    # Build adjacency with Gaussian weights
    row_list = []
    col_list = []
    data_list = []

    for i in range(n):
        diffs = nodes - nodes[i]  # [n, 3]
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists <= cutoff
        dists_masked = dists[mask]

        if dists_masked.size == 0:
            # Isolated node - self loop only
            row_list.append(i)
            col_list.append(i)
            data_list.append(1.0)
            continue

        # Gaussian weights for neighbors within cutoff
        weights = np.exp(-dists_masked ** 2 / (2 * sigma ** 2))

        # Apply radius mask: only keep nodes within radius r
        neighbors = np.where(mask)[0]
        radius_mask = dists <= r
        radius_neighbors = neighbors[radius_mask[mask]]  # within radius

        if radius_neighbors.size == 0:
            # No neighbors within radius - self loop
            row_list.append(i)
            col_list.append(i)
            data_list.append(1.0)
        else:
            for j_idx, j in enumerate(radius_neighbors):
                w = np.exp(-(dists[radius_mask][j_idx]) ** 2 / (2 * sigma ** 2))
                row_list.append(i)
                col_list.append(j)
                data_list.append(w)

    W = sp.coo_matrix(
        (data_list, (row_list, col_list)),
        shape=(n, n),
        dtype=np.float32
    ).tocsr()

    # Row normalization: L = D^{-1} W
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums_inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    D_inv = sp.diags(row_sums_inv)
    L = D_inv @ W
    return L.tocsr()


def main():
    t_start = time.time()

    shared_dir = Path(__file__).parent.parent / "output" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 0e-v2: Full-node Graph Laplacian")
    print("=" * 60)

    # Load mesh
    print("\n[1] Loading mesh...")
    mesh = np.load(shared_dir / "mesh.npz")
    nodes = mesh["nodes"].astype(np.float32)  # [11535, 3]
    elements = mesh["elements"]  # [60026, 4]
    n_nodes = nodes.shape[0]
    print(f"    nodes: {nodes.shape}, elements: {elements.shape}")

    # ----------------------------------------------------------------
    # 1. Topological Laplacian (from tetrahedral adjacency)
    # ----------------------------------------------------------------
    print("\n[2] Computing topological Laplacian from tetrahedra...")
    t0 = time.time()
    edges = extract_edges_from_elements(elements)
    print(f"    unique edges: {edges.shape[0]}")
    W = build_sparse_adjacency(n_nodes, edges)
    print(f"    adjacency built in {time.time()-t0:.1f}s, nnz={W.nnz}")

    t0 = time.time()
    Lap = topological_laplacian(W)
    lap_path = shared_dir / "graph_laplacian_full.Lap.npz"
    sp.save_npz(lap_path, Lap)
    t_elapsed = time.time() - t0
    sparsity = Lap.nnz / (n_nodes * n_nodes) * 100
    print(f"    Lap_full: shape=({n_nodes},{n_nodes}), nnz={Lap.nnz}, sparsity={sparsity:.4f}%, time={t_elapsed:.1f}s")

    # ----------------------------------------------------------------
    # 2. Kernel Laplacians (r = 1, 2, 3, 4 mm)
    # ----------------------------------------------------------------
    radii = [0.001, 0.002, 0.003, 0.004]  # meters (nodes are in mm)
    # Actually nodes are in mm, so radii in mm
    radii_mm = [1.0, 2.0, 3.0, 4.0]

    for lap_idx, (r, sigma) in enumerate(zip(radii_mm, radii_mm)):
        print(f"\n[3.{lap_idx}] Kernel Laplacian r={r}mm (n_Lap{lap_idx})...")
        t0 = time.time()
        L_r = kernel_laplacian_radius(nodes, r, sigma=r)
        out_path = shared_dir / f"graph_laplacian_full.n_Lap{lap_idx}.npz"
        sp.save_npz(out_path, L_r)
        t_elapsed = time.time() - t0
        sparsity = L_r.nnz / (n_nodes * n_nodes) * 100
        size_mb = out_path.stat().st_size / 1e6
        print(f"    n_Lap{lap_idx}: shape=({n_nodes},{n_nodes}), nnz={L_r.nnz}, sparsity={sparsity:.4f}%, size={size_mb:.2f}MB, time={t_elapsed:.1f}s")

    # ----------------------------------------------------------------
    # 3. kNN indices for cross-attention (k=32)
    # ----------------------------------------------------------------
    print("\n[4] Computing kNN indices (k=32)...")
    t0 = time.time()
    k = 32
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", metric="euclidean")
    nn.fit(nodes)
    distances, indices = nn.kneighbors(nodes)
    # Remove self (first column)
    knn_idx = indices[:, 1:].astype(np.int32)  # [11535, 32]
    knn_path = shared_dir / "knn_idx_full.npy"
    np.save(knn_path, knn_idx)
    t_elapsed = time.time() - t0
    print(f"    knn_idx: {knn_idx.shape}, time={t_elapsed:.1f}s")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("Summary - output files in output/shared/")
    print("=" * 60)

    files = [
        "graph_laplacian_full.Lap.npz",
        "graph_laplacian_full.n_Lap0.npz",
        "graph_laplacian_full.n_Lap1.npz",
        "graph_laplacian_full.n_Lap2.npz",
        "graph_laplacian_full.n_Lap3.npz",
        "knn_idx_full.npy",
    ]
    for fname in files:
        fpath = shared_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            if fname.endswith(".npy"):
                arr = np.load(fpath)
                print(f"  {fname}: {arr.shape}, {size_mb:.2f}MB")
            else:
                mat = sp.load_npz(fpath)
                sparsity = mat.nnz / (n_nodes * n_nodes) * 100
                print(f"  {fname}: ({n_nodes},{n_nodes}), nnz={mat.nnz}, sparsity={sparsity:.4f}%, {size_mb:.2f}MB")
        else:
            print(f"  {fname}: NOT FOUND")

    print(f"\nTotal elapsed time: {total_time:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
