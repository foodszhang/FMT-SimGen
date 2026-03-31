"""
Graph Laplacian computation for MS-GDUN training.

Computes:
1. Standard topological Laplacian from surface mesh adjacency
2. Multi-scale kernel graph Laplacian with radii 1, 2, 3, 4 mm
"""

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
from typing import Tuple, Dict


def build_surface_adjacency(surface_faces: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    """Build adjacency matrix from surface triangular faces.

    Parameters
    ----------
    surface_faces : np.ndarray
        Surface triangles [F x 3] with global node indices (0-based).
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    sparse.csr_matrix
        Symmetric adjacency matrix [n_nodes x n_nodes].
    """
    n_faces = surface_faces.shape[0]
    rows = []
    cols = []

    for f in range(n_faces):
        i, j, k = surface_faces[f]
        rows.extend([i, j, k, i, j, k])
        cols.extend([j, i, k, k, i, j])

    data = np.ones(len(rows))
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    adj = (adj > 0).astype(float)

    return adj


def remap_surface_indices(surface_faces: np.ndarray, surface_node_indices: np.ndarray) -> np.ndarray:
    """Remap global node indices to local (0-based) surface node indices.

    Parameters
    ----------
    surface_faces : np.ndarray
        Surface triangles [F x 3] with global node indices.
    surface_node_indices : np.ndarray
        Global indices of surface nodes [S].

    Returns
    -------
    np.ndarray
        Remapped surface faces [F x 3] with local indices (0 to S-1).
    """
    S = len(surface_node_indices)
    global_to_local = np.full(surface_node_indices.max() + 1, -1, dtype=np.int64)
    global_to_local[surface_node_indices] = np.arange(S, dtype=np.int64)

    remapped = global_to_local[surface_faces]
    return remapped


def compute_topological_laplacian(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """Compute normalized topological Laplacian: L = I - D^{-1/2} * A * D^{-1/2}.

    Parameters
    ----------
    adj : sparse.csr_matrix
        Symmetric adjacency matrix.

    Returns
    -------
    sparse.csr_matrix
        Normalized Laplacian matrix.
    """
    n = adj.shape[0]
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_sqrt_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))

    D_inv_sqrt = sparse.diags(deg_sqrt_inv)
    L = sparse.eye(n, format="csr") - D_inv_sqrt @ adj @ D_inv_sqrt

    return L


def compute_kernel_laplacian_kdtree(
    surface_coords: np.ndarray,
    radii: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
) -> Dict[str, sparse.csr_matrix]:
    """Compute multi-scale kernel graph Laplacian using KDTree.

    Parameters
    ----------
    surface_coords : np.ndarray
        Surface node coordinates [S x 3].
    radii : tuple of float
        Kernel radii in mm.

    Returns
    -------
    Dict[str, sparse.csr_matrix]
        Dictionary with keys 'n_Lap0', 'n_Lap1', 'n_Lap2', 'n_Lap3' for each radius.
    """
    S = surface_coords.shape[0]
    tree = KDTree(surface_coords)

    lapacians = {}

    for idx, r in enumerate(radii):
        cutoff = 3.0 * r

        rows = []
        cols = []
        data = []

        for i in range(S):
            neighbor_indices = tree.query_ball_point(surface_coords[i], r=cutoff)
            x_i = surface_coords[i]

            for j in neighbor_indices:
                if i == j:
                    continue
                x_j = surface_coords[j]
                dist_sq = np.sum((x_i - x_j) ** 2)
                weight = np.exp(-dist_sq / (2.0 * r**2))
                rows.append(i)
                cols.append(j)
                data.append(weight)

        W = sparse.csr_matrix((data, (rows, cols)), shape=(S, S))
        W = W + W.T

        D = np.array(W.sum(axis=1)).flatten()
        D_sqrt_inv = 1.0 / np.sqrt(np.maximum(D, 1e-12))
        D_inv_sqrt = sparse.diags(D_sqrt_inv)

        L = sparse.eye(S, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
        lapacians[f"n_Lap{idx}"] = L

    return lapacians


def compute_laplacian_statistics(
    L: sparse.csr_matrix,
    lap_name: str = "Laplacian",
) -> Dict[str, float]:
    """Compute statistics for Laplacian validation.

    Parameters
    ----------
    L : sparse.csr_matrix
        Laplacian matrix.
    lap_name : str
        Name for logging.

    Returns
    -------
    Dict[str, float]
        Statistics dictionary.
    """
    L_dense = L.toarray() if sparse.issparse(L) else L

    n = L_dense.shape[0]
    row_sums = L_dense.sum(axis=1)

    sym_max = np.max(np.abs(L_dense - L_dense.T))

    stats = {
        "shape": (n, n),
        "nnz": np.count_nonzero(L_dense),
        "sparsity": 1.0 - np.count_nonzero(L_dense) / (n * n),
        "row_sum_max": np.max(np.abs(row_sums)),
        "sym_max": sym_max,
    }
    return stats


def compute_kernel_neighbors_stats(
    surface_coords: np.ndarray,
    radius: float,
) -> Dict[str, float]:
    """Compute neighbor statistics for kernel Laplacian.

    Parameters
    ----------
    surface_coords : np.ndarray
        Surface node coordinates [S x 3].
    radius : float
        Kernel radius in mm.

    Returns
    -------
    Dict[str, float]
        Neighbor statistics.
    """
    S = surface_coords.shape[0]
    tree = KDTree(surface_coords)
    cutoff = 3.0 * radius

    neighbor_counts = []
    for i in range(S):
        neighbor_indices = tree.query_ball_point(surface_coords[i], r=cutoff)
        neighbor_counts.append(len(neighbor_indices) - 1)

    return {
        "radius": radius,
        "mean_neighbors": np.mean(neighbor_counts),
        "min_neighbors": np.min(neighbor_counts),
        "max_neighbors": np.max(neighbor_counts),
    }
