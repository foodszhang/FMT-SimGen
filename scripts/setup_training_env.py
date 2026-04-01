#!/usr/bin/env python3
"""
Step 0-setup: Rebuild shared assets and compute full-node Laplacian for training.

Usage:
    uv run python scripts/setup_training_env.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import yaml

from fmt_simgen.dataset.builder import DatasetBuilder
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.physics.optical_params import OpticalParameterManager


# ---------------------------------------------------------------------------
# 1. Rebuild mesh + system matrix components to output/shared
# ---------------------------------------------------------------------------
print("=" * 60)
print("[1] Rebuilding shared assets to output/shared")
print("=" * 60)

with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["mesh"]["output_path"] = "output/shared"
builder = DatasetBuilder(cfg)
assets = builder.build_shared_assets(force_regenerate=True)
print(f"Mesh: {assets['mesh']}, Matrix: {assets['matrix']}")

mesh_data = np.load(assets["mesh"])
nodes = mesh_data["nodes"]
elements = mesh_data["elements"]
surface_index = mesh_data["surface_node_indices"]
n_nodes = nodes.shape[0]
n_surface = len(surface_index)
print(f"Mesh: {n_nodes} nodes, {n_surface} surface nodes")


# ---------------------------------------------------------------------------
# 2. Compute forward matrix A [S, N] and save
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[2] Computing forward matrix A")
print("=" * 60)

t0 = time.time()
tissues_cfg = cfg.get("physics", {}).get("tissues", {})
n_medium = cfg.get("physics", {}).get("n", 1.37)
opt_mgr = OpticalParameterManager(tissues_cfg, n=n_medium)

solver = FEMSolver(
    nodes=nodes,
    elements=elements,
    surface_faces=mesh_data["surface_faces"],
    tissue_labels=mesh_data["tissue_labels"],
    opt_params_manager=opt_mgr,
)
solver.assemble_system_matrix()
A = solver.compute_forward_matrix()
print(f"A: {A.shape}, computed in {time.time()-t0:.1f}s")

sp.save_npz("output/shared/system_matrix.A.npz", sp.csr_matrix(A))
print(f"Saved to output/shared/system_matrix.A.npz")


# ---------------------------------------------------------------------------
# 3. Compute full-node Laplacian (topological + kernel)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[3] Computing full-node Graph Laplacian")
print("=" * 60)

def extract_edges(elements):
    M = elements.shape[0]
    combos = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.zeros((M * 6, 2), dtype=elements.dtype)
    for idx, (i, j) in enumerate(combos):
        edges[idx * M:(idx + 1) * M, 0] = elements[:, i]
        edges[idx * M:(idx + 1) * M, 1] = elements[:, j]
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def topological_laplacian(W, n):
    d = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ (sp.diags(d) - W) @ D_inv_sqrt).tocsr()


def kernel_laplacian_radius(nodes, r):
    n = nodes.shape[0]
    sigma = r
    cutoff = 3.0 * sigma
    row_list, col_list, data_list = [], [], []
    for i in range(n):
        diffs = nodes - nodes[i]
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists <= cutoff
        if not mask.any():
            row_list.append(i); col_list.append(i); data_list.append(1.0)
            continue
        neighbors = np.where(mask)[0]
        dists_masked = dists[mask]
        weights = np.exp(-dists_masked ** 2 / (2 * sigma ** 2))
        radius_mask = dists <= r
        radius_neighbors = neighbors[radius_mask[mask]]
        if radius_neighbors.size == 0:
            row_list.append(i); col_list.append(i); data_list.append(1.0)
        else:
            for j_idx, j in enumerate(radius_neighbors):
                w = np.exp(-(dists[radius_mask][j_idx]) ** 2 / (2 * sigma ** 2))
                row_list.append(i); col_list.append(j); data_list.append(w)
    W = sp.coo_matrix((data_list, (row_list, col_list)), shape=(n, n), dtype=np.float32).tocsr()
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums_inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    return (sp.diags(row_sums_inv) @ W).tocsr()


# Topological Laplacian
t0 = time.time()
edges = extract_edges(elements)
row = np.concatenate([edges[:, 0], edges[:, 1]])
col = np.concatenate([edges[:, 1], edges[:, 0]])
W = sp.coo_matrix(
    (np.ones(2 * len(edges), dtype=np.float32), (row, col)),
    shape=(n_nodes, n_nodes)
).tocsr()
L = topological_laplacian(W, n_nodes)
sp.save_npz("output/shared/graph_laplacian_full.Lap.npz", L)
print(f"Lap_full: nnz={L.nnz}, sparsity={L.nnz/(n_nodes**2)*100:.4f}%, time={time.time()-t0:.1f}s")

# Kernel Laplacians r=1,2,3,4mm
nodes_f = nodes.astype(np.float32)
for lap_idx, r in enumerate([1.0, 2.0, 3.0, 4.0]):
    t0 = time.time()
    L_r = kernel_laplacian_radius(nodes_f, r)
    sp.save_npz(f"output/shared/graph_laplacian_full.n_Lap{lap_idx}.npz", L_r)
    print(f"n_Lap{lap_idx} r={r}mm: nnz={L_r.nnz}, sparsity={L_r.nnz/(n_nodes**2)*100:.4f}%, time={time.time()-t0:.1f}s")

# kNN indices
t0 = time.time()
nn = NearestNeighbors(n_neighbors=33, algorithm="kd_tree", metric="euclidean")
nn.fit(nodes_f)
_, indices = nn.kneighbors(nodes_f)
knn_idx = indices[:, 1:].astype(np.int32)
np.save("output/shared/knn_idx_full.npy", knn_idx)
print(f"knn_idx: {knn_idx.shape}, time={time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# 4. Verify dimensions and update train_config.yaml
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[4] Final verification and config update")
print("=" * 60)

mesh_check = np.load("output/shared/mesh.npz")
A_check = sp.load_npz("output/shared/system_matrix.A.npz")
L_check = sp.load_npz("output/shared/graph_laplacian_full.Lap.npz")
knn_check = np.load("output/shared/knn_idx_full.npy")

print(f"nodes:    {mesh_check['nodes'].shape}")
print(f"surface:  {mesh_check['surface_node_indices'].shape}")
print(f"A:        {A_check.shape}")
print(f"L:        {L_check.shape}")
print(f"knn_idx:  {knn_check.shape}")

# Update train_config.yaml
train_cfg_path = Path("train/config/train_config.yaml")
with open(train_cfg_path) as f:
    train_cfg = yaml.safe_load(f)

train_cfg['model']['n_nodes'] = int(n_nodes)
train_cfg['model']['n_surface'] = int(n_surface)
train_cfg['model']['n_neighbors'] = 32

with open(train_cfg_path, "w") as f:
    yaml.dump(train_cfg, f, default_flow_style=False)
print(f"Updated train/config/train_config.yaml: n_nodes={n_nodes}, n_surface={n_surface}")

print("\nDone. Next: uv run python train/train.py")
