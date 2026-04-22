#!/usr/bin/env python3
"""Verify exterior hull surface_faces correctness.

Go criteria:
1. surface_faces.shape[0] reduction: 15-40% vs old
2. Right-hand tet rate = 100%
3. FEM validate: max_response_value > 0, not NaN
4. Normal outward rate >= 99%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from fmt_simgen.mesh.mesh_generator import MeshGenerator
from fmt_simgen import DigimouseAtlas
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.physics.optical_params import OpticalParameterManager
import yaml

# Load trunk volume (pre-cropped at 0.2mm, labels already merged)
trunk_data = np.load("output/shared/trunk_volume.npz", allow_pickle=True)
trunk_volume = trunk_data["trunk_volume"]
trunk_voxel_size = float(trunk_data["voxel_size_mm"])  # 0.2mm

print(f"Trunk volume: {trunk_volume.shape}, voxel_size={trunk_voxel_size}mm")
print(f"Labels in trunk volume: {np.unique(trunk_volume)}")

# ── Old mesh reference ────────────────────────────────────────────────────────
OLD_PATHS = [
    Path("assets/mesh/mesh.npz"),
    Path("output/shared/mesh.npz"),
]

def load_old():
    for p in OLD_PATHS:
        if p.exists():
            d = np.load(p)
            return d["surface_faces"].shape[0], d["nodes"].shape[0]
    return None

old_info = load_old()
if old_info:
    old_n_faces, old_n_nodes = old_info
    print(f"Old mesh: {old_n_faces} surface_faces, {old_n_nodes} nodes")
else:
    old_n_faces = None
    print("No old mesh found for comparison")

# ── Generate new mesh ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating new mesh...")
print("=" * 60)

config = {
    "target_nodes": 10000,
    "surface_maxvol": 0.5,
    "deep_maxvol": 5.0,
    "roi_maxvol": 1.0,
    "output_path": "output/shared/",
}

ATLAS_PATH = "/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr"

gen = MeshGenerator(config)
mesh_data = gen.generate(
    atlas_volume=trunk_volume,  # pre-cropped trunk at 0.2mm
    voxel_size=trunk_voxel_size,
    downsample_factor=4,  # effective 0.8mm, matches old pipeline
    crop_to_trunk=False,  # already in trunk-local frame
)

nodes = mesh_data.nodes
elems = mesh_data.elements
surf_faces = mesh_data.surface_faces

print(f"\nNew mesh stats:")
print(f"  n_nodes = {nodes.shape[0]}")
print(f"  n_elements = {elems.shape[0]}")
print(f"  n_surface_faces = {surf_faces.shape[0]}")
print(f"  n_surface_nodes = {len(mesh_data.surface_node_indices)}")

# ── Criterion 1: surface_faces reduction ─────────────────────────────────────
print("\n" + "=" * 60)
print("GO CHECK 1: surface_faces reduction")
print("=" * 60)
if old_n_faces is not None:
    reduction = (old_n_faces - surf_faces.shape[0]) / old_n_faces * 100
    print(f"  Old: {old_n_faces}, New: {surf_faces.shape[0]}")
    print(f"  Reduction: {reduction:.1f}%")
    if 15 <= reduction <= 40:
        print("  PASS: reduction in 15-40% range")
    else:
        print(f"  WARN: reduction {reduction:.1f}% outside 15-40% range")
else:
    print("  SKIP: no old baseline")

# ── Criterion 2: right-hand tet orientation ────────────────────────────────────
print("\n" + "=" * 60)
print("GO CHECK 2: right-hand tet orientation")
print("=" * 60)
v0 = nodes[elems[:, 0]]
v1 = nodes[elems[:, 1]]
v2 = nodes[elems[:, 2]]
v3 = nodes[elems[:, 3]]
cross23 = np.cross(v1 - v0, v2 - v0)
det = np.sum(cross23 * (v3 - v0), axis=1)
rh_rate = float((det > 0).mean())
print(f"  Right-hand rate: {rh_rate:.6f}")
if rh_rate == 1.0:
    print("  PASS: 100% right-hand tets")
else:
    print(f"  FAIL: det>0 rate = {rh_rate}, expected 1.0")

# ── Criterion 3: FEM validate ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GO CHECK 3: FEM validate")
print("=" * 60)

config_path = Path(__file__).parent.parent / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)
physics_cfg = cfg.get("physics", {})
tissues_cfg = physics_cfg.get("tissues", {})
n_medium = physics_cfg.get("n", 1.37)
opt_mgr = OpticalParameterManager(tissues_cfg, n=n_medium)

try:
    solver = FEMSolver(
        nodes=nodes,
        elements=elems,
        surface_faces=surf_faces,
        tissue_labels=mesh_data.tissue_labels,
        opt_params_manager=opt_mgr,
    )
    solver.assemble_system_matrix()
    solver.validate()
    max_b = float(solver.max_response_value)
    max_coord = solver.max_response_coord
    B = solver.M
    print(f"  B.nnz = {B.nnz}")
    print(f"  max_response_value = {max_b:.6f}")
    print(f"  max_response_coord = {max_coord}")
    if not np.isnan(max_b) and max_b > 0:
        print("  PASS: FEM solution valid")
    else:
        print(f"  FAIL: invalid FEM solution (max_b={max_b})")
except Exception as e:
    print(f"  FAIL: exception: {e}")
    import traceback; traceback.print_exc()

# ── Criterion 4: normal outward (spot check on 50 random faces) ───────────────
print("\n" + "=" * 60)
print("GO CHECK 4: normal outward (spot check)")
print("=" * 60)
np.random.seed(42)
n_check = min(50, surf_faces.shape[0])
check_idx = np.random.choice(surf_faces.shape[0], n_check, replace=False)

# Build a lookup: (sorted_face_tuple) → (tet_idx, face_id)
T = elems.shape[0]
f0 = elems[:, [1, 2, 3]]
f1 = elems[:, [0, 3, 2]]
f2 = elems[:, [0, 1, 3]]
f3 = elems[:, [0, 2, 1]]
all_faces_arr = np.concatenate([f0, f1, f2, f3], axis=0).astype(np.int32)
all_tets_arr = np.repeat(np.arange(T, dtype=np.int32), 4)
all_fids_arr = np.tile(np.arange(4, dtype=np.int32), T)

sorted_all = np.sort(all_faces_arr, axis=1)
order = np.lexsort((sorted_all[:, 2], sorted_all[:, 1], sorted_all[:, 0]))
sorted_all = sorted_all[order]
all_tets_arr = all_tets_arr[order]
all_fids_arr = all_fids_arr[order]

diff = np.diff(sorted_all, axis=0, prepend=[[-1, -1, -1]])
is_diff = np.any(diff != 0, axis=1)
run_start = np.zeros(len(sorted_all), dtype=bool)
run_start[0] = True
run_start[1:] = is_diff[1:]
run_id = np.cumsum(run_start)
n_runs = int(run_id[-1])
run_counts = np.bincount(run_id, minlength=n_runs + 1)
ext_mask = run_counts[run_id] == 1

ext_sorted = sorted_all[ext_mask]
ext_tets = all_tets_arr[ext_mask]
ext_fids = all_fids_arr[ext_mask]

# Build dict: sorted_face → (tet, fid)
face_to_info = {}
for i in range(len(ext_sorted)):
    key = tuple(ext_sorted[i])
    face_to_info[key] = (ext_tets[i], ext_fids[i])

n_outward = 0
n_unmatched = 0
for fi in check_idx:
    f = surf_faces[fi]
    key = tuple(sorted(f))
    if key not in face_to_info:
        n_unmatched += 1
        continue
    tet_idx, fid = face_to_info[key]

    # Face normal
    va, vb, vc = nodes[f[0]], nodes[f[1]], nodes[f[2]]
    e1, e2 = vb - va, vc - va
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        continue
    n /= ln

    # Opposite vertex = the absent vertex (the one NOT in the face)
    opp = nodes[elems[tet_idx, fid]]
    centroid = (va + vb + vc) / 3.0
    to_outside = centroid - opp
    if np.dot(n, to_outside) > 0:
        n_outward += 1

checked = n_check - n_unmatched
outward_rate = n_outward / checked if checked > 0 else 0.0
print(f"  Checked {checked}/{n_check} ({n_unmatched} unmatched)")
print(f"  Outward rate: {outward_rate:.4f}")
if outward_rate >= 0.99:
    print("  PASS: outward rate >= 99%")
elif outward_rate >= 0.90:
    print(f"  WARN: outward rate {outward_rate:.4f} >= 90% but < 99%")
else:
    print(f"  FAIL: outward rate {outward_rate:.4f} < 90%")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"New mesh: {nodes.shape[0]} nodes, {elems.shape[0]} elements, {surf_faces.shape[0]} exterior faces")
if old_n_faces:
    pct = (old_n_faces - surf_faces.shape[0]) / old_n_faces * 100
    print(f"Old surface_faces: {old_n_faces} → New: {surf_faces.shape[0]} ({pct:+.1f}%)")
print(f"Right-hand tet rate: {rh_rate:.6f}")
