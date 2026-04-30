#!/usr/bin/env python3
"""
Step 0b: Generate multi-label tetrahedral mesh from Digimouse atlas using iso2mesh cgalmesh.

This script generates a conforming multi-tissue mesh with multi-organ interface surfaces:
1. Load atlas with nibabel (handles NIfTI coordinate system correctly)
2. Crop to trunk region Y=[340, 740] voxels at 0.1mm
3. Apply tissue mapping (original 22 labels → 10 DE tissue classes)
4. Downsample by factor 2 (effective 0.2mm voxel)
5. Generate mesh with iso2mesh v2m (cgalmesh backend, multiple isovalues)
6. Convert node coords: node_idx * effective_voxel = trunk-local mm (crop origin = trunk origin)
7. Separate exterior hull faces from interior organ/interface faces
8. Save mesh and generate visualizations

Usage:
    python scripts/step0b_generate_mesh_cgalmesh.py [--maxvol 3.0] [--radbound 2.5] [--distbound 2.0]

Output:
    output/shared/digimouse_trunk_mesh.npz
    output/shared/digimouse_trunk_mesh.mat
    output/visualizations/mesh_*.png
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
import iso2mesh
from scipy import ndimage
import scipy.io as sio
import yaml

from fmt_simgen.frame_contract import VOLUME_EXTENTS_MM, assert_in_trunk_bbox
from fmt_simgen.mesh.mesh_generator import MeshGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "shared"
VIS_DIR = Path(__file__).parent.parent / "output" / "visualizations"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


LABEL_NAMES = {
    0: "background",
    1: "soft_tissue",
    2: "bone",
    3: "brain",
    4: "heart",
    5: "stomach",
    6: "abdominal",
    7: "liver",
    8: "kidney",
    9: "lung",
}


def load_atlas_with_nibabel(atlas_path: str) -> np.ndarray:
    """Load Digimouse atlas using nibabel."""
    logger.info(f"Loading atlas with nibabel: {atlas_path}")
    img = nib.load(atlas_path)
    volume = np.asarray(img.dataobj).astype(np.uint8)

    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
        logger.info("Removed 4th dimension")

    logger.info(f"Atlas shape: {volume.shape}")
    return volume


def apply_tissue_mapping(volume: np.ndarray, mapping: dict) -> np.ndarray:
    """Apply tissue label mapping."""
    logger.info("Applying tissue mapping...")
    mapped = np.zeros_like(volume, dtype=np.uint8)
    for orig_label, new_label in mapping.items():
        mask = volume == orig_label
        mapped[mask] = new_label

    unique_labels = np.unique(mapped)
    logger.info(f"Mapped labels: {unique_labels}")
    return mapped


def downsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample volume using nearest-neighbor interpolation."""
    logger.info(f"Downsampling with factor {factor}...")
    zoom_factors = [1.0 / factor] * 3
    ds = ndimage.zoom(volume, zoom_factors, order=0, prefilter=False)
    ds = ds.astype(np.uint8)
    logger.info(f"Downsampled shape: {ds.shape}")
    return ds


def extract_exterior_and_interface_faces(
    node_idx_arr: np.ndarray,
    elements: np.ndarray,
    face_arr: np.ndarray,
    tissue_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract exterior hull and interior interface faces.

    Parameters
    ----------
    node_idx_arr : np.ndarray
        Node coordinates in voxel index space [N, 3+], 1-based from v2m.
    elements : np.ndarray
        Tetrahedron node indices [T, 4], 0-based.
    face_arr : np.ndarray
        Face array from v2m [F, 4] = [n0, n1, n2, face_label], 1-based.
    tissue_labels : np.ndarray
        Tissue label for each element [T].

    Returns
    -------
    tuple
        (exterior_faces, interface_faces), both 0-based triangle node indices.
    """
    # Build face_key (sorted 0-based node tuple) -> tet indices
    face_key_to_tets: dict[tuple, list] = {}
    for i_tet, elem in enumerate(elements):
        n0, n1, n2, n3 = elem
        for face_nodes in [(n0, n1, n2), (n0, n1, n3), (n0, n2, n3), (n1, n2, n3)]:
            key = tuple(sorted(face_nodes))
            face_key_to_tets.setdefault(key, []).append(i_tet)

    # Separate exterior (1 tet) and interface (2 tets, different labels)
    exterior_keys = set()
    interface_keys = set()
    for key, tet_list in face_key_to_tets.items():
        if len(tet_list) == 1:
            exterior_keys.add(key)
        elif len(tet_list) == 2:
            t0, t1 = tet_list
            if tissue_labels[t0] != tissue_labels[t1]:
                interface_keys.add(key)

    logger.info(f"Face keys: exterior={len(exterior_keys)}, interface={len(interface_keys)}")

    # face_arr is 1-based: [n0, n1, n2, face_label]
    # face may appear multiple times (once per isovalue), deduplicate by sorted node tuple
    exterior_faces_list = []
    interface_faces_list = []
    seen_exterior = set()
    seen_interface = set()
    for f in face_arr[:, :3]:
        key = tuple(sorted(f - 1))  # 1-based → 0-based
        if key in exterior_keys and key not in seen_exterior:
            exterior_faces_list.append(f - 1)
            seen_exterior.add(key)
        if key in interface_keys and key not in seen_interface:
            interface_faces_list.append(f - 1)
            seen_interface.add(key)

    exterior_faces = np.array(exterior_faces_list, dtype=np.int32)
    interface_faces = np.array(interface_faces_list, dtype=np.int32)

    logger.info(f"Matched faces: exterior={len(exterior_faces)}, interface={len(interface_faces)}")
    return exterior_faces, interface_faces


def visualize_mesh(
    nodes_mm: np.ndarray,
    elem_0based: np.ndarray,
    tissue_labels: np.ndarray,
    exterior_faces: np.ndarray,
    interface_faces: np.ndarray,
    output_dir: Path,
):
    """Generate mesh visualizations using PyVista."""
    import pyvista as pv

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build UnstructuredGrid
    cells = np.hstack([np.full((len(elem_0based), 1), 4), elem_0based])
    grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA] * len(elem_0based), nodes_mm)
    grid.cell_data["tissue"] = tissue_labels

    def build_surface(nodes, faces):
        if len(faces) == 0:
            return None
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces])
        return pv.PolyData(nodes, faces_pv)

    surf_ext = build_surface(nodes_mm, exterior_faces)
    surf_iface = build_surface(nodes_mm, interface_faces)

    # 1. External surface
    if surf_ext:
        logger.info("Rendering external surface...")
        p = pv.Plotter(off_screen=True, window_size=[1200, 800])
        p.add_mesh(surf_ext, color="lightblue", opacity=0.9, smooth_shading=True)
        p.add_axes()
        p.add_title(f"External Surface\n{len(nodes_mm):,} nodes | {len(exterior_faces):,} faces")
        p.screenshot(str(output_dir / "mesh_external_surface.png"))
        p.close()
        logger.info(f"[✓] mesh_external_surface.png")

    # 2. Internal organ interfaces
    if surf_iface:
        logger.info("Rendering internal interfaces...")
        p = pv.Plotter(off_screen=True, window_size=[1200, 800])
        p.add_mesh(surf_iface, color="yellow", opacity=0.8, smooth_shading=True)
        if surf_ext:
            p.add_mesh(surf_ext, color="lightblue", opacity=0.3, style="wireframe")
        p.add_axes()
        p.add_title(f"Internal Organ Interfaces\n{len(interface_faces):,} interface faces")
        p.screenshot(str(output_dir / "mesh_internal_interfaces.png"))
        p.close()
        logger.info(f"[✓] mesh_internal_interfaces.png")

    # 3. Cross-sections
    logger.info("Rendering cross-sections...")
    p = pv.Plotter(off_screen=True, window_size=[1600, 600], shape=(1, 3))

    p.subplot(0, 0)
    slice_xy = grid.slice(normal="z", origin=[0, 0, 10.0])
    p.add_mesh(slice_xy, scalars="tissue", show_edges=True, cmap="tab10")
    p.add_axes()
    p.add_title("XY at Z=10mm")

    p.subplot(0, 1)
    slice_xz = grid.slice(normal="y", origin=[0, 20.0, 0])
    p.add_mesh(slice_xz, scalars="tissue", show_edges=True, cmap="tab10")
    p.add_axes()
    p.add_title("XZ at Y=20mm")

    p.subplot(0, 2)
    slice_yz = grid.slice(normal="x", origin=[18.0, 0, 0])
    p.add_mesh(slice_yz, scalars="tissue", show_edges=True, cmap="tab10")
    p.add_axes()
    p.add_title("YZ at X=18mm")

    p.screenshot(str(output_dir / "mesh_cross_sections.png"))
    p.close()
    logger.info(f"[✓] mesh_cross_sections.png")

    # 4. Multi-tissue 3D view
    logger.info("Rendering multi-tissue 3D view...")
    p = pv.Plotter(off_screen=True, window_size=[1600, 800])
    p.add_mesh(grid, scalars="tissue", show_edges=False, cmap="tab10", opacity=0.7)
    p.add_axes()
    p.add_title(f"Multi-tissue Mesh\n{len(nodes_mm):,} nodes | {len(elem_0based):,} tets")
    p.screenshot(str(output_dir / "mesh_multilabel_3d.png"))
    p.close()
    logger.info(f"[✓] mesh_multilabel_3d.png")

    # 5. Per-organ surfaces (3x3 grid)
    logger.info("Rendering per-organ surfaces...")
    p = pv.Plotter(off_screen=True, window_size=[1600, 1000], shape=(4, 3))
    LABEL_COLORS = [
        "gray", "lightpink", "beige", "plum", "red",
        "mintcream", "yellow", "dodgerblue", "purple", "lightblue",
    ]
    for idx, (label, name) in enumerate(sorted(LABEL_NAMES.items())):
        organ_grid = grid.threshold(value=[label - 0.5, label + 0.5], scalars="tissue")
        p.subplot(idx // 3, idx % 3)
        if organ_grid.n_cells > 0:
            organ_surf = organ_grid.extract_surface()
            p.add_mesh(organ_surf, color=LABEL_COLORS[label], opacity=0.9)
            p.add_title(f"{name}\n{organ_grid.n_cells:,} elems")
        else:
            p.add_title(f"{name}\nNo elements")
        p.add_axes()
    p.screenshot(str(output_dir / "mesh_organ_surfaces.png"))
    p.close()
    logger.info(f"[✓] mesh_organ_surfaces.png")


def compute_and_print_statistics(
    nodes_mm: np.ndarray,
    elem_0based: np.ndarray,
    tissue_labels: np.ndarray,
    exterior_faces: np.ndarray,
    interface_faces: np.ndarray,
) -> None:
    """Compute and print mesh statistics."""
    logger.info("=" * 60)
    logger.info("MESH STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Nodes: {len(nodes_mm):,}")
    logger.info(f"Tetrahedra: {len(elem_0based):,}")
    logger.info(f"Exterior faces: {len(exterior_faces):,}")
    logger.info(f"Interface faces: {len(interface_faces):,}")
    logger.info(f"\nBounding box (mm):")
    logger.info(f"  X: [{nodes_mm[:, 0].min():.2f}, {nodes_mm[:, 0].max():.2f}]")
    logger.info(f"  Y: [{nodes_mm[:, 1].min():.2f}, {nodes_mm[:, 1].max():.2f}]")
    logger.info(f"  Z: [{nodes_mm[:, 2].min():.2f}, {nodes_mm[:, 2].max():.2f}]")

    logger.info(f"\nElements per tissue:")
    for label in sorted(np.unique(tissue_labels)):
        mask = tissue_labels == label
        count = np.sum(mask)
        name = LABEL_NAMES.get(int(label), f"unknown_{label}")
        logger.info(f"  {name:12s}: {count:,} elements ({100*count/len(tissue_labels):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 0b: Generate multi-label mesh using iso2mesh cgalmesh"
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default=None,
        help="Atlas path (overrides config). E.g., /path/to/atlas_380x992x208.hdr",
    )
    parser.add_argument(
        "--maxvol",
        type=float,
        default=3.0,
        help="Maximum tetrahedron volume in mm³ (default: 3.0)",
    )
    parser.add_argument(
        "--radbound",
        type=float,
        default=2.5,
        help="Maximum surface triangle edge length (default: 2.5)",
    )
    parser.add_argument(
        "--distbound",
        type=float,
        default=2.0,
        help="Distance to tissue boundary (default: 2.0)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsampling factor (default: 2, gives 0.2mm effective voxel)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="digimouse_trunk_mesh",
        help="Output name prefix (default: digimouse_trunk_mesh)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 0b: Multi-label Mesh Generation (iso2mesh cgalmesh)")
    logger.info("=" * 60)

    # ── Load config ──────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # ── Atlas path ──────────────────────────────────────────────────────────
    if args.atlas:
        atlas_path = args.atlas
    else:
        atlas_path = config["atlas"]["path"]
    logger.info(f"Atlas path: {atlas_path}")

    # ── Trunk crop parameters ──────────────────────────────────────────────
    y_start = config["mcx"]["trunk_crop"]["y_start"]  # 340 at 0.1mm
    y_end = config["mcx"]["trunk_crop"]["y_end"]      # 740 at 0.1mm
    logger.info(f"Trunk crop Y: [{y_start}, {y_end}] at 0.1mm")

    # ── Tissue mapping ──────────────────────────────────────────────────────
    tissue_mapping = config["atlas"]["tissue_merge"]
    logger.info(f"Tissue mapping: {len(tissue_mapping)} entries")

    # ── Load atlas ──────────────────────────────────────────────────────────
    volume = load_atlas_with_nibabel(atlas_path)
    logger.info(f"Original atlas shape: {volume.shape}")

    # ── Crop trunk ──────────────────────────────────────────────────────────
    trunk = volume[:, y_start:y_end, :].copy()
    logger.info(f"Trunk shape after crop: {trunk.shape}")

    # ── Apply tissue mapping ─────────────────────────────────────────────────
    trunk_mapped = apply_tissue_mapping(trunk, tissue_mapping)

    # ── Downsample ───────────────────────────────────────────────────────────
    downsample_factor = args.downsample
    voxel_size = config["atlas"]["voxel_size"]  # 0.1mm original
    effective_voxel = voxel_size * downsample_factor  # 0.2mm

    vol_ds = downsample_volume(trunk_mapped, downsample_factor)
    logger.info(f"Downsampled shape: {vol_ds.shape} at {effective_voxel}mm")

    # ── Generate mesh with iso2mesh v2m ─────────────────────────────────────
    non_zero_labels = [int(l) for l in np.unique(vol_ds) if l > 0]
    logger.info(f"Non-zero labels for v2m: {non_zero_labels}")

    opt = {"radbound": args.radbound, "distbound": args.distbound}
    logger.info(
        f"v2m: maxvol={args.maxvol}, radbound={args.radbound}, "
        f"distbound={args.distbound}, method=cgalmesh"
    )

    node_idx, elem, face = iso2mesh.v2m(
        vol_ds,
        isovalues=non_zero_labels,
        opt=opt,
        maxvol=args.maxvol,
        method="cgalmesh",
    )

    logger.info(f"v2m output: node={node_idx.shape}, elem={elem.shape}, face={face.shape}")

    # ── Convert node coordinates to trunk-local mm ───────────────────────────
    # Crop window Y=[340,740] at 0.1mm = physical Y=[34,74]mm
    # Crop origin in atlas = Y_start * 0.1mm = 34mm
    # Crop origin in physical = trunk origin = (0, 0, 0) in trunk-local frame
    # Therefore: node_idx * effective_voxel = trunk-local mm (NO offset needed)
    nodes_phys = node_idx[:, :3] * effective_voxel

    logger.info(f"Nodes physical range X: [{nodes_phys[:,0].min():.2f}, {nodes_phys[:,0].max():.2f}]")
    logger.info(f"Nodes physical range Y: [{nodes_phys[:,1].min():.2f}, {nodes_phys[:,1].max():.2f}]")
    logger.info(f"Nodes physical range Z: [{nodes_phys[:,2].min():.2f}, {nodes_phys[:,2].max():.2f}]")

    # ── Process elements ─────────────────────────────────────────────────────
    # 1-based → 0-based
    elements = elem[:, :4].astype(np.int32) - 1
    tissue_labels = elem[:, 4].astype(np.int32)

    # Fix tet orientation
    elements = MeshGenerator._ensure_tet_orientation(nodes_phys, elements)

    # ── Extract exterior hull and interior interface faces ──────────────────
    exterior_faces, interface_faces = extract_exterior_and_interface_faces(
        node_idx, elements, face, tissue_labels
    )

    # ── Compute and print statistics ─────────────────────────────────────────
    compute_and_print_statistics(
        nodes_phys, elements, tissue_labels, exterior_faces, interface_faces
    )

    # ── Verification: assert nodes are in trunk-local frame ──────────────────
    logger.info("\n=== Frame Verification ===")
    try:
        assert_in_trunk_bbox(nodes_phys, tol_mm=3.0)
        logger.info("✓ assert_in_trunk_bbox passed: nodes are in trunk-local frame")
    except AssertionError as e:
        logger.error(f"✗ Frame verification failed: {e}")
        raise

    # Check bbox vs trunk size
    nodes_min, nodes_max = nodes_phys.min(axis=0), nodes_phys.max(axis=0)
    logger.info(f"Mesh bbox: min={nodes_min}, max={nodes_max}")
    logger.info(f"Trunk size: [0, {VOLUME_EXTENTS_MM[0]}] × [0, {VOLUME_EXTENTS_MM[1]}] × [0, {VOLUME_EXTENTS_MM[2]}]")

    # Check exterior surface quality
    V_ext = len(np.unique(exterior_faces))
    F_ext = len(exterior_faces)
    fv_ratio = F_ext / max(V_ext, 1)
    logger.info(f"Exterior surface: F/V = {fv_ratio:.3f} (expect ≈ 2.0 for closed 2-manifold)")

    # ── Save mesh ─────────────────────────────────────────────────────────────
    output_name = args.output_name
    mesh_npz = OUTPUT_DIR / f"{output_name}.npz"
    mesh_mat = OUTPUT_DIR / f"{output_name}.mat"

    np.savez(
        mesh_npz,
        nodes=nodes_phys,
        elements=elements,
        tissue_labels=tissue_labels,
        surface_faces=exterior_faces,
        interface_faces=interface_faces,
        surface_node_indices=np.unique(exterior_faces),
    )
    logger.info(f"Saved: {mesh_npz}")

    sio.savemat(
        mesh_mat,
        {
            "node": nodes_phys,
            "elem": np.hstack([elements, tissue_labels.reshape(-1, 1)]),
            "face_exterior": exterior_faces + 1,  # 0-based → 1-based for MATLAB
            "face_interface": interface_faces + 1,
        },
    )
    logger.info(f"Saved: {mesh_mat}")

    # ── Visualizations ────────────────────────────────────────────────────────
    if not args.no_viz:
        logger.info("\nGenerating visualizations...")
        try:
            visualize_mesh(
                nodes_phys,
                elements,
                tissue_labels,
                exterior_faces,
                interface_faces,
                VIS_DIR,
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("MESH GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  NPZ: {mesh_npz}")
    logger.info(f"  MAT: {mesh_mat}")
    if not args.no_viz:
        logger.info(f"  Visualizations: {VIS_DIR}/mesh_*.png")


if __name__ == "__main__":
    main()
