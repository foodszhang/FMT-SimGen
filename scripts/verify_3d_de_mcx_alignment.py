#!/usr/bin/env python3
"""
3D verification: DE gt_voxels vs MCX source alignment.

Renders side-by-side 3D views of:
- DE: FEM mesh surface (transparent blue) + tumor core (red, gt_voxels > 0.5)
  + surface nodes colored by measurement_b
- MCX: FEM mesh surface (transparent blue) + source pattern (yellow, value==1.0)
  + surface voxels colored by fluence

Usage:
    python scripts/verify_3d_de_mcx_alignment.py --sample sample_0000 --samples_dir data/small_uniform_5samples/samples
    python scripts/verify_3d_de_mcx_alignment.py --sample sample_0000 --samples_dir data/mesh_20k_test/samples --mesh output/shared_mesh_20k/digimouse_trunk_mesh_20k.npz
"""

import json
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pyvista as pv
import jdata as jd
from scipy import ndimage


def resolve_shared_mesh_path(mesh_path: Path = None) -> Path:
    if mesh_path and mesh_path.exists():
        return mesh_path
    candidates = [
        Path("output/shared/digimouse_trunk_mesh.npz"),
        Path("output/shared/mesh.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No shared mesh file found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_shared_mesh(mesh_path: Path = None) -> np.lib.npyio.NpzFile:
    return np.load(resolve_shared_mesh_path(mesh_path))


def load_label_volume(shared_dir: Path = None) -> np.ndarray | None:
    if shared_dir is None:
        shared_dir = Path("output/shared")
    label_path = shared_dir / "mcx_volume_trunk.bin"
    if not label_path.exists():
        return None
    vol = np.fromfile(label_path, dtype=np.uint8)
    if len(vol) != 104 * 200 * 190:
        return None
    vol = vol.reshape((104, 200, 190)).transpose(2, 1, 0)  # XYZ
    return vol


def get_body_surface_mask(shape: tuple[int, int, int], shared_dir: Path = None) -> np.ndarray:
    label_vol = load_label_volume(shared_dir)
    if label_vol is None or label_vol.shape != shape:
        nx, ny, nz = shape
        surface = np.zeros(shape, dtype=bool)
        surface[0, :, :] = True
        surface[nx - 1, :, :] = True
        surface[:, 0, :] = True
        surface[:, ny - 1, :] = True
        surface[:, :, 0] = True
        surface[:, :, nz - 1] = True
        return surface

    body = label_vol > 0
    eroded = ndimage.binary_erosion(
        body,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    return body & (~eroded)


def load_de_gt_voxels(sample_dir: Path) -> np.ndarray:
    """Load DE gt_voxels [X=190, Y=200, Z=104] at 0.2mm."""
    return np.load(sample_dir / "gt_voxels.npy")


def load_mcx_fluence(sample_dir: Path) -> np.ndarray:
    """Load MCX fluence from .jnii file, return in XYZ order."""
    jnii_files = list(sample_dir.glob("*.jnii"))
    if not jnii_files:
        raise FileNotFoundError(f"No .jnii file found in {sample_dir}")
    from fmt_simgen.mcx_projection import load_jnii_volume
    return load_jnii_volume(jnii_files[0])


def get_source_pos_and_shape(sample_dir: Path) -> tuple:
    """Read Pos and Pattern shape from MCX JSON.

    Returns (Pos_voxel, pattern_shape) where Pos_voxel is the offset
    in voxel units and pattern_shape is (nx, ny, nz) of the pattern.
    """
    mcx_json_path = sample_dir / f"{sample_dir.name}.json"
    if not mcx_json_path.exists():
        for f in sample_dir.glob("*.json"):
            d = json.load(open(f))
            if "Optode" in d:
                mcx_json_path = f
                break
    if not mcx_json_path.exists():
        return np.array([0, 0, 0]), (0, 0, 0)

    mcx_json = json.load(open(mcx_json_path))
    src_cfg = mcx_json.get("Optode", {}).get("Source", {})
    Pos = np.array(src_cfg.get("Pos", [0, 0, 0]), dtype=np.int32)
    pattern = src_cfg.get("Pattern", {})
    shape = (pattern.get("Nx", 0), pattern.get("Ny", 0), pattern.get("Nz", 0))
    return Pos, shape


def build_mesh_surface(mesh_path: Path = None) -> pv.PolyData:
    """Build PyVista PolyData from mesh.npz surface faces."""
    mesh = load_shared_mesh(mesh_path)
    nodes = mesh["nodes"].astype(np.float64)
    sf = mesh["surface_faces"].astype(np.int32)
    n_faces = len(sf)
    faces = np.column_stack([np.full(n_faces, 3), sf[:, 0], sf[:, 1], sf[:, 2]]).ravel()
    return pv.PolyData(nodes, faces)


def render_de_side(
    plotter: pv.Plotter,
    sample_dir: Path,
    mesh_path: Path = None,
    spacing: float = 0.2,
) -> None:
    """Render DE side: mesh + tumor core + surface nodes colored by b."""
    mesh_surf = build_mesh_surface(mesh_path)
    if mesh_surf is not None:
        plotter.add_mesh(mesh_surf, color="lightblue", opacity=0.2,
                        style="surface", show_edges=False, label="mesh surface")

    # Tumor core from gt_voxels
    gt_voxels = load_de_gt_voxels(sample_dir)
    tumor_core = gt_voxels > 0.5
    if tumor_core.any():
        xi, yi, zi = np.where(tumor_core)
        pts = np.column_stack([
            (xi + 0.5) * spacing,
            (yi + 0.5) * spacing,
            (zi + 0.5) * spacing,
        ])
        cloud = pv.PolyData(pts)
        plotter.add_mesh(cloud, color="red", opacity=0.85, point_size=3,
                        label=f"tumor core ({tumor_core.sum():,} vox)")

    # Surface nodes colored by measurement_b
    mesh = load_shared_mesh(mesh_path)
    surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
    b = np.load(sample_dir / "measurement_b.npy")
    if len(b) == len(surface_nodes):
        surf_cloud = pv.PolyData(surface_nodes)
        surf_cloud["b"] = b
        plotter.add_mesh(surf_cloud, color="cyan", opacity=0.5, point_size=2,
                        scalars="b", cmap="plasma", show_edges=False,
                        label="surface b values")


def render_mcx_side(
    plotter: pv.Plotter,
    sample_dir: Path,
    mesh_path: Path = None,
    spacing: float = 0.2,
) -> None:
    """Render MCX side: mesh + source pattern (binary, at Pos offset) + surface fluence."""
    mesh_surf = build_mesh_surface(mesh_path)
    if mesh_surf is not None:
        plotter.add_mesh(mesh_surf, color="lightblue", opacity=0.2,
                        style="surface", show_edges=False, label="mesh surface")

    # Source pattern — binary voxels (value == 1.0) at Pos offset
    Pos_vox, pattern_shape = get_source_pos_and_shape(sample_dir)
    src_files = list(sample_dir.glob("source-*.bin"))
    if src_files and pattern_shape[0] > 0:
        src = np.fromfile(src_files[0], dtype=np.float32)
        # Binary stored as (nz, ny, nx), need to transpose back to (nx, ny, nz)
        # where pattern_shape = (Nx, Ny, Nz) = (nx, ny, nz)
        stored_shape = (pattern_shape[2], pattern_shape[1], pattern_shape[0])  # (nz, ny, nx)
        src_3d = src.reshape(stored_shape).transpose(2, 1, 0)  # back to (nx, ny, nz)
        # For uniform source: only show voxels == 1.0 (not > 0 which would include partial)
        binary_mask = src_3d == 1.0
        if binary_mask.any():
            tx, ty, tz = np.where(binary_mask)
            # MCX mapping: pattern[tx, ty, tz] → volume[Pos_z+tx, Pos_y+ty, Pos_x+tz]
            # Physical: X=(tz+Pos_x+0.5)*sp, Y=(ty+Pos_y+0.5)*sp, Z=(tx+Pos_z+0.5)*sp
            tpts = np.column_stack([
                (tz + Pos_vox[2] + 0.5) * spacing,  # X = pattern_z + Pos_x
                (ty + Pos_vox[1] + 0.5) * spacing,  # Y = pattern_y + Pos_y
                (tx + Pos_vox[0] + 0.5) * spacing,  # Z = pattern_x + Pos_z
            ])
            tcloud = pv.PolyData(tpts)
            plotter.add_mesh(tcloud, color="yellow", opacity=0.95, point_size=4,
                            label=f"MCX source (Pos={Pos_vox})")

    # Row 1 只展示 mesh + source，不展示 surface fluence


def render_surface_fluence_side(
    plotter: pv.Plotter,
    sample_dir: Path,
    mesh_path: Path = None,
    spacing: float = 0.2,
    shared_dir: Path = None,
) -> None:
    """Render surface fluence comparison: DE b values vs MCX fluence at surface."""
    mesh = load_shared_mesh(mesh_path)
    surface_nodes = mesh["nodes"][mesh["surface_node_indices"]]
    b = np.load(sample_dir / "measurement_b.npy")

    # DE surface: colored by b value
    surf_cloud_de = pv.PolyData(surface_nodes)
    surf_cloud_de["b"] = b
    plotter.add_mesh(surf_cloud_de, opacity=0.9, scalars="b", cmap="plasma",
                     show_edges=False, label=f"DE surface b (range [{b.min():.3f}, {b.max():.3f}])")

    # MCX surface: colored by fluence at boundary voxels
    jnii_files = list(sample_dir.glob("*.jnii"))
    if jnii_files:
        data = jd.load(str(jnii_files[0]))
        nifti = data["NIFTIData"][:, :, :, 0, 0]
        fluence = np.transpose(nifti, (2, 1, 0))

        # Trunk/body surface voxels (not cube boundary)
        surf_mask = get_body_surface_mask(fluence.shape, shared_dir)

        # Only keep non-zero fluence at surface
        surf_fluence = fluence * surf_mask

        # Get surface voxel coordinates and values
        xi, yi, zi = np.where(surf_fluence > 0)
        if len(xi) > 0:
            pts = np.column_stack([
                (xi + 0.5) * spacing,
                (yi + 0.5) * spacing,
                (zi + 0.5) * spacing,
            ])
            vals = surf_fluence[xi, yi, zi]
            cloud = pv.PolyData(pts)
            cloud["fluence_log"] = np.log1p(vals.astype(np.float64))
            fmax = np.percentile(cloud["fluence_log"], 99) if len(vals) > 0 else 1.0
            plotter.add_mesh(cloud, opacity=0.8, scalars="fluence_log", cmap="hot",
                             clim=[0, fmax], show_edges=False, point_size=3,
                             label=f"MCX surface fluence (max={fluence.max():.0f})")


def render_comparison_3d(
    sample_dir: Path,
    output_path: Path,
    mesh_path: Path = None,
    spacing: float = 0.2,
    shared_dir: Path = None,
) -> None:
    """Render 3-row 2-column 3D comparison."""
    plotter = pv.Plotter(window_size=[1600, 1500], off_screen=True, shape=(3, 2))

    # Row 0, Col 0: DE tumor core
    plotter.subplot(0, 0)
    render_de_side(plotter, sample_dir, mesh_path, spacing)
    plotter.add_axes()
    plotter.add_legend()
    plotter.add_title("DE: mesh + tumor core (red) + surface b (cyan)")

    # Row 0, Col 1: MCX source
    plotter.subplot(0, 1)
    render_mcx_side(plotter, sample_dir, mesh_path, spacing)
    plotter.add_axes()
    plotter.add_legend()
    plotter.add_title("MCX: mesh + source (yellow)")

    # Row 1, Col 0: DE surface fluence (b values at mesh nodes) - NO mesh overlay
    plotter.subplot(1, 0)
    mesh = load_shared_mesh(mesh_path)
    surf_cloud_de = pv.PolyData(mesh["nodes"][mesh["surface_node_indices"]])
    b = np.load(sample_dir / "measurement_b.npy")
    surf_cloud_de["b"] = b
    plotter.add_mesh(surf_cloud_de, opacity=0.95, scalars="b", cmap="plasma", show_edges=False, point_size=5,
                     label=f"DE surface b")
    plotter.add_axes()
    plotter.add_legend()
    plotter.add_title(f"DE surface b (range [{b.min():.3f}, {b.max():.3f}])")

    # Row 1, Col 1: MCX surface fluence (at trunk/body surface voxels) - NO mesh overlay
    plotter.subplot(1, 1)

    # MCX surface fluence at boundary voxels
    jnii_files = list(sample_dir.glob("*.jnii"))
    if jnii_files:
        data = jd.load(str(jnii_files[0]))
        nifti = data["NIFTIData"][:, :, :, 0, 0]
        fluence = np.transpose(nifti, (2, 1, 0))

        surf_mask = get_body_surface_mask(fluence.shape, shared_dir)

        surf_fluence = fluence * surf_mask
        xi, yi, zi = np.where(surf_fluence > 0)
        if len(xi) > 0:
            pts = np.column_stack([
                (xi + 0.5) * spacing,
                (yi + 0.5) * spacing,
                (zi + 0.5) * spacing,
            ])
            vals = surf_fluence[xi, yi, zi]
            cloud = pv.PolyData(pts)
            cloud["fluence_log"] = np.log1p(vals.astype(np.float64))
            fmax = np.percentile(cloud["fluence_log"], 99) if len(vals) > 0 else 1.0
            plotter.add_mesh(cloud, opacity=0.9, scalars="fluence_log", cmap="hot",
                             clim=[0, fmax], show_edges=False, point_size=4,
                             label=f"MCX surface fluence")

    plotter.add_axes()
    plotter.add_legend()
    plotter.add_title("MCX surface fluence (at trunk/body surface)")

    # Row 2, Col 0: gt_voxels with mesh surface
    plotter.subplot(2, 0)
    mesh_surf = build_mesh_surface(mesh_path)
    if mesh_surf is not None:
        plotter.add_mesh(mesh_surf, color="lightblue", opacity=0.2,
                        style="surface", show_edges=False)
    gt_voxels = load_de_gt_voxels(sample_dir)
    tumor_voxels = gt_voxels > 0.5
    if tumor_voxels.any():
        xi, yi, zi = np.where(tumor_voxels)
        pts_vox = np.column_stack([
            (xi + 0.5) * spacing,
            (yi + 0.5) * spacing,
            (zi + 0.5) * spacing,
        ])
        cloud_vox = pv.PolyData(pts_vox)
        plotter.add_mesh(cloud_vox, color="red", opacity=0.85, point_size=3,
                        label=f"gt_voxels ({tumor_voxels.sum():,})")
    plotter.add_axes()
    plotter.add_title("gt_voxels (voxel grid)")

    # Row 2, Col 1: gt_nodes with mesh surface
    plotter.subplot(2, 1)
    if mesh_surf is not None:
        plotter.add_mesh(mesh_surf, color="lightblue", opacity=0.2,
                        style="surface", show_edges=False)
    mesh = load_shared_mesh(mesh_path)
    nodes = mesh["nodes"]
    gt_nodes = np.load(sample_dir / "gt_nodes.npy")
    tumor_nodes = gt_nodes > 0.5
    if tumor_nodes.any():
        pts_nodes = nodes[tumor_nodes]
        cloud_nodes = pv.PolyData(pts_nodes)
        plotter.add_mesh(cloud_nodes, color="green", opacity=0.85, point_size=3,
                        label=f"gt_nodes ({tumor_nodes.sum():,})")
    plotter.add_axes()
    plotter.add_title("gt_nodes (FEM nodes)")

    plotter.render()
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"Saved: {output_path}")


def render_ortho_slices(
    sample_dir: Path,
    output_path: Path,
    spacing: float = 0.2,
) -> None:
    """Render 3 orthographic slice sets for deeper inspection."""
    Pos_vox, pattern_shape = get_source_pos_and_shape(sample_dir)

    gt_voxels = load_de_gt_voxels(sample_dir)
    jnii_files = list(sample_dir.glob("*.jnii"))
    fluence = None
    if jnii_files:
        data = jd.load(str(jnii_files[0]))
        nifti = data["NIFTIData"][:, :, :, 0, 0]
        fluence = np.transpose(nifti, (2, 1, 0))

    # Load source for overlay
    src_files = list(sample_dir.glob("source-*.bin"))
    src_3d = None
    if src_files and pattern_shape[0] > 0:
        src = np.fromfile(src_files[0], dtype=np.float32)
        src_3d = src.reshape(pattern_shape)

    plotter = pv.Plotter(window_size=[1600, 1000], off_screen=True, shape=(2, 3))

    # DE slices: gt_voxels > 0.5
    for i, (axis_name, label) in enumerate([("X", "Sagittal"), ("Y", "Coronal"), ("Z", "Axial")]):
        plotter.subplot(0, i)
        tumor_core = gt_voxels > 0.5
        if tumor_core.any():
            x, y, z = np.where(tumor_core)
            pts = np.column_stack([
                (x + 0.5) * spacing, (y + 0.5) * spacing, (z + 0.5) * spacing
            ])
            cloud = pv.PolyData(pts)
            plotter.add_mesh(cloud, color="red", opacity=0.8, point_size=5)
        plotter.add_title(f"DE {label} (tumor >0.5)")
        plotter.add_axes()

    # MCX slices: fluence + source pattern overlay
    for i, (axis_name, label) in enumerate([("X", "Sagittal"), ("Y", "Coronal"), ("Z", "Axial")]):
        plotter.subplot(1, i)
        if fluence is not None:
            # Show non-zero fluence as point cloud, colored by value
            fluence_mask = fluence > 0
            if fluence_mask.any():
                x, y, z = np.where(fluence_mask)
                pts = np.column_stack([
                    (x + 0.5) * spacing, (y + 0.5) * spacing, (z + 0.5) * spacing
                ])
                vals = fluence[fluence_mask].ravel()
                cloud = pv.PolyData(pts)
                cloud["fluence"] = vals
                fmax = np.percentile(vals, 99) if len(vals) > 0 else 1.0
                plotter.add_mesh(cloud, opacity=0.3, scalars="fluence", cmap="hot",
                                point_size=2, clim=[0, fmax])
        # Overlay source pattern at Pos offset
        if src_3d is not None:
            binary_mask = src_3d == 1.0
            if binary_mask.any():
                tx, ty, tz = np.where(binary_mask)
                tpts = np.column_stack([
                    (tx + Pos_vox[2] + 0.5) * spacing,  # X = pattern_x + Pos_x
                    (ty + Pos_vox[1] + 0.5) * spacing,  # Y = pattern_y + Pos_y
                    (tz + Pos_vox[0] + 0.5) * spacing,  # Z = pattern_z + Pos_z
                ])
                tcloud = pv.PolyData(tpts)
                plotter.add_mesh(tcloud, color="yellow", opacity=0.9, point_size=5)
        plotter.add_title(f"MCX {label} (yellow=source, Pos={Pos_vox})")
        plotter.add_axes()

    plotter.render()
    slice_path = output_path.parent / (output_path.stem + "_slices.png")
    plotter.screenshot(str(slice_path))
    plotter.close()
    print(f"Saved slices: {slice_path}")


def main():
    parser = argparse.ArgumentParser(description="3D DE vs MCX alignment verification")
    parser.add_argument("--sample", type=str, default="sample_0000")
    parser.add_argument("--samples_dir", type=str, default="data/small_uniform_5samples/samples")
    parser.add_argument("--output_dir", type=str, default="output/verification")
    parser.add_argument("--mesh", type=str, default=None, help="Mesh file path (default: auto-detect)")
    parser.add_argument("--shared-dir", type=str, default=None, help="Shared directory for mcx_volume_trunk.bin (default: output/shared)")
    args = parser.parse_args()

    sample_dir = Path(args.samples_dir) / args.sample
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh) if args.mesh else None
    shared_dir = Path(args.shared_dir) if args.shared_dir else None

    print(f"Processing {args.sample}...")
    if mesh_path:
        print(f"Using mesh: {mesh_path}")

    gt_voxels = load_de_gt_voxels(sample_dir)
    print(f"DE gt_voxels: shape={gt_voxels.shape}, max={gt_voxels.max():.4f}")
    print(f"  tumor core (>0.5): {(gt_voxels > 0.5).sum():,} voxels")

    tp = json.load(open(sample_dir / "tumor_params.json"))
    print(f"  source_type: {tp.get('source_type')}, foci: {len(tp.get('foci', []))}")

    output_path = output_dir / f"{args.sample}_3d_de_mcx_alignment.png"
    render_comparison_3d(sample_dir, output_path, mesh_path, shared_dir=shared_dir)
    render_ortho_slices(sample_dir, output_path)


if __name__ == "__main__":
    main()
