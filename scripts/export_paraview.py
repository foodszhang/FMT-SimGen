#!/usr/bin/env python3
"""
Export ParaView/ITK-SNAP visualization files.

Output:
    output/paraview/
    ├── mesh_tissue.vtu              # Tetrahedral mesh with tissue labels
    ├── atlas_labels.nii.gz          # Atlas tissue labels (if available)
    └── sample_XXXX/
        ├── sample_XXXX_gt_mesh.vtu  # Fluorescence GT on mesh
        ├── sample_XXXX_surface_b.vtu # Surface measurements
        └── sample_XXXX_gt_voxels.nii.gz # Voxel GT (ITK-SNAP)

Usage:
    python scripts/export_paraview.py
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "paraview"


def export_mesh_tissue():
    """Export tetrahedral mesh with tissue labels."""
    logger.info("Exporting mesh_tissue.vtu...")

    import meshio

    mesh_data = np.load("output/shared/mesh.npz", allow_pickle=True)
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    tissue_labels = mesh_data["tissue_labels"]

    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", elements)],
        cell_data={"tissue_label": [tissue_labels.astype(np.float64)]},
    )
    mesh.write(str(OUTPUT_DIR / "mesh_tissue.vtu"))
    logger.info(f"  Saved: {OUTPUT_DIR / 'mesh_tissue.vtu'}")


def export_samples():
    """Export all samples to ParaView format."""
    logger.info("Exporting sample files...")

    import meshio
    import nibabel as nib

    mesh_data = np.load("output/shared/mesh.npz", allow_pickle=True)
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    surface_faces = mesh_data["surface_faces"]
    surface_node_indices = mesh_data["surface_node_indices"]

    voxel_spacing = 0.2
    mesh_center = nodes.mean(axis=0)
    origin = mesh_center - 15.0

    idx_map = np.full(nodes.shape[0], -1, dtype=np.int64)
    idx_map[surface_node_indices] = np.arange(len(surface_node_indices))
    remapped_faces = idx_map[surface_faces]

    sample_dirs = sorted(Path("data").glob("sample_*"))
    logger.info(f"  Found {len(sample_dirs)} samples")

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        sample_output = OUTPUT_DIR / sample_name
        sample_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Processing {sample_name}...")

        gt_nodes = np.load(sample_dir / "gt_nodes.npy")
        gt_voxels = np.load(sample_dir / "gt_voxels.npy")
        meas_b = np.load(sample_dir / "measurement_b.npy")

        mesh_vtu = sample_output / f"{sample_name}_gt_mesh.vtu"
        m = meshio.Mesh(
            points=nodes,
            cells=[("tetra", elements)],
            point_data={"fluorescence": gt_nodes},
        )
        m.write(str(mesh_vtu))

        surface_vtu = sample_output / f"{sample_name}_surface_b.vtu"
        surf = meshio.Mesh(
            points=nodes[surface_node_indices],
            cells=[("triangle", remapped_faces)],
            point_data={"measurement_b": meas_b},
        )
        surf.write(str(surface_vtu))

        affine = np.diag([voxel_spacing, voxel_spacing, voxel_spacing, 1.0])
        affine[:3, 3] = origin
        img = nib.Nifti1Image(gt_voxels.astype(np.float32), affine)
        nib.save(img, str(sample_output / f"{sample_name}_gt_voxels.nii.gz"))

        logger.info(f"    {mesh_vtu.name}, {surface_vtu.name}, *_gt_voxels.nii.gz")


def export_atlas():
    """Export atlas labels if available."""
    logger.info("Exporting atlas_labels.nii.gz...")

    atlas_path = Path("output/shared/atlas_full.npz")
    if not atlas_path.exists():
        logger.warning("  atlas_full.npz not found, skipping atlas export")
        return

    import nibabel as nib

    atlas_data = np.load(atlas_path, allow_pickle=True)
    vol = atlas_data.get("tissue_labels", atlas_data.get("original_labels"))
    voxel_size = float(atlas_data["voxel_size"])

    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, str(OUTPUT_DIR / "atlas_labels.nii.gz"))
    logger.info(f"  Saved: {OUTPUT_DIR / 'atlas_labels.nii.gz'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ParaView Export Script")
    logger.info("=" * 60)

    export_mesh_tissue()
    export_samples()
    export_atlas()

    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    logger.info("\nFile listing:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            logger.info(f"  {f.relative_to(OUTPUT_DIR)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()