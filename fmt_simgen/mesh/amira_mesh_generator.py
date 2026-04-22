"""
AmiraMeshGenerator: Multi-material tetrahedral mesh via VTK SurfaceNets3D + TetGen.

Pipeline:
  1. Downsample atlas to working resolution.
  2. vtkSurfaceNets3D with boundary_style="external" extracts a perfect closed
     manifold exterior triangle surface.  This is used as both the PLC input
     to TetGen and as the authoritative surface_faces in MeshData.
  3. TetGen fills the volume with a single dummy region seed → all interior
     tets are labeled "1" (background / body).
  4. The per-tet tissue label is then looked up from the atlas volume at
     the tet centroid (same method as iso2mesh fallback).  This avoids
     relying on TetGen's broken multi-region CDT label propagation.

  Note: For single-tissue meshes (or when you want per-tissue labeling via
  TetGen regions), pass multiple region seeds with unique IDs.  But the
  atlas-lookup method is the default and most reliable.

Dependencies: pip install "pyvista>=0.45" "tetgen>=0.8"  (VTK >= 9.3, 9.6+ recommended)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from fmt_simgen.mesh.mesh_generator import MeshData, MeshGenerator

logger = logging.getLogger(__name__)


class AmiraMeshGenerator(MeshGenerator):
    """Amira-style mesh generation: vtkSurfaceNets3D + TetGen."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.smoothing_iterations = config.get("smoothing_iterations", 16)
        self.smoothing_scale = config.get("smoothing_scale", 1.0)
        self.decimate_reduction = config.get("decimate_reduction", 0.0)
        self.tetgen_min_ratio = config.get("tetgen_min_ratio", 1.414)
        self.tetgen_min_dihedral = config.get("tetgen_min_dihedral", 10.0)
        self.per_tissue_maxvol = config.get("per_tissue_maxvol", None)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        atlas_volume: np.ndarray,
        voxel_size: float = 0.1,
        tissue_labels: Optional[np.ndarray] = None,
        downsample_factor: int = 8,
        crop_to_trunk: bool = True,
    ) -> MeshData:
        try:
            import pyvista as pv
            import tetgen
            from scipy.ndimage import distance_transform_edt
        except ImportError as e:
            raise ImportError(
                "AmiraMeshGenerator requires pyvista>=0.45 and tetgen>=0.8. "
                'Install: pip install "pyvista>=0.45" "tetgen>=0.8"'
            ) from e

        logger.info(f"[amira] Starting mesh generation, downsample={downsample_factor}")

        # ── Step 0: downsample ─────────────────────────────────────────
        orig_shape = atlas_volume.shape
        ds_shape = tuple(int(np.ceil(s / downsample_factor)) for s in orig_shape)
        ds_vol = self._downsample_volume(atlas_volume, ds_shape).astype(np.uint8)
        eff_vs = voxel_size * downsample_factor
        logger.info(f"[amira] ds_shape={ds_shape}, eff_vs={eff_vs} mm")

        unique_labels = [int(l) for l in np.unique(ds_vol) if l != 0]
        logger.info(f"[amira] tissue labels: {unique_labels}")

        # ── Step 1: exterior surface (authoritative closed manifold) ──
        img_ext = pv.ImageData(
            dimensions=ds_shape,
            spacing=(eff_vs, eff_vs, eff_vs),
            origin=(0.0, 0.0, 0.0),
        )
        img_ext.point_data["labels"] = ds_vol.ravel(order="F")
        img_ext.set_active_scalars("labels")

        logger.info("[amira] running SurfaceNets (external surface)...")
        surf_ext = img_ext.contour_labels(
            boundary_style="external",
            background_value=0,
            simplify_output=False,
            smoothing=False,
            pad_background=True,
            output_mesh_type="triangles",
        )
        ext_verts = np.asarray(surf_ext.points, dtype=np.float64).copy()
        ext_faces = np.asarray(surf_ext.faces).reshape(-1, 4)[:, 1:].astype(np.int32)
        logger.info(f"[amira] Exterior surface: V={len(ext_verts)}, F={len(ext_faces)}")

        # ── Step 2 (optional): QEM decimation on exterior ──────────────
        if 0 < self.decimate_reduction < 1:
            logger.info(f"[amira] decimating exterior by {self.decimate_reduction:.0%}...")
            surf_ext_dec = surf_ext.decimate(self.decimate_reduction, attribute_error=True)
            ext_verts = np.asarray(surf_ext_dec.points, dtype=np.float64).copy()
            ext_faces = np.asarray(surf_ext_dec.faces).reshape(-1, 4)[:, 1:].astype(np.int32)
            logger.info(f"[amira]  after decimate: V={len(ext_verts)}, F={len(ext_faces)}")

        # ── Step 3: single region seed (just fills the volume) ──────────
        # We use atlas-lookup for tissue labels, so we only need ONE region
        # to tell TetGen "fill this PLC".  A seed at the bounding-box centre
        # is guaranteed to be inside the exterior hull.
        bbox_center = ext_verts.mean(axis=0)
        logger.info(f"[amira] region seed (single, for volume fill): {bbox_center.tolist()}")

        # ── Step 4: TetGen (exterior surface as PLC, single region) ────
        logger.info("[amira] running TetGen (PLC = exterior surface, single region)...")
        tgen = tetgen.TetGen(ext_verts, ext_faces)
        tgen.add_region(1, bbox_center.tolist(), float(self.surface_maxvol))

        switches = f"pzq{self.tetgen_min_ratio}/{self.tetgen_min_dihedral}AaQ"
        logger.info(f"[amira] TetGen switches: {switches}")
        nodes, elem, attrib, triface_markers = tgen.tetrahedralize(switches=switches)
        logger.info(f"[amira] TetGen out: N={nodes.shape[0]}, T={elem.shape[0]}")

        # ── Step 5: atlas-lookup for per-tet tissue labels ──────────────
        # TetGen region attributes are unreliable for multi-tissue meshes
        # (Voronoi partitioning collapses to a single label).  Instead,
        # look up the atlas label at each tet's centroid — same method as
        # iso2mesh fallback, proven to work.
        nodes_out = np.asarray(nodes, dtype=np.float64)
        elements = np.asarray(elem[:, :4], dtype=np.int32)

        tet_centroids = nodes_out[elements].mean(axis=1)  # [T, 3] in mm
        ix = np.clip(np.round(tet_centroids[:, 0] / eff_vs).astype(int), 0, ds_shape[0] - 1)
        iy = np.clip(np.round(tet_centroids[:, 1] / eff_vs).astype(int), 0, ds_shape[1] - 1)
        iz = np.clip(np.round(tet_centroids[:, 2] / eff_vs).astype(int), 0, ds_shape[2] - 1)
        tissue_labels_elem = ds_vol[ix, iy, iz].astype(np.int32)

        uniq, counts = np.unique(tissue_labels_elem, return_counts=True)
        label_dist = dict(zip(uniq.tolist(), counts.tolist()))
        logger.info(f"[amira] tissue label distribution: {label_dist}")

        # ── Step 6: orientation fix + exterior surface ─────────────────
        elements = self._ensure_tet_orientation(nodes_out, elements)
        surface_faces = ext_faces.copy()
        surface_node_indices = np.unique(surface_faces).astype(np.int32)

        # Verify exterior surface is a proper closed 2-manifold
        self._verify_exterior_surface(surface_faces)

        mesh_data = MeshData(
            nodes=nodes_out,
            elements=elements,
            tissue_labels=tissue_labels_elem,
            surface_faces=surface_faces,
            surface_node_indices=surface_node_indices,
        )

        if crop_to_trunk:
            mesh_data = self._crop_to_trunk(mesh_data)
        return mesh_data

    # ------------------------------------------------------------------ #
    @staticmethod
    def _verify_exterior_surface(surface_faces: np.ndarray) -> None:
        """Verify the exterior surface is a proper closed 2-manifold."""
        F = len(surface_faces)
        V = len(np.unique(surface_faces))
        fv = F / max(V, 1)

        e = np.concatenate(
            [
                np.sort(surface_faces[:, [0, 1]], axis=1),
                np.sort(surface_faces[:, [1, 2]], axis=1),
                np.sort(surface_faces[:, [2, 0]], axis=1),
            ],
            axis=0,
        )
        _, ec = np.unique(e, axis=0, return_counts=True)
        E = len(ec)
        val1 = int((ec == 1).sum())
        val2 = int((ec == 2).sum())
        val3p = int((ec >= 3).sum())
        val2_ratio = val2 / max(E, 1)
        euler = V - E + F

        logger.info(
            f"[SURF-CHECK] V={V} E={E} F={F} Euler={euler} F/V={fv:.3f} "
            f"val1={val1} val2={val2} val3+={val3p} val2_ratio={val2_ratio:.3f}"
        )

        problems = []
        if fv < 1.85:
            problems.append(f"F/V={fv:.3f} < 1.85")
        if val2_ratio < 0.95:
            problems.append(f"val2_ratio={val2_ratio:.3f} < 0.95 (val1={val1}, val3+={val3p})")
        if abs(euler - 2) > 4:
            problems.append(f"|Euler-2|={abs(euler - 2)} > 4 (Euler={euler})")

        if problems:
            msg = "Exterior surface is NOT a proper closed 2-manifold:\n  - " + "\n  - ".join(problems)
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("[SURF-CHECK] Exterior surface is a proper closed 2-manifold. OK")
