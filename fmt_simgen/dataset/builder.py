"""
DatasetBuilder: Orchestrate dataset generation pipeline.

Pipeline stages:
1. build_shared_assets(): Generate mesh and system matrix (once)
2. build_samples(): Generate N samples (tumor → dual sampling → forward)

Each sample produces:
- measurement_b.npy: Surface measurements [N_d]
- gt_nodes.npy: Ground truth at FEM nodes [N_n]
- gt_voxels.npy: Ground truth at voxels [Nx × Ny × Nz]
- tumor_params.json: Tumor parameters for reproducibility
"""

import gc
import resource
import sys
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from fmt_simgen.atlas.digimouse import DigimouseAtlas
from fmt_simgen.mesh.mesh_generator import MeshGenerator, MeshData
from fmt_simgen.physics.optical_params import OpticalParameterManager
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.tumor.tumor_generator import TumorGenerator
from fmt_simgen.sampling.dual_sampler import DualSampler, VoxelGridConfig
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOXEL_SIZE_MM


@dataclass
class SampleOutput:
    """Container for single sample output."""

    measurement_b: np.ndarray
    gt_nodes: np.ndarray
    gt_voxels: np.ndarray
    tumor_params: Dict
    visible_mask: Optional[np.ndarray] = None


class DatasetBuilder:
    """Build FMT simulation datasets."""

    def __init__(self, config: Dict):
        """Initialize dataset builder.

        Parameters
        ----------
        config : Dict
            Full configuration dictionary.
        """
        self.config = config

        self.atlas_config = config.get("atlas", {})
        self.mesh_config = config.get("mesh", {})
        self.physics_config = config.get("physics", {})
        self.tumor_config = config.get("tumor", {})
        self.dataset_config = config.get("dataset", {})
        self.quality_config = config.get("quality_filter", {})

        self.atlas: Optional[DigimouseAtlas] = None
        self.mesh_generator: Optional[MeshGenerator] = None
        self.opt_manager: Optional[OpticalParameterManager] = None
        self.fem_solver: Optional[FEMSolver] = None
        self.tumor_generator: Optional[TumorGenerator] = None
        self.camera: Optional[TurntableCamera] = None

        self._mesh_data: Optional[MeshData] = None
        self._visible_mask: Optional[np.ndarray] = None
        self._visible_indices: Optional[np.ndarray] = None

    def build_shared_assets(self, force_regenerate: bool = False) -> Dict[str, Path]:
        """Generate mesh and system matrix (once per dataset).

        Parameters
        ----------
        force_regenerate : bool, default False
            If True, regenerate even if files exist.

        Returns
        -------
        Dict[str, Path]
            Paths to generated asset files.
        """
        default_output = Path("output/shared")
        mesh_path = Path(self.mesh_config.get("output_path", str(default_output)))
        mesh_file = mesh_path / "mesh.npz"
        matrix_file = mesh_path / "system_matrix"

        # ── Step 0: Atlas nodes for TumorGenerator KDTree (before any rebase) ─
        atlas_nodes: np.ndarray | None = None

        if (
            not force_regenerate
            and mesh_file.exists()
            and matrix_file.with_suffix(".M.npz").exists()
        ):
            # ── Existing assets (assets/mesh/ already populated) ─────────────────
            print(f"Shared assets already exist at {mesh_path}, loading.")
            self._load_mesh_data(str(mesh_file))   # loads + rebases nodes in-place
            self._setup_optical_manager()
            self._setup_fem_solver()               # use trunk-local nodes (A is translation-invariant)
            # Save rebased mesh to mesh directory
            np.savez(
                mesh_file,
                nodes=self._mesh_data.nodes,
                elements=self._mesh_data.elements,
                tissue_labels=self._mesh_data.tissue_labels,
                surface_faces=self._mesh_data.surface_faces,
                surface_node_indices=self._mesh_data.surface_node_indices,
            )
            # Also save to output/shared/ where DU2Vox expects the mesh
            shared_mesh = Path("output/shared/mesh.npz")
            shared_mesh.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                shared_mesh,
                nodes=self._mesh_data.nodes,
                elements=self._mesh_data.elements,
                tissue_labels=self._mesh_data.tissue_labels,
                surface_faces=self._mesh_data.surface_faces,
                surface_node_indices=self._mesh_data.surface_node_indices,
            )
            atlas_nodes = self._mesh_data.nodes.copy()
            matrix_file = self.fem_solver.save_system_matrix(str(mesh_path / "system_matrix"))
            self._write_frame_manifest()
            self._verify_frame_consistency()
            print(f"Shared assets loaded (mesh re-saved in trunk-local frame)")
            return {"mesh": shared_mesh, "matrix": matrix_file}

        if not force_regenerate and default_output.exists():
            existing_mesh = default_output / "mesh.npz"
            existing_matrix = default_output / "system_matrix.M.npz"
            if existing_mesh.exists() and existing_matrix.exists():
                print(f"Loading existing assets from {default_output}")
                self._load_mesh_data(str(existing_mesh))
                self._setup_optical_manager()
                self._setup_fem_solver()
                # Save rebased mesh alongside existing matrix
                np.savez(
                    existing_mesh,
                    nodes=self._mesh_data.nodes,
                    elements=self._mesh_data.elements,
                    tissue_labels=self._mesh_data.tissue_labels,
                    surface_faces=self._mesh_data.surface_faces,
                    surface_node_indices=self._mesh_data.surface_node_indices,
                )
                atlas_nodes = self._mesh_data.nodes.copy()
                matrix_file = self.fem_solver.save_system_matrix(
                    str(existing_matrix.parent / "system_matrix")
                )
                self._write_frame_manifest()
                self._verify_frame_consistency()
                return {
                    "mesh": existing_mesh,
                    "matrix": existing_matrix.parent / "system_matrix",
                }

        # ── Fresh generation ─────────────────────────────────────────────────
        print("Building shared assets...")

        self._setup_atlas()
        self._setup_mesh_generator()
        self._setup_optical_manager()

        print("  Generating tetrahedral mesh...")
        self._mesh_data = self.mesh_generator.generate(
            atlas_volume=self.atlas.volume, voxel_size=self.atlas.voxel_size
        )
        # self._mesh_data.nodes is atlas-frame; save this frame before rebase
        atlas_nodes = self._mesh_data.nodes.copy()

        print("  Assembling FEM system matrix...")
        self._setup_fem_solver()
        self.fem_solver.assemble_system_matrix()

        # ── Rebase to trunk-local and save mesh.npz in trunk-local frame ─────
        mcx_cfg = self.config.get("mcx", {})
        trunk_offset_atlas = np.array(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]),
                                      dtype=np.float64)
        self._mesh_data.nodes = (
            self._mesh_data.nodes.astype(np.float64) - trunk_offset_atlas
        ).astype(self._mesh_data.nodes.dtype)
        mesh_file = self.mesh_generator.save(self._mesh_data, "mesh")

        matrix_file = self.fem_solver.save_system_matrix(
            str(mesh_path / "system_matrix")
        )
        self._write_frame_manifest()
        print(f"Shared assets saved to {mesh_path} (mesh in trunk-local mm)")

        # ── U5: Frame consistency check ─────────────────────────────────────────
        self._verify_frame_consistency()

        return {"mesh": mesh_file, "matrix": matrix_file}

    def _load_mesh_data(self, mesh_path: str) -> None:
        """Load mesh data from npz file and rebase to trunk-local if needed.

        After loading, self._mesh_data.nodes is guaranteed to be in trunk-local frame.
        Manifest is updated at output/shared/frame_manifest.json (canonical location).
        The caller (build_shared_assets) is responsible for saving the rebased mesh
        to output/shared/mesh.npz (where DU2Vox expects it).
        """
        mesh_path = Path(mesh_path)
        mcx_cfg = self.config.get("mcx", {})
        trunk_offset = np.array(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]),
                               dtype=np.float64)

        data = np.load(mesh_path, allow_pickle=True)
        nodes = data["nodes"].astype(np.float64)

        # Canonical manifest location: output/shared/ (DU2Vox shared_dir)
        manifest_path = Path("output/shared/frame_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            mesh_frame = manifest.get("fem_mesh", {}).get("frame", "atlas_corner_mm")
        else:
            mesh_frame = "atlas_corner_mm"  # assume old format

        # Always copy manifest to mesh directory (assets/mesh/) for build_samples
        mesh_dir_manifest = mesh_path.parent / "frame_manifest.json"
        if manifest_path.exists() and manifest_path != mesh_dir_manifest:
            manifest = json.loads(manifest_path.read_text())
            if mesh_frame == "atlas_corner_mm":
                # Rebase to trunk-local
                nodes = nodes - trunk_offset
                manifest["fem_mesh"]["frame"] = "mcx_trunk_local_mm"
                manifest["fem_mesh"]["bbox_world_mm"]["min"] = (
                    nodes.min(axis=0).tolist()
                )
                manifest["fem_mesh"]["bbox_world_mm"]["max"] = (
                    nodes.max(axis=0).tolist()
                )
                manifest_path.write_text(json.dumps(manifest, indent=2))
            # Copy to mesh directory (always, to keep them in sync)
            import shutil
            shutil.copy(manifest_path, mesh_dir_manifest)

        self._mesh_data = MeshData(
            nodes=nodes.astype(data["nodes"].dtype),
            elements=data["elements"],
            surface_faces=data["surface_faces"],
            tissue_labels=data["tissue_labels"],
            surface_node_indices=data["surface_node_indices"],
        )
        print(
            f"  Loaded mesh: {self._mesh_data.nodes.shape[0]} nodes, "
            f"{self._mesh_data.elements.shape[0]} elements (trunk-local)"
        )

    def _load_fem_solver(self, matrix_prefix: str) -> None:
        """Load FEM solver with precomputed matrices.

        Parameters
        ----------
        matrix_prefix : str
            Path prefix for matrix files (without .K.npz etc).
        """
        matrix_prefix = Path(matrix_prefix)
        self.fem_solver = FEMSolver(
            nodes=self._mesh_data.nodes,
            elements=self._mesh_data.elements,
            surface_faces=self._mesh_data.surface_faces,
            tissue_labels=self._mesh_data.tissue_labels,
            opt_params_manager=self.opt_manager,
        )
        self.fem_solver.load_system_matrix(str(matrix_prefix))
        print(f"  Loaded system matrices from {matrix_prefix}")

    def _verify_frame_consistency(self) -> None:
        """U5: Verify mesh and mcx_volume are in the same trunk-local frame.

        Checks that mesh node bbox and mcx_volume body bbox agree within 2mm
        per axis. This prevents frame regress silently.
        """
        from fmt_simgen.frame_contract import (
            VOXEL_SIZE_MM, TRUNK_SIZE_MM, assert_in_trunk_bbox,
        )
        import logging
        logger = logging.getLogger(__name__)

        # Load mcx_volume_trunk.bin (ZYX order)
        mcx_bin = Path("output/shared/mcx_volume_trunk.bin")
        if not mcx_bin.exists():
            logger.warning("mcx_volume_trunk.bin not found — skipping frame check")
            return

        mcx_raw = np.fromfile(mcx_bin, dtype=np.uint8)
        mcx_zyx = mcx_raw.reshape((104, 200, 190))
        mcx_xyz = mcx_zyx.transpose(2, 1, 0)  # → (190, 200, 104) XYZ

        # mcx body voxel indices → trunk-local mm
        mcx_body_idx = np.argwhere(mcx_xyz > 0)  # (M, 3) XYZ voxel indices
        mcx_body_mm = mcx_body_idx.astype(np.float64) * VOXEL_SIZE_MM

        mesh_nodes = self._mesh_data.nodes  # (N, 3) trunk-local mm

        logger.info("=== U5 Frame Consistency Check ===")
        for axis, name in enumerate("XYZ"):
            m_lo = float(mesh_nodes[:, axis].min())
            m_hi = float(mesh_nodes[:, axis].max())
            v_lo = float(mcx_body_mm[:, axis].min())
            v_hi = float(mcx_body_mm[:, axis].max())
            diff_lo = abs(m_lo - v_lo)
            diff_hi = abs(m_hi - v_hi)
            logger.info(
                f"  {name}: mesh [{m_lo:.2f}, {m_hi:.2f}] vs "
                f"mcx [{v_lo:.2f}, {v_hi:.2f}] "
                f"(diff_lo={diff_lo:.2f}, diff_hi={diff_hi:.2f})"
            )
            assert diff_lo < 2.0 and diff_hi < 2.0, (
                f"Frame mismatch on {name}: "
                f"mesh=[{m_lo:.2f},{m_hi:.2f}] mcx=[{v_lo:.2f},{v_hi:.2f}]"
            )

        assert_in_trunk_bbox(mesh_nodes, tol_mm=3.0)
        assert_in_trunk_bbox(mcx_body_mm, tol_mm=0.5)
        logger.info("✅ Frame consistency verified: mesh ↔ mcx_volume aligned within 2mm")

    def _write_frame_manifest(self) -> None:
        """Write frame_manifest.json with authoritative frame metadata.

        Manifest lives in output/shared/ (dataset shared directory).
        mesh.npz is saved in mcx_trunk_local_mm frame (rebase done at write time).
        Also syncs to assets/mesh/ (where the mesh files live).
        """
        mcx_cfg = self.config.get("mcx", {})
        trunk_offset = list(mcx_cfg.get("trunk_offset_mm", [0, 30, 0]))
        vs_zyx = mcx_cfg.get("volume_shape", [104, 200, 190])
        vx = float(mcx_cfg.get("voxel_size_mm", VOXEL_SIZE_MM))
        bbox_max = [vs_zyx[2] * vx, vs_zyx[1] * vx, vs_zyx[0] * vx]

        # self._mesh_data.nodes is trunk-local (rebase done in build_shared_assets)
        nodes_trunk = self._mesh_data.nodes.astype(np.float64)

        # Always use output/shared/ as the canonical manifest location
        shared_dir = Path("output/shared")
        manifest_path = shared_dir / "frame_manifest.json"

        # Preserve existing frame if manifest already updated by load_mesh_data
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            fem_mesh_frame = existing.get("fem_mesh", {}).get(
                "frame", "atlas_corner_mm"
            )
        else:
            fem_mesh_frame = "mcx_trunk_local_mm"

        voxel_spacing = self.dataset_config.get("voxel_spacing", VOXEL_SIZE_MM)
        roi_size = self.dataset_config.get("voxel_grid_roi_size_mm", 30.0)
        roi_center = np.array(bbox_max) / 2
        offset = (roi_center - roi_size / 2).tolist()
        shape_gt = [int(np.ceil(roi_size / voxel_spacing))] * 3

        manifest = {
            "version": 1,
            "world_frame": "mcx_trunk_local_mm",
            "atlas_to_world_offset_mm": trunk_offset,
            "mcx_volume": {
                "shape_xyz": [vs_zyx[2], vs_zyx[1], vs_zyx[0]],
                "voxel_size_mm": vx,
                "bbox_world_mm": {"min": [0, 0, 0], "max": bbox_max},
            },
            "fem_mesh": {
                "file": "mesh.npz",
                "frame": fem_mesh_frame,  # preserve load_mesh_data update
                "n_nodes": int(nodes_trunk.shape[0]),
                "bbox_world_mm": {
                    "min": nodes_trunk.min(0).tolist(),
                    "max": nodes_trunk.max(0).tolist(),
                },
            },
            "voxel_grid_gt": {
                "shape": shape_gt,
                "spacing_mm": voxel_spacing,
                "offset_world_mm": offset,
                "frame": "mcx_trunk_local_mm",
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[FRAME] Wrote {manifest_path}")

        # Also sync to assets/mesh/ where step0b originally placed it
        import shutil
        mesh_dir = Path(self.mesh_config.get("output_path", "output/shared"))
        mesh_dir.mkdir(parents=True, exist_ok=True)
        dest = mesh_dir / "frame_manifest.json"
        if manifest_path.resolve() != dest.resolve():
            shutil.copy(manifest_path, dest)

    def build_samples(self, num_samples: Optional[int] = None, start_index: int = 0) -> None:
        """Generate tumor samples with measurements.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate. Uses config default if None.
        start_index : int, optional
            Starting sample index. Existing complete samples at or after
            start_index are skipped (supports batched subprocess runs).
            Default 0.
        """
        if num_samples is None:
            num_samples = self.dataset_config.get("num_samples", 50)

        self._ensure_setup()

        voxel_spacing = self.dataset_config.get("voxel_spacing", VOXEL_SIZE_MM)
        roi_size = self.dataset_config.get("voxel_grid_roi_size_mm", 30.0)

        # Experiment-isolated output: data/{experiment_name}/samples/
        experiment_name = self.dataset_config.get("experiment_name", "default")
        base_output = Path(self.dataset_config.get("output_path", "data/"))
        experiment_output = base_output / experiment_name
        samples_output = experiment_output / "samples"

        samples_output.mkdir(parents=True, exist_ok=True)

        # ================== [FIX v3] 统一到 mcx_trunk_local_mm frame ==================
        # 1) 读 MCX frame 元数据
        mcx_cfg = self.config.get("mcx", {})
        if not mcx_cfg:
            raise ValueError("[FIX v3] config.mcx 必填 — frame 对齐依赖它")
        trunk_offset_atlas = np.array(mcx_cfg["trunk_offset_mm"], dtype=np.float64)  # [0,30,0]
        trunk_voxel_size = float(mcx_cfg["voxel_size_mm"])  # 0.2
        # volume_shape 是 ZYX 顺序
        vs_zyx = mcx_cfg["volume_shape"]
        trunk_size_mm = np.array(
            [vs_zyx[2] * trunk_voxel_size,  # X
             vs_zyx[1] * trunk_voxel_size,  # Y
             vs_zyx[0] * trunk_voxel_size],  # Z
            dtype=np.float64,
        )  # ≈ [38.0, 40.0, 20.8]

        # 2) Check if mesh needs rebase (load_mesh_data already rebased if needed)
        # After load_mesh_data: mesh is trunk-local if manifest says mcx_trunk_local_mm,
        # or atlas if manifest is missing/outdated (old saved mesh).
        mesh_path = Path(self.mesh_config.get("output_path", "output/shared"))
        manifest_path = mesh_path / "frame_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            mesh_frame = manifest.get("fem_mesh", {}).get("frame", "atlas_corner_mm")
        else:
            mesh_frame = "atlas_corner_mm"

        if mesh_frame == "atlas_corner_mm":
            # Defensive rebase for legacy meshes (load_mesh_data already rebased + saved
            # for manifest=atlas_corner_mm; this is a fallback for edge cases)
            nodes_trunk = self._mesh_data.nodes.astype(np.float64) - trunk_offset_atlas
            self._mesh_data.nodes = nodes_trunk.astype(self._mesh_data.nodes.dtype)
            print(f"  [FRAME] Warning: legacy atlas-frame mesh rebase in build_samples")
        else:
            # Mesh already trunk-local (load_mesh_data did it)
            nodes_trunk = self._mesh_data.nodes.astype(np.float64)

        # 硬断言：trunk-local 节点应有 ≥50% 落在 MCX volume 内
        in_mcx = np.all(
            (nodes_trunk >= -1.0) & (nodes_trunk <= trunk_size_mm + 1.0), axis=1
        )
        inside_ratio = float(in_mcx.mean())
        print(f"  [FRAME] Mesh frame={mesh_frame}, {inside_ratio*100:.1f}% nodes inside MCX bbox.")
        assert inside_ratio >= 0.50, (
            f"[FIX v3] 只有 {inside_ratio*100:.1f}% 节点在 MCX bbox 内，"
            f"trunk_offset_mm 可能不匹配（正常 full-body mesh 约 57-60%）"
        )

        # 3) ROI：以 MCX volume 几何中心为中心的 roi_size^3 立方
        roi_center = trunk_size_mm / 2.0  # ≈ [19, 20, 10.4]
        roi_half = roi_size / 2.0
        node_min = roi_center - roi_half  # trunk-local mm
        node_max = roi_center + roi_half
        shape = np.ceil((node_max - node_min) / voxel_spacing).astype(int)

        voxel_grid_config = VoxelGridConfig(
            shape=tuple(shape),
            spacing=voxel_spacing,
            offset=node_min,  # ← trunk-local frame
        )

        # 4) mesh_bbox 也用 rebased nodes（tumor placement 不受影响，因为 tumor_generator 内部另有 atlas-frame 逻辑，见 1.2）
        mesh_min = nodes_trunk.min(axis=0)
        mesh_max = nodes_trunk.max(axis=0)
        mesh_bbox = {"min": mesh_min.tolist(), "max": mesh_max.tolist()}

        # 5) TumorGenerator：传入 trunk_offset 让它把 center 转成 trunk-local
        self.tumor_generator = TumorGenerator(
            config=self.tumor_config,
            atlas=self.atlas,
            mesh_bbox=mesh_bbox,
            mesh_nodes=nodes_trunk,  # ← trunk-local
            tissue_labels=self._mesh_data.tissue_labels,
            elements=self._mesh_data.elements,
            organ_constraint_disabled=self.tumor_config.get(
                "organ_constraint_disabled", False
            ),
            trunk_offset_mm=trunk_offset_atlas,
            mcx_bbox_mm=(np.zeros(3), trunk_size_mm),
            gt_offset_mm=voxel_grid_config.offset,  # gt_voxels bounding box
            gt_shape=voxel_grid_config.shape,  # gt_voxels shape
            gt_spacing_mm=voxel_spacing,  # gt_voxels spacing (used in organ constraint)
        )

        dual_sampler = DualSampler(
            nodes=nodes_trunk, voxel_grid_config=voxel_grid_config
        )
        # ============================================================================

        # ============ Setup TurntableCamera for visible surface filtering ============
        view_config = self.config.get("view_config", {})
        if view_config.get("angles"):
            self.camera = TurntableCamera(view_config)
            node_normals = self.camera.compute_surface_normals(
                self._mesh_data.nodes, self._mesh_data.surface_faces
            )
            visible_per_angle = self.camera.get_all_visible_nodes_per_angle(
                self._mesh_data.nodes, node_normals
            )
            # Union of visible nodes across all angles
            visible_all = set()
            for angle, vis_nodes in visible_per_angle.items():
                visible_all.update(vis_nodes.tolist())
            # visible_all contains absolute node indices; map to surface positions
            surface_node_set = set(self._mesh_data.surface_node_indices.tolist())
            surface_pos_map = {
                node: pos for pos, node in enumerate(self._mesh_data.surface_node_indices)
            }
            visible_surface_pos = sorted(
                surface_pos_map[n] for n in visible_all if n in surface_node_set
            )
            self._visible_indices = np.array(visible_surface_pos, dtype=np.int32)
            total_surface = len(self._mesh_data.surface_node_indices)
            n_visible = len(self._visible_indices)
            print(f"  ViewConfig: {n_visible}/{total_surface} surface nodes visible "
                  f"({100*n_visible/total_surface:.1f}%) across {len(view_config['angles'])} angles")
            # Build visible_mask [S] where True = visible at any angle
            self._visible_mask = np.zeros(total_surface, dtype=bool)
            self._visible_mask[self._visible_indices] = True
            # Print per-angle stats
            for angle in sorted(visible_per_angle.keys()):
                n_angle = len(visible_per_angle[angle])
                print(f"    Angle {angle:4d}°: {n_angle:4d} visible "
                      f"({100*n_angle/total_surface:.1f}%)")
            # Save visible_mask.npy to output/shared/ (shared across all samples)
            shared_dir = Path("output/shared")
            shared_dir.mkdir(parents=True, exist_ok=True)
            np.save(shared_dir / "visible_mask.npy", self._visible_mask)
            print(f"  visible_mask saved to {shared_dir / 'visible_mask.npy'}")
        else:
            self.camera = None
            self._visible_mask = None
            self._visible_indices = None

        # Quality filter settings
        filter_enabled = self.quality_config.get("enabled", False)
        min_b_max = self.quality_config.get("min_b_max", 0.001)
        min_gt_frac = self.quality_config.get("min_gt_nonzero_frac", 0.001)
        min_gt_nonzero_count = self.quality_config.get("min_gt_nonzero_count", 0)
        max_retries = self.quality_config.get("max_retries", 5)

        # Only accumulate metadata for manifest, not full SampleOutput objects
        samples_metadata: List[Dict] = []
        rejected_count = 0
        skipped_count = 0

        # ============ Pre-compute stratified sample plan ============
        # Assign depth_tier per sample based on depth_distribution weights,
        # then assign num_foci and depth_mm within each tier.
        depth_config = self.tumor_config["depth_distribution"]
        tiers = list(depth_config.keys())
        weights = [depth_config[t]["weight"] for t in tiers]
        depth_ranges = {t: depth_config[t]["range"] for t in tiers}

        foci_dist = self.tumor_config["num_foci_distribution"]
        foci_vals = list(foci_dist.keys())
        foci_probs = list(foci_dist.values())

        tier_counts = np.random.choice(tiers, size=num_samples, p=weights)
        samples_plan: List[tuple] = []
        for i in range(num_samples):
            tier = tier_counts[i]
            n_foci = int(np.random.choice(foci_vals, p=foci_probs))
            lo, hi = depth_ranges[tier]
            depth_mm = float(np.random.uniform(lo, hi))
            samples_plan.append((n_foci, depth_mm, tier))

        # Generate until we have exactly num_samples valid saves.
        # When organ constraint or quality filter rejects parameters, we draw
        # fresh ones rather than retrying the same (deterministically failing) slot.
        saved_count = start_index
        plan_idx = start_index  # position in samples_plan (skip already-generated plan entries)

        # Required files for a complete sample
        _required_files = ["measurement_b.npy", "gt_nodes.npy", "gt_voxels.npy", "tumor_params.json"]

        while saved_count < num_samples:
            sample_dir = samples_output / f"sample_{saved_count:04d}"

            # Skip existing complete samples (supports resume from partial runs)
            if sample_dir.is_dir():
                missing = [fn for fn in _required_files if not (sample_dir / fn).exists()]
                if not missing:
                    print(f"  [SKIP] sample_{saved_count:04d} already exists, skipping")
                    saved_count += 1
                    continue
                # Incomplete directory: remove and regenerate
                import shutil
                print(f"  [REBUILD] sample_{saved_count:04d} incomplete ({missing}), removing")
                shutil.rmtree(sample_dir)

            # Draw parameters: use plan until exhausted, then draw from same distribution
            if plan_idx < len(samples_plan):
                n_foci, depth_mm, depth_tier = samples_plan[plan_idx]
                plan_idx += 1
            else:
                tier = np.random.choice(tiers, p=weights)
                n_foci = int(np.random.choice(foci_vals, p=foci_probs))
                lo, hi = depth_ranges[tier]
                depth_mm = float(np.random.uniform(lo, hi))
                depth_tier = tier

            # Memory diagnostic every sample
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"  [MEM] sample {saved_count}/{num_samples} | RSS = {rss_mb:.0f} MB", flush=True)

            print(f"  Generating sample {saved_count + 1}/{num_samples}...", flush=True)

            last_tumor_sample = None
            last_gt_nodes = None
            last_gt_voxels = None
            last_measurement_b = None
            last_b_max = 0.0
            last_gt_nonzero_frac = 0.0
            last_gt_nonzero_count = 0

            # Track organ_failure separately: if organ fails even once, params are bad
            organ_failed_all_attempts = True
            quality_passed = False

            for attempt in range(max_retries + 1):
                tumor_sample = self.tumor_generator.generate_sample(
                    num_foci=n_foci,
                    depth_mm=depth_mm,
                    depth_tier=depth_tier,
                )

                # Compute gt_nodes and measurement_b first (small, ~100KB total)
                gt_nodes = dual_sampler.sample_to_nodes(tumor_sample)
                measurement_b = self.fem_solver.forward(gt_nodes)

                b_max = float(np.max(np.abs(measurement_b)))
                gt_nonzero_frac = float(np.count_nonzero(gt_nodes)) / len(gt_nodes)
                gt_nonzero_count = int(np.count_nonzero(gt_nodes))

                b_ok = b_max >= min_b_max
                frac_ok = gt_nonzero_frac >= min_gt_frac
                count_ok = gt_nonzero_count >= min_gt_nonzero_count
                organ_ok = getattr(tumor_sample, '_organ_constraint_passed', True)

                quality_this_attempt = b_ok and frac_ok and count_ok
                organ_failed_all_attempts = organ_failed_all_attempts and not organ_ok

                if (not filter_enabled or quality_this_attempt) and organ_ok:
                    # Only compute gt_voxels (13MB) after quality check passes
                    gt_voxels = dual_sampler.sample_to_voxels(tumor_sample)
                    quality_passed = True
                    last_tumor_sample = tumor_sample
                    last_gt_nodes = gt_nodes
                    last_gt_voxels = gt_voxels
                    # Apply visible node mask: zero out invisible surface nodes
                    if self._visible_mask is not None:
                        measurement_b = measurement_b[self._visible_mask]  # [N_surface] -> [V_visible]
                    last_measurement_b = measurement_b
                    last_b_max = b_max
                    last_gt_nonzero_frac = gt_nonzero_frac
                    last_gt_nonzero_count = gt_nonzero_count
                    break

                # Failed: release memory before retry
                del gt_nodes, measurement_b, tumor_sample
                gc.collect()
                if attempt < max_retries:
                    reason = "organ constraint" if not organ_ok else "quality filter"
                    print(f"    [FILTER] Sample rejected ({reason}), retrying {attempt + 1}/{max_retries}")
                else:
                    reason = "organ constraint" if not organ_ok else "quality filter"
                    print(f"    [FILTER] Sample rejected ({reason}), retries exhausted")

            # Gate saving: organ failure = always abandon (don't retry same params)
            # quality failure with organ pass = abandon params (retry with fresh)
            organ_ok = getattr(last_tumor_sample, '_organ_constraint_passed', True)
            if filter_enabled and (organ_failed_all_attempts or not quality_passed):
                rejected_count += 1
                skipped_count += 1
                msg = "organ constraint" if organ_failed_all_attempts else "quality filter"
                print(f"    [WARNING] Sample failed ({msg}), abandoning and retrying with new parameters")
                continue

            tumor_params = last_tumor_sample.to_dict()
            tumor_params["organ_constraint_passed"] = organ_ok

            sample = SampleOutput(
                measurement_b=last_measurement_b,
                gt_nodes=last_gt_nodes,
                gt_voxels=last_gt_voxels,
                tumor_params=tumor_params,
                visible_mask=self._visible_mask,
            )
            self._save_sample(sample, samples_output, saved_count)

            # Validate using in-memory sample object (no disk re-read)
            # Only validate first 10 samples to avoid O(n_nodes) distance computation overhead
            if saved_count < 10:
                self._validate_sample(sample, self._mesh_data.nodes)

            samples_metadata.append({
                "id": f"sample_{saved_count:04d}",
                "num_foci": tumor_params["num_foci"],
                "depth_tier": tumor_params.get("depth_tier", "unknown"),
                "depth_mm": tumor_params.get("depth_mm"),
                "b_max": last_b_max,
                "b_mean": float(np.mean(np.abs(last_measurement_b))),
                "gt_max": float(np.max(last_gt_nodes)),
                "gt_nonzero_count": last_gt_nonzero_count,
                "gt_nonzero_frac": last_gt_nonzero_frac,
                "has_gt_voxels": True,
            })

            saved_count += 1

            del sample, last_measurement_b, last_gt_nodes, last_gt_voxels, last_tumor_sample
            gc.collect()

            # Force glibc to return freed memory to OS (fights malloc fragmentation)
            import ctypes
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        print(f"Dataset saved to {experiment_output}")

        if rejected_count > 0:
            print(f"[WARNING] {rejected_count} samples did not pass quality filter")
        if skipped_count > 0:
            print(f"[WARNING] {skipped_count} samples were skipped due to generation failure")

        # Generate train/val splits (80/20)
        self._generate_splits(experiment_output, [s["id"] for s in samples_metadata])

        # Generate manifest from metadata only
        self._generate_manifest(experiment_output, samples_metadata)

        # [FIX v3] 写 frame_manifest.json（在 mesh rebase 完成之后）
        self._write_frame_manifest()

    def _generate_splits(self, experiment_output: Path, sample_ids: List[str]) -> None:
        """Generate train/val split files.

        Parameters
        ----------
        experiment_output : Path
            Path to experiment output directory.
        sample_ids : List[str]
            List of sample IDs that were actually generated.
        """
        sample_ids = list(sample_ids)
        random.seed(42)
        random.shuffle(sample_ids)
        split_idx = int(len(sample_ids) * 0.8)
        train_ids = sorted(sample_ids[:split_idx])
        val_ids = sorted(sample_ids[split_idx:])

        splits_dir = experiment_output / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        with open(splits_dir / "train.txt", "w") as f:
            f.write("\n".join(train_ids) + "\n")
        with open(splits_dir / "val.txt", "w") as f:
            f.write("\n".join(val_ids) + "\n")
        print(f"Splits: {len(train_ids)} train, {len(val_ids)} val")

    def _generate_manifest(self, experiment_output: Path, samples_metadata: List[Dict]) -> None:
        """Generate dataset manifest JSON.

        Parameters
        ----------
        experiment_output : Path
            Path to experiment output directory.
        samples_metadata : List[Dict]
            List of sample metadata dicts.
        """
        manifest = {
            "experiment_name": self.dataset_config.get("experiment_name", "default"),
            "source_type": self.tumor_config.get("source_type", "gaussian"),
            "num_samples": len(samples_metadata),
            "mesh_nodes": int(self._mesh_data.nodes.shape[0]),
            "mesh_surface_nodes": int(len(self._mesh_data.surface_node_indices)),
            "visible_surface_nodes": int(np.sum(self._visible_mask)) if self._visible_mask is not None else None,
            "visible_ratio": float(np.mean(self._visible_mask)) if self._visible_mask is not None else None,
            "config": self.config,
            "samples": samples_metadata,
        }

        manifest_path = experiment_output / "dataset_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Manifest saved to {manifest_path}")

    def _setup_atlas(self) -> None:
        """Setup atlas loader."""
        atlas_path = self.atlas_config.get("path")
        if atlas_path is None:
            raise ValueError("Atlas path not specified in config.")

        self.atlas = DigimouseAtlas(atlas_path).load()

        merge_rules = self.atlas_config.get("tissue_merge", {})
        if merge_rules:
            self.atlas.merge_tissues(merge_rules, inplace=True)

    def _setup_mesh_generator(self) -> None:
        """Setup mesh generator."""
        self.mesh_generator = MeshGenerator(self.mesh_config)

    def _setup_optical_manager(self) -> None:
        """Setup optical parameter manager."""
        tissues = self.physics_config.get("tissues", {})
        n = self.physics_config.get("n", 1.37)
        self.opt_manager = OpticalParameterManager(tissues, n=n)

    def _setup_fem_solver(self) -> None:
        """Setup FEM solver."""
        if self._mesh_data is None:
            raise RuntimeError("Mesh data not loaded. Call build_shared_assets first.")

        self.fem_solver = FEMSolver(
            nodes=self._mesh_data.nodes,
            elements=self._mesh_data.elements,
            surface_faces=self._mesh_data.surface_faces,
            tissue_labels=self._mesh_data.tissue_labels,
            opt_params_manager=self.opt_manager,
        )

    def _ensure_setup(self) -> None:
        """Ensure all components are set up."""
        if self._mesh_data is None:
            self.build_shared_assets()

        if self.atlas is None:
            self._setup_atlas()

        if self.opt_manager is None:
            self._setup_optical_manager()

        if self.fem_solver is None:
            self._setup_fem_solver()

    def _save_sample(self, sample: SampleOutput, output_path: Path, index: int) -> None:
        """Save sample data to files."""
        sample_dir = output_path / f"sample_{index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / "measurement_b.npy", sample.measurement_b)
        np.save(sample_dir / "gt_nodes.npy", sample.gt_nodes)
        np.save(sample_dir / "gt_voxels.npy", sample.gt_voxels)

        with open(sample_dir / "tumor_params.json", "w") as f:
            json.dump(sample.tumor_params, f, indent=2)

    def _validate_sample(self, sample: SampleOutput, nodes: np.ndarray) -> None:
        """Validate sample and print diagnostic info.

        Parameters
        ----------
        sample : SampleOutput
            Generated sample to validate.
        nodes : np.ndarray
            FEM mesh nodes [N x 3].
        """
        warnings = []
        source_type = sample.tumor_params.get("source_type", "gaussian")

        for focus in sample.tumor_params["foci"]:
            center = np.array(focus["center"])
            # Get radius - for ellipsoids it may be None, use rx as fallback
            radius = focus.get("radius")
            if radius is None:
                # Ellipsoid: use rx as characteristic radius
                radius = focus.get("rx", 0.5)

            # For uniform source, skip sigma-based validation
            if source_type == "uniform":
                cutoff = radius
            else:
                sigma = radius
                cutoff = 3.0 * sigma

            dists = np.linalg.norm(nodes - center, axis=1)
            nodes_inside = np.sum(dists <= cutoff)

            if nodes_inside < 3:
                warnings.append(
                    f"  WARNING: Focus at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) has only {nodes_inside} nodes in cutoff={cutoff:.1f}mm range"
                )

        gt_nonzero = np.count_nonzero(sample.gt_nodes)
        meas_nonzero = np.count_nonzero(sample.measurement_b)

        if warnings:
            for w in warnings:
                print(w)
        print(f"    gt_nodes: nonzero={gt_nonzero}, max={sample.gt_nodes.max():.4f}")
        print(
            f"    measurement_b: nonzero={meas_nonzero}, max={sample.measurement_b.max():.4f}"
        )
