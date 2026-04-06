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


@dataclass
class SampleOutput:
    """Container for single sample output."""

    measurement_b: np.ndarray
    gt_nodes: np.ndarray
    gt_voxels: np.ndarray
    tumor_params: Dict


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

        self._mesh_data: Optional[MeshData] = None

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

        if (
            not force_regenerate
            and mesh_file.exists()
            and matrix_file.with_suffix(".M.npz").exists()
        ):
            print(f"Shared assets already exist at {mesh_path}, skipping.")
            self._load_mesh_data(str(mesh_file))
            self._setup_optical_manager()
            self._load_fem_solver(str(matrix_file))
            return {"mesh": mesh_file, "matrix": matrix_file}

        if not force_regenerate and default_output.exists():
            existing_mesh = default_output / "mesh.npz"
            existing_matrix = default_output / "system_matrix.M.npz"
            if existing_mesh.exists() and existing_matrix.exists():
                print(f"Loading existing assets from {default_output}")
                self._load_mesh_data(str(existing_mesh))
                self._setup_optical_manager()
                self._load_fem_solver(str(existing_matrix.parent / "system_matrix"))
                return {
                    "mesh": existing_mesh,
                    "matrix": existing_matrix.parent / "system_matrix",
                }

        print("Building shared assets...")

        self._setup_atlas()
        self._setup_mesh_generator()
        self._setup_optical_manager()

        print("  Generating tetrahedral mesh...")
        self._mesh_data = self.mesh_generator.generate(
            atlas_volume=self.atlas.volume, voxel_size=self.atlas.voxel_size
        )

        mesh_file = self.mesh_generator.save(self._mesh_data, "mesh")

        print("  Assembling FEM system matrix...")
        self._setup_fem_solver()
        self.fem_solver.assemble_system_matrix()

        matrix_file = self.fem_solver.save_system_matrix(
            str(mesh_path / "system_matrix")
        )

        print(f"Shared assets saved to {mesh_path}")

        return {"mesh": mesh_file, "matrix": matrix_file}

    def _load_mesh_data(self, mesh_path: str) -> None:
        """Load mesh data from npz file.

        Parameters
        ----------
        mesh_path : str
            Path to mesh npz file.
        """
        data = np.load(mesh_path, allow_pickle=True)
        self._mesh_data = MeshData(
            nodes=data["nodes"],
            elements=data["elements"],
            surface_faces=data["surface_faces"],
            tissue_labels=data["tissue_labels"],
            surface_node_indices=data["surface_node_indices"],
        )
        print(
            f"  Loaded mesh: {self._mesh_data.nodes.shape[0]} nodes, "
            f"{self._mesh_data.elements.shape[0]} elements"
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

    def build_samples(self, num_samples: Optional[int] = None) -> List[SampleOutput]:
        """Generate tumor samples with measurements.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to generate. Uses config default if None.

        Returns
        -------
        List[SampleOutput]
            List of generated samples.
        """
        if num_samples is None:
            num_samples = self.dataset_config.get("num_samples", 50)

        self._ensure_setup()

        voxel_spacing = self.dataset_config.get("voxel_spacing", 0.1)
        roi_size = self.dataset_config.get("voxel_grid_roi_size_mm", 40.0)

        # Experiment-isolated output: data/{experiment_name}/samples/
        experiment_name = self.dataset_config.get("experiment_name", "default")
        base_output = Path(self.dataset_config.get("output_path", "data/"))
        experiment_output = base_output / experiment_name
        samples_output = experiment_output / "samples"

        samples_output.mkdir(parents=True, exist_ok=True)

        mesh_center = self._mesh_data.nodes.mean(axis=0)
        roi_half = roi_size / 2.0
        node_min = mesh_center - roi_half
        node_max = mesh_center + roi_half
        shape = np.ceil((node_max - node_min) / voxel_spacing).astype(int)

        voxel_grid_config = VoxelGridConfig(
            shape=tuple(shape),
            spacing=voxel_spacing,
            offset=node_min,
        )

        mesh_min = self._mesh_data.nodes.min(axis=0)
        mesh_max = self._mesh_data.nodes.max(axis=0)
        mesh_bbox = {"min": mesh_min.tolist(), "max": mesh_max.tolist()}

        # FIXED: Pass mesh_nodes, tissue_labels, elements for organ constraint validation
        self.tumor_generator = TumorGenerator(
            config=self.tumor_config,
            atlas=self.atlas,
            mesh_bbox=mesh_bbox,
            mesh_nodes=self._mesh_data.nodes,
            tissue_labels=self._mesh_data.tissue_labels,
            elements=self._mesh_data.elements,
        )

        dual_sampler = DualSampler(
            nodes=self._mesh_data.nodes, voxel_grid_config=voxel_grid_config
        )

        # Quality filter settings
        filter_enabled = self.quality_config.get("enabled", False)
        min_b_max = self.quality_config.get("min_b_max", 0.001)
        min_gt_frac = self.quality_config.get("min_gt_nonzero_frac", 0.001)
        max_retries = self.quality_config.get("max_retries", 5)

        samples: List[SampleOutput] = []
        rejected_count = 0

        for i in range(num_samples):
            print(f"  Generating sample {i + 1}/{num_samples}...")

            for attempt in range(max_retries + 1):
                tumor_sample = self.tumor_generator.generate_sample()

                gt_nodes = dual_sampler.sample_to_nodes(tumor_sample)

                gt_voxels = dual_sampler.sample_to_voxels(tumor_sample)

                measurement_b = self.fem_solver.forward(gt_nodes)

                b_max = float(np.max(np.abs(measurement_b)))
                gt_nonzero_frac = float(np.count_nonzero(gt_nodes)) / len(gt_nodes)

                if not filter_enabled:
                    break
                if b_max >= min_b_max and gt_nonzero_frac >= min_gt_frac:
                    break
                if attempt < max_retries:
                    print(f"    [FILTER] Sample {i} rejected (b_max={b_max:.6f}, "
                          f"gt_frac={gt_nonzero_frac:.6f}), retrying {attempt + 1}/{max_retries}")
                else:
                    print(f"    [FILTER] Sample {i} kept after {max_retries} retries "
                          f"(b_max={b_max:.6f})")

            if filter_enabled and b_max < min_b_max:
                rejected_count += 1

            sample = SampleOutput(
                measurement_b=measurement_b,
                gt_nodes=gt_nodes,
                gt_voxels=gt_voxels,
                tumor_params=tumor_sample.to_dict(),
            )
            samples.append(sample)

            self._save_sample(sample, samples_output, i)

            self._validate_sample(sample, self._mesh_data.nodes)

        print(f"Dataset saved to {experiment_output}")

        if rejected_count > 0:
            print(f"[WARNING] {rejected_count} samples had b_max below threshold")

        # Generate train/val splits (80/20)
        self._generate_splits(experiment_output, len(samples))

        # Generate manifest
        self._generate_manifest(experiment_output, samples)

        return samples

    def _generate_splits(self, experiment_output: Path, num_samples: int) -> None:
        """Generate train/val split files.

        Parameters
        ----------
        experiment_output : Path
            Path to experiment output directory.
        num_samples : int
            Total number of samples.
        """
        sample_ids = [f"sample_{i:04d}" for i in range(num_samples)]
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

    def _generate_manifest(self, experiment_output: Path, samples: List[SampleOutput]) -> None:
        """Generate dataset manifest JSON.

        Parameters
        ----------
        experiment_output : Path
            Path to experiment output directory.
        samples : List[SampleOutput]
            List of generated samples.
        """
        manifest = {
            "experiment_name": self.dataset_config.get("experiment_name", "default"),
            "source_type": self.tumor_config.get("source_type", "gaussian"),
            "num_samples": len(samples),
            "mesh_nodes": int(self._mesh_data.nodes.shape[0]),
            "mesh_surface_nodes": int(len(self._mesh_data.surface_node_indices)),
            "config": self.config,
            "samples": []
        }

        for i, sample in enumerate(samples):
            tumor_params = sample.tumor_params
            manifest["samples"].append({
                "id": f"sample_{i:04d}",
                "num_foci": tumor_params["num_foci"],
                "depth_tier": tumor_params.get("depth_tier", "unknown"),
                "depth_mm": tumor_params.get("depth_mm"),
                "b_max": float(np.max(np.abs(sample.measurement_b))),
                "b_mean": float(np.mean(np.abs(sample.measurement_b))),
                "gt_max": float(np.max(sample.gt_nodes)),
                "gt_nonzero_count": int(np.count_nonzero(sample.gt_nodes)),
                "gt_nonzero_frac": float(np.count_nonzero(sample.gt_nodes)) / len(sample.gt_nodes),
                "has_gt_voxels": True,
            })

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
