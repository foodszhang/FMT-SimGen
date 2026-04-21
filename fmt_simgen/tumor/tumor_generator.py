"""
TumorGenerator: Generate tumors as analytic functions in continuous 3D space.

Key design principles:
- Tumors are defined as continuous analytic functions, NOT tied to any mesh
- Each tumor focus is an AnalyticFocus with center, shape, and parameters
- TumorSample evaluates the analytic function at arbitrary coordinates
- Constraints (inter-foci distance, depth, etc.) are checked during generation

Supported shapes:
- sphere: Gaussian sphere with radius parameter
- ellipsoid: Gaussian ellipsoid with axis ratios
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class ShapeType(Enum):
    """Supported tumor shape types."""

    SPHERE = "sphere"
    ELLIPSOID = "ellipsoid"


class SourceType(Enum):
    """Supported tumor source types."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


@dataclass
class AnalyticFocus:
    """Single tumor focus defined as an analytic function in 3D space."""

    center: np.ndarray
    shape: ShapeType
    params: Dict

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate the analytic function at given coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates [N×3] at which to evaluate.

        Returns
        -------
        np.ndarray
            Function values [N] at each coordinate.
        """
        source_type = self.params.get("source_type", "gaussian")

        if source_type == "uniform":
            return self._evaluate_uniform(coords)
        elif self.shape == ShapeType.SPHERE:
            return self._evaluate_sphere(coords)
        elif self.shape == ShapeType.ELLIPSOID:
            return self._evaluate_ellipsoid(coords)
        else:
            raise ValueError(f"Unknown shape type: {self.shape}")

    def _evaluate_uniform(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate uniform (binary) source within radius.

        d(x) = 1.0 for ||x - center|| <= radius
               0.0 otherwise
        """
        radius = self.params.get("radius", 1.0)
        cutoff = radius

        diff = coords - self.center
        in_bbox = np.all(np.abs(diff) <= cutoff, axis=1)

        values = np.zeros(coords.shape[0], dtype=np.float32)
        if np.any(in_bbox):
            subset = diff[in_bbox]
            dist2 = np.sum(subset ** 2, axis=1)
            mask = dist2 <= cutoff ** 2
            values[np.where(in_bbox)[0][mask]] = 1.0
        return values

    def _evaluate_sphere(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian sphere with sigma = radius, truncated at 4*sigma.

        d(x) = exp(-||x - center||^2 / (2 * sigma^2)) for ||x - center|| <= 4*sigma
               0 otherwise
        """
        radius = self.params.get("radius", 1.0)
        sigma = radius
        cutoff = 4.0 * sigma

        diff = coords - self.center
        in_bbox = np.all(np.abs(diff) <= cutoff, axis=1)

        values = np.zeros(coords.shape[0], dtype=np.float32)
        if np.any(in_bbox):
            subset = diff[in_bbox]
            dist2 = np.sum(subset ** 2, axis=1)
            mask = dist2 <= cutoff ** 2
            values[np.where(in_bbox)[0][mask]] = np.exp(
                -dist2[mask] / (2.0 * sigma ** 2)
            ).astype(np.float32)
        return values

    def _evaluate_ellipsoid(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian ellipsoid with sigma = radii, truncated at 4*sigma.

        d(x) = exp(-sum((x_i - c_i)^2 / (2 * sigma_i^2))) for all ||x - center|| <= 4*sigma_max
               0 otherwise
        """
        radii = np.array(
            [
                self.params.get("rx", 1.0),
                self.params.get("ry", 1.0),
                self.params.get("rz", 1.0),
            ],
            dtype=np.float32,
        )
        sigma = radii
        cutoff = 4.0 * np.max(sigma)

        diff = coords - self.center
        in_bbox = np.all(np.abs(diff) <= cutoff, axis=1)

        values = np.zeros(coords.shape[0], dtype=np.float32)
        if np.any(in_bbox):
            subset = diff[in_bbox]
            normalized = subset / sigma
            dist2 = np.sum(normalized ** 2, axis=1)
            mask = dist2 <= cutoff ** 2
            values[np.where(in_bbox)[0][mask]] = np.exp(
                -0.5 * dist2[mask]
            ).astype(np.float32)
        return values


@dataclass
class TumorSample:
    """A tumor sample containing multiple foci."""

    foci: List[AnalyticFocus] = field(default_factory=list)

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate tumor distribution at given coordinates using max of all foci.

        For multi-focal tumors, takes the maximum intensity across all foci
        at each point (avoids signal buildup in overlapping regions).

        Parameters
        ----------
        coords : np.ndarray
            Coordinates [N×3] at which to evaluate.

        Returns
        -------
        np.ndarray
            Combined tumor values [N] (max of all foci).
        """
        if not self.foci:
            return np.zeros(coords.shape[0])

        values = np.zeros(coords.shape[0])
        for focus in self.foci:
            focus_values = focus.evaluate(coords)
            values = np.maximum(values, focus_values)

        return values

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_foci": len(self.foci),
            "depth_tier": getattr(self, "_depth_tier", None),
            "depth_mm": getattr(self, "_depth_mm", None),
            "source_type": self.foci[0].params.get("source_type", "gaussian") if self.foci else "gaussian",
            "foci": [
                {
                    "center": list(focus.center),  # trunk-local mm ← 主字段
                    "center_atlas_mm": list(focus.center_atlas_mm),  # 调试用
                    "shape": focus.shape.value,
                    "params": focus.params,
                    # Stage 2 voxel generation needs:
                    "radius": focus.params.get("radius"),
                    "rx": focus.params.get("rx"),
                    "ry": focus.params.get("ry"),
                    "rz": focus.params.get("rz"),
                }
                for focus in self.foci
            ],
        }


class TumorGenerator:
    """Generate tumor samples as collections of analytic foci."""

    def __init__(
        self,
        config: Dict,
        atlas=None,
        mesh_bbox=None,
        mesh_nodes=None,
        tissue_labels=None,
        elements=None,
        organ_constraint_disabled: bool = False,
        trunk_offset_mm=None,
        mcx_bbox_mm=None,  # (min_xyz, max_xyz) tuple of np.ndarray
        gt_offset_mm=None,  # gt_voxels offset in trunk-local mm
        gt_shape=None,  # gt_voxels shape (Nx, Ny, Nz)
        gt_spacing_mm=None,  # voxel spacing of gt_voxels grid
        merged_voxel_volume=None,  # trunk-cropped merged atlas volume [X,Y,Z] at voxel_size_mm
        voxel_size_mm: float = 0.2,
    ):
        """Initialize tumor generator.

        Parameters
        ----------
        config : Dict
            Configuration dictionary with tumor parameters:
            - regions: List[str] - ["dorsal", "lateral"]
            - num_foci_distribution: Dict[int, float] - {1: 0.30, 2: 0.35, 3: 0.35}
            - shapes: List[str] - ["sphere", "ellipsoid"]
            - radius_range: List[float, float] - [min, max] in mm
            - depth_range: List[float, float] - [min, max]皮下深度 in mm
            - depth_distribution: Dict - depth tier configuration
            - min_inter_foci_distance_ratio: float - ratio * 灶直径
            - ellipsoid_axis_ratio: List[float, float] - [rx_ratio, rz_ratio]
            - max_cluster_radius: float - max distance from anchor for multi-focus
            - min_foci_distance_abs: float - min absolute inter-foci distance (mm)
        atlas : DigimouseAtlas, optional
            Atlas for region sampling. If None, use config-based ranges.
        mesh_bbox : Dict, optional
            Mesh bounding box for depth estimation.
            {"min": [x_min, y_min, z_min], "max": [x_max, y_max, z_max]}
        mesh_nodes : np.ndarray, optional
            FEM mesh nodes [N×3] for organ constraint validation.
        tissue_labels : np.ndarray, optional
            Tissue labels [N] for mesh nodes.
        elements : np.ndarray, optional
            FEM mesh elements [N_tets×4] for organ constraint validation.
        organ_constraint_disabled : bool, optional
            If True, skip organ boundary constraint check (use for verification
            datasets where the constraint is too restrictive for the mesh).
        trunk_offset_mm : array-like, optional
            Offset from atlas-corner to trunk-local world frame [0,34,0].
            Used to convert foci.center from atlas mm to trunk-local mm.
        mcx_bbox_mm : tuple, optional
            (min_xyz, max_xyz) tuple of MCX volume bbox in trunk-local mm.
            Used to reject tumors outside MCX volume.
        gt_offset_mm : array-like, optional
            Offset of gt_voxels grid origin in trunk-local mm.
            Used with gt_shape to reject tumors whose Gaussian support
            falls entirely outside the gt_voxels lookup domain.
        gt_shape : tuple, optional
            Shape of gt_voxels grid (Nx, Ny, Nz).
        gt_spacing_mm : float, optional
            Voxel spacing of gt_voxels grid. Used for organ constraint
            (defaults to 0.2 if not provided).
        merged_voxel_volume : np.ndarray, optional
            Trunk-cropped merged atlas volume [X, Y, Z] at voxel_size_mm resolution.
            Used for voxel-based organ constraint validation (replaces KD-Tree).
        voxel_size_mm : float, default 0.2
            Voxel size of merged_voxel_volume in mm.
        """
        self.config = config
        self.atlas = atlas
        self.mesh_bbox = mesh_bbox
        self.mesh_nodes = mesh_nodes
        self.tissue_labels = tissue_labels
        self.elements = elements
        self._organ_constraint_disabled = organ_constraint_disabled
        self.trunk_offset_mm = (
            np.asarray(trunk_offset_mm, dtype=np.float64)
            if trunk_offset_mm is not None else np.zeros(3)
        )
        self.mcx_bbox_mm = mcx_bbox_mm
        self.gt_offset_mm = (
            np.asarray(gt_offset_mm, dtype=np.float64)
            if gt_offset_mm is not None else None
        )
        self.gt_shape = gt_shape
        self.gt_spacing_mm = gt_spacing_mm if gt_spacing_mm is not None else 0.2
        self.merged_voxel_volume = merged_voxel_volume
        self.voxel_size_mm = voxel_size_mm

        self.regions = config.get("regions", ["dorsal", "lateral"])
        self.num_foci_dist = config.get(
            "num_foci_distribution", {1: 0.30, 2: 0.35, 3: 0.35}
        )
        self.shapes = [
            ShapeType(s) for s in config.get("shapes", ["sphere", "ellipsoid"])
        ]
        self.radius_range = config.get("radius_range", [1.0, 2.5])
        self.depth_range = config.get("depth_range", [1.5, 8.0])
        self.depth_distribution = config.get("depth_distribution", {})
        self.min_inter_ratio = config.get("min_inter_foci_distance_ratio", 1.5)
        self.ellipsoid_ratio = config.get("ellipsoid_axis_ratio", [1.2, 1.5])
        self.max_cluster_radius = config.get("max_cluster_radius", 8.0)
        self.min_foci_distance_abs = config.get("min_foci_distance_abs", 3.0)

        self._rng = np.random.default_rng()

        # ── Rejection diagnostic counters ───────────────────────────
        self._reject_stats = {
            "layer1_organ": 0,
            "layer1_depth": 0,
            "layer2_mcx_bbox": 0,
            "max_attempts_hit": 0,
            "accepted": 0,
        }

        # ── Precompute sampling coordinates for all depth tiers ──
        self._precompute_sampling_coords()

    def _precompute_sampling_coords(self) -> None:
        """Pre-compute sampling coordinates for all depth tier + region combinations.

        This triggers EDT computation once upfront, then all subsequent calls
        to get_subcutaneous_coords() are cache hits (O(1)).
        """
        if self.atlas is None:
            return

        logger.info("  Pre-computing subcutaneous sampling coordinates...")
        # Trigger full-range cache build for each region
        for region in self.regions:
            self.atlas.get_subcutaneous_coords(
                depth_range_mm=tuple(self.depth_range),
                regions=[region],
            )
            # Also pre-compute each depth tier
            for tier_name, tier_cfg in self.depth_distribution.items():
                tier_range = tuple(tier_cfg["range"])
                self.atlas.get_subcutaneous_coords(
                    depth_range_mm=tier_range,
                    regions=[region],
                )
        logger.info("  Sampling coordinate pre-computation done.")

    def generate_sample(
        self,
        num_foci: Optional[int] = None,
        depth_mm: Optional[float] = None,
        depth_tier: Optional[str] = None,
    ) -> TumorSample:
        """Generate a single tumor sample.

        For multi-focal samples, uses cluster-based placement where
        additional foci are placed near the anchor focus.

        Parameters
        ----------
        num_foci : int, optional
            If provided, forces the exact number of foci instead of
            randomly sampling from the distribution.
        depth_mm : float, optional
            If provided, forces the tumor depth in mm (from body surface).
        depth_tier : str, optional
            Depth tier name (shallow/medium/deep) for reporting.

        Returns
        -------
        TumorSample
            Generated tumor sample with random foci.
        """
        if num_foci is None:
            num_foci = self._sample_num_foci()
        shape = self._rng.choice(self.shapes)
        radius = self._rng.uniform(*self.radius_range)

        foci: List[AnalyticFocus] = []
        cluster_radius = self.max_cluster_radius
        organ_constraint_passed = True

        for focus_idx in range(num_foci):
            center = None
            is_anchor = focus_idx == 0
            attempts = 0
            max_attempts = 100

            while center is None and attempts < max_attempts:
                attempts += 1

                if is_anchor:
                    candidate_center, from_atlas = self._sample_position_with_source(
                        forced_depth=depth_mm
                    )
                else:
                    candidate_center = self._sample_cluster_position(
                        foci[0] if foci else None, cluster_radius
                    )

                if candidate_center is None:
                    continue

                if not from_atlas if is_anchor else True:
                    depth = self._get_depth_at_position(candidate_center)
                    if depth < self.depth_range[0] or depth > self.depth_range[1]:
                        self._reject_stats["layer1_depth"] += 1
                        if is_anchor:
                            candidate_center, from_atlas = (
                                self._sample_position_with_source(forced_depth=depth_mm)
                            )
                        else:
                            candidate_center = None
                        continue

                # Check organ boundary constraint for anchor focus
                if is_anchor and self.mesh_nodes is not None:
                    if not self.is_valid_placement(candidate_center, radius):
                        self._reject_stats["layer1_organ"] += 1
                        # Try 20 independent random positions instead of fixed depths
                        valid_fallback = None
                        for _ in range(20):
                            fb_center, fb_from_atlas = self._sample_position_with_source(
                                forced_depth=depth_mm
                            )
                            if fb_center is None:
                                continue
                            fb_depth_val = self._get_depth_at_position(fb_center)
                            if not (self.depth_range[0] <= fb_depth_val <= self.depth_range[1]):
                                continue
                            if self.is_valid_placement(fb_center, radius):
                                valid_fallback = (fb_center, fb_from_atlas)
                                break
                        if valid_fallback is not None:
                            candidate_center, from_atlas = valid_fallback
                        else:
                            candidate_center = None
                            continue

                params = self._get_shape_params(shape, radius)
                new_focus = AnalyticFocus(
                    center=candidate_center, shape=shape, params=params
                )
                test_foci = foci + [new_focus]

                if self._check_constraints(test_foci):
                    foci.append(new_focus)
                    center = candidate_center
                else:
                    if not is_anchor and cluster_radius < 12.0:
                        cluster_radius = 12.0

            if center is None and focus_idx > 0:
                cluster_radius = 12.0
                candidate_center = self._sample_cluster_position(
                    foci[0] if foci else None, cluster_radius
                )
                if candidate_center is not None:
                    depth = self._get_depth_at_position(candidate_center)
                    if self.depth_range[0] <= depth <= self.depth_range[1]:
                        params = self._get_shape_params(shape, radius)
                        new_focus = AnalyticFocus(
                            center=candidate_center, shape=shape, params=params
                        )
                        foci.append(new_focus)

            if center is None:
                if is_anchor:
                    self._reject_stats["max_attempts_hit"] += 1
                organ_constraint_passed = False

        # Store metadata in the sample via a wrapper or side-effect
        sample = TumorSample(foci=foci)
        sample._depth_tier = depth_tier
        sample._depth_mm = depth_mm
        sample._organ_constraint_passed = organ_constraint_passed

        # [FIX v3] atlas-corner mm → mcx_trunk_local_mm
        for focus in sample.foci:
            focus.center_atlas_mm = np.asarray(focus.center, dtype=np.float64).copy()
            focus.center = (focus.center_atlas_mm - self.trunk_offset_mm).tolist()

        # Reject tumors outside MCX volume (only when organ_constraint is enabled)
        if not self._organ_constraint_disabled and self.mcx_bbox_mm is not None:
            lo, hi = self.mcx_bbox_mm
            if self.gt_offset_mm is not None and self.gt_shape is not None:
                gt_hi = self.gt_offset_mm + self.gt_spacing_mm * np.array(self.gt_shape)
                strict_lo = np.maximum(lo, self.gt_offset_mm)
                strict_hi = np.minimum(hi, gt_hi)
            else:
                strict_lo = lo
                strict_hi = hi
            for focus in sample.foci:
                c = np.asarray(focus.center)
                r = focus.params.get("radius", 1.0)
                if np.any(c - r < strict_lo) or np.any(c + r > strict_hi):
                    self._reject_stats["layer2_mcx_bbox"] += 1
                    sample._organ_constraint_passed = False
                    break

        if sample._organ_constraint_passed:
            self._reject_stats["accepted"] += 1

        return sample

    def is_valid_placement(self, center: np.ndarray, radius: float) -> bool:
        """Check if tumor placement is valid (no cross-organ boundary).

        Rules:
        - Only soft tissue (skin, muscle, fat, tongue) -> OK
        - Involves 1 organ type -> OK
        - Involves 2+ different organ types -> Invalid

        Biological basis:
        - Subcutaneous xenograft stays within single tissue layer
        - Orthotopic tumors grow within specific organ, not cross-boundary
        - Different organs have very different optical properties
          (muscle mu_a=0.087 vs liver mu_a=0.352), breaking FEM homogeneity

        Parameters
        ----------
        center : np.ndarray
            Tumor center [3] in mm.
        radius : float
            Tumor radius in mm.

        Returns
        -------
        bool
            True if placement is valid.
        """
        if self._organ_constraint_disabled:
            return True  # Constraint disabled via config
        if self.merged_voxel_volume is None:
            return True  # No voxel volume available, skip validation

        # Query sphere of radius=r in trunk-local coords
        vol = self.merged_voxel_volume
        vs = self.voxel_size_mm
        vol_shape = vol.shape  # [X, Y, Z]

        # Convert trunk-local center to voxel indices
        cx, cy, cz = center[0] / vs, center[1] / vs, center[2] / vs
        r_vox = radius / vs

        # Bounding box of sphere in voxel space
        x_min = max(0, int(cx - r_vox) - 1)
        x_max = min(vol_shape[0], int(cx + r_vox) + 2)
        y_min = max(0, int(cy - r_vox) - 1)
        y_max = min(vol_shape[1], int(cy + r_vox) + 2)
        z_min = max(0, int(cz - r_vox) - 1)
        z_max = min(vol_shape[2], int(cz + r_vox) + 2)

        # Extract subvolume and compute distance mask
        subvol = vol[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        sx, sy, sz = np.meshgrid(
            np.arange(subvol.shape[0]),
            np.arange(subvol.shape[1]),
            np.arange(subvol.shape[2]),
            indexing='ij',
        )
        # Distances from sphere center (in voxels)
        dist = np.sqrt(
            (sx - (cx - x_min)) ** 2
            + (sy - (cy - y_min)) ** 2
            + (sz - (cz - z_min)) ** 2
        )
        mask = dist <= r_vox
        labels_in_sphere = subvol[mask]

        if labels_in_sphere.size == 0:
            return True

        # Volume fraction rule
        unique, counts = np.unique(labels_in_sphere, return_counts=True)
        fractions = counts / len(labels_in_sphere)

        # Soft tissue labels: {1} = skin (includes subcutaneous in this atlas)
        soft_labels = {1}
        organ_mask = ~np.isin(unique, list(soft_labels) + [0])
        organ_unique = unique[organ_mask]
        organ_counts = counts[organ_mask]

        if len(organ_unique) == 0:
            return True  # Pure soft tissue

        dominant_idx = np.argmax(organ_counts)
        dominant_label = organ_unique[dominant_idx]
        dominant_fraction = organ_counts[dominant_idx] / len(labels_in_sphere)

        # Dominant organ must be ≥ 80%
        if dominant_fraction < 0.80:
            return False

        # Other organs total ≤ 10%
        other_fraction = 1.0 - dominant_fraction - (
            counts[np.isin(unique, list(soft_labels) + [0])].sum() / len(labels_in_sphere)
        )
        if other_fraction > 0.10:
            return False

        return True

    def _sample_cluster_position(
        self, anchor: Optional[AnalyticFocus], cluster_radius: float
    ) -> Optional[np.ndarray]:
        """Sample a position near the anchor for cluster placement.

        Parameters
        ----------
        anchor : AnalyticFocus or None
            Anchor focus to cluster around.
        cluster_radius : float
            Maximum distance from anchor.

        Returns
        -------
        np.ndarray or None
            Sampled position or None if sampling fails.
        """
        if anchor is None:
            return self._sample_position_with_source()[0]

        anchor_center = anchor.center

        for _ in range(50):
            r = self._rng.uniform(0, cluster_radius)
            theta = self._rng.uniform(0, 2 * np.pi)
            phi = self._rng.uniform(0, np.pi)

            offset = r * np.array(
                [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
            )

            candidate = anchor_center + offset

            if self.mesh_bbox is not None:
                bbox_min = np.array(self.mesh_bbox["min"])
                bbox_max = np.array(self.mesh_bbox["max"])
                if np.any(candidate < bbox_min) or np.any(candidate > bbox_max):
                    continue

            return candidate

        return None

    def _sample_num_foci(self) -> int:
        """Sample number of foci from distribution."""
        num_foci_list = list(self.num_foci_dist.keys())
        probs = list(self.num_foci_dist.values())
        return int(self._rng.choice(num_foci_list, p=probs))

    def _sample_position(self) -> np.ndarray:
        """Sample a position in the subcutaneous region.

        Returns
        -------
        np.ndarray
            Sampled position [3] in mm.
        """
        pos, _ = self._sample_position_with_source()
        return pos

    def _sample_position_with_source(
        self, forced_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """Sample a position and return whether it came from atlas.

        Parameters
        ----------
        forced_depth : float, optional
            If provided, force the depth in mm from body surface.

        Returns
        -------
        Tuple[np.ndarray, bool]
            (position [3] in mm, actually_from_atlas bool)
        """
        if self.atlas is not None:
            pos, used_atlas = self._sample_from_atlas_with_flag(
                forced_depth=forced_depth
            )
            return pos, used_atlas
        else:
            pos = self._sample_from_config(forced_depth=forced_depth)
            return pos, False

    def _sample_from_atlas_with_flag(
        self, forced_depth: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """Attempt to sample from atlas skin surface region.

        Uses pre-cached skin surface coordinates for O(1) random sampling.
        This is the correct placement for subcutaneous xenograft tumors when
        there is no distinct subcutaneous fat/muscle layer in the atlas.

        Parameters
        ----------
        forced_depth : float, optional
            Ignored for skin surface sampling (depth is fixed at body surface).

        Returns
        -------
        Tuple[np.ndarray, bool]
            (position [3] in mm, actually_from_atlas bool)
            Returns (config_position, False) if atlas sampling fails.
        """
        region = self._rng.choice(self.regions)

        # Use skin surface coordinates (no depth filtering needed since
        # skin IS the body surface — no subcutaneous layer in this atlas)
        coords = self.atlas.get_skin_surface_coords(
            regions=[region],
            torso_only=True,
        )

        if len(coords) == 0:
            return self._sample_from_config(forced_depth=forced_depth), False

        idx = self._rng.integers(len(coords))
        pos = coords[idx].copy()
        return pos, True

    def _sample_from_config(
        self, forced_depth: Optional[float] = None
    ) -> np.ndarray:
        """Sample position from config-based ranges (fallback).

        Mesh coordinate ranges (0-based physical coords):
        X: [2.4, 34.4] mm, Y: [4.8, 92.8] mm, Z: [1.6, 20.0] mm

        Dorsal side = high Z values (Z > 12mm)
        Lateral sides = extreme X values (X < 8 or X > 28)
        Trunk region = Y in [20, 70] mm (exclude head and tail)

        Parameters
        ----------
        forced_depth : float, optional
            If provided, force the depth in mm from body surface.
        """
        region = self._rng.choice(self.regions) if self.regions else "dorsal"

        if region == "dorsal":
            # For dorsal, Z corresponds to depth from back surface
            # Body surface is at Z ≈ 10-11mm, so depth = Z - 10
            if forced_depth is not None:
                z_center = 10.0 + forced_depth
                z_center = max(10.5, min(19.0, z_center))
            else:
                z_center = self._rng.uniform(15.0, 19.0)
            return np.array(
                [
                    self._rng.uniform(10.0, 28.0),
                    self._rng.uniform(20.0, 70.0),
                    z_center,
                ],
                dtype=np.float64,
            )
        elif region == "lateral":
            x_side = self._rng.choice(
                [
                    self._rng.uniform(3.0, 8.0),
                    self._rng.uniform(28.0, 33.0),
                ]
            )
            if forced_depth is not None:
                z_center = 10.0 + forced_depth
                z_center = max(10.5, min(14.0, z_center))
            else:
                z_center = self._rng.uniform(8.0, 14.0)
            return np.array(
                [
                    x_side,
                    self._rng.uniform(20.0, 70.0),
                    z_center,
                ],
                dtype=np.float64,
            )
        else:
            if forced_depth is not None:
                z_center = 10.0 + forced_depth
                z_center = max(10.5, min(19.0, z_center))
            else:
                z_center = self._rng.uniform(2.0, 5.0)
            return np.array(
                [
                    self._rng.uniform(10.0, 28.0),
                    self._rng.uniform(20.0, 70.0),
                    z_center,
                ],
                dtype=np.float64,
            )

    def _get_depth_at_position(self, position: np.ndarray) -> float:
        """Estimate depth from nearest body surface.

        Uses mesh bounding box as surface approximation.
        Depth = min distance to any bbox face.

        Parameters
        ----------
        position : np.ndarray
            Position [3] in mm.

        Returns
        -------
        float
            Estimated depth in mm (positive value).
        """
        if self.mesh_bbox is not None:
            bbox_min = np.array(self.mesh_bbox["min"])
            bbox_max = np.array(self.mesh_bbox["max"])
            distances = np.concatenate(
                [
                    position - bbox_min,
                    bbox_max - position,
                ]
            )
            return float(np.min(np.abs(distances)))

        return 2.0

    def _get_shape_params(self, shape: ShapeType, base_radius: float) -> Dict:
        """Get shape parameters.

        Parameters
        ----------
        shape : ShapeType
            Shape type.
        base_radius : float
            Base radius in mm.

        Returns
        -------
        Dict
            Shape parameters.
        """
        source_type = self.config.get("source_type", "gaussian")
        params = {"source_type": source_type}

        if shape == ShapeType.SPHERE:
            params["radius"] = base_radius
        elif shape == ShapeType.ELLIPSOID:
            rx_ratio, rz_ratio = self.ellipsoid_ratio
            params["rx"] = base_radius * rx_ratio
            params["ry"] = base_radius
            params["rz"] = base_radius * rz_ratio
        else:
            params["radius"] = base_radius

        return params

    def _check_constraints(self, foci_list: List[AnalyticFocus]) -> bool:
        """Check if foci satisfy inter-foci distance constraints.

        Parameters
        ----------
        foci_list : List[AnalyticFocus]
            List of foci to check.

        Returns
        -------
        bool
            True if constraints are satisfied.
        """
        if len(foci_list) < 2:
            return True

        for i in range(len(foci_list)):
            for j in range(i + 1, len(foci_list)):
                dist = np.linalg.norm(foci_list[i].center - foci_list[j].center)

                r_i = foci_list[i].params.get("radius", 1.0)
                r_j = foci_list[j].params.get("radius", 1.0)
                min_dist_by_diameter = self.min_inter_ratio * (r_i + r_j)
                min_dist = max(min_dist_by_diameter, self.min_foci_distance_abs)

                if dist < min_dist:
                    return False

        return True
