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

    center: np.ndarray  # trunk-local mm (always — internal invariant)
    shape: ShapeType
    params: Dict
    center_atlas_mm: Optional[np.ndarray] = None  # atlas frame, for debug serialization only

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
                    "center": list(focus.center),  # trunk-local mm
                    "center_atlas_mm": (
                        list(focus.center_atlas_mm)
                        if focus.center_atlas_mm is not None
                        else None
                    ),
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
            # layer1_organ细分：来自anchor还是cluster
            "anchor_is_valid_false": 0,       # anchor首调is_validplacement失败
            "anchor_fallback_exhausted": 0,    # anchor 20次fallback全失败
            "cluster_is_valid_false": 0,       # cluster首调is_validplacement失败
            "cluster_fallback_exhausted": 0,    # cluster 20次fallback全失败
            # is_validplacement内部三条规则各自的失败计数
            "reject_air": 0,           # 球体碰到空气
            "reject_body_frac": 0,     # body_frac < 0.85
            "reject_multi_organ": 0,   # unique organ > 1
            # 旧兼容key（保留方便看总数）
            "layer1_organ": 0,
            "layer1_depth": 0,
            "layer2_mcx_bbox": 0,
            "max_attempts_hit": 0,
            "accepted": 0,
            "zone_no_voxels": 0,
        }

        # ── Zone pool: anatomy-guided safe anchor positions ──
        self._safe_zone_pools: Dict[str, np.ndarray] = {}
        self._safe_shell_mask: Optional[np.ndarray] = None
        self._zone_volumes: Dict[str, float] = {}

        # ── Precompute sampling coordinates for all depth tiers ──
        self._precompute_sampling_coords()

        # ── Build safe shell + zone pools ─────────────────────────
        self._build_safe_zone_pools()

        # ── Pre-compute final-valid pools per radius bin (S3) ────
        self._final_valid_pools: Dict[str, Dict[str, np.ndarray]] = {}
        self._build_final_valid_pools()

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

    def _build_safe_zone_pools(self) -> None:
        """Build anatomy-guided zone pools using EDT-depth-filtered subcutaneous coords.

        Zone design uses the atlas's Euclidean Distance Transform (EDT) to
        identify voxels at specific depths below the body surface. This ensures
        zone pool centers are guaranteed to be at least (radius + 0.3mm) below
        the surface — sphere will not intersect air.

        Zone definitions (trunk-local mm, all verified against atlas centroid):
        - upper_back:      Y in [0, 16], mid-Z (dorsal)
        - mid_back:        Y in [14, 28], mid-Z (dorsal)
        - lower_back:      Y in [26, 40], mid-Z (dorsal)
        - left_flank:      X in [4, 10], Y in [4, 36]
        - right_flank:     X in [28, 34], Y in [4, 36]
        - paraspinal_l:    X in [8, 14], Y in [0, 40]
        - paraspinal_r:    X in [24, 30], Y in [0, 40]

        Each zone pool = atlas subcutaneous coords within zone x/y bounds,
        at depth range [2.0, 8.0]mm (covers r=2.0..3.5mm with margin).
        """
        if self.atlas is None:
            logger.warning("  No atlas — skipping zone pools")
            return

        self._safe_zone_pools.clear()
        self._zone_volumes.clear()

        # Use depth range that covers r=2.0..3.5mm (our radius range)
        # depth_min = 2.0 + 0.3 = 2.3mm, depth_max = 3.5 + 5.0 = 8.5mm
        # We use [2.0, 8.5] to be conservative
        depth_min_mm = 2.0
        depth_max_mm = 8.5

        # Get all subcutaneous coords at the target depth (atlas mm frame)
        all_depth_filtered = self.atlas.get_subcutaneous_coords(
            depth_range_mm=(depth_min_mm, depth_max_mm),
            regions=["dorsal", "lateral"],
            torso_only=True,
        )
        logger.info(
            f"  Total depth-filtered subcutaneous coords: {len(all_depth_filtered):,} "
            f"(depth {depth_min_mm}–{depth_max_mm}mm in atlas mm)"
        )

        # Convert to trunk-local mm
        all_trunk_local = all_depth_filtered - self.trunk_offset_mm  # [N, 3] trunk-local mm

        # Zone definitions: (name, x_range_mm, y_range_mm)
        ZONE_DEFS = [
            ("upper_back",     (4.0, 34.0), (0.0, 16.0)),
            ("mid_back",       (4.0, 34.0), (14.0, 28.0)),
            ("lower_back",     (4.0, 34.0), (26.0, 40.0)),
            ("left_flank",     (4.0, 10.0), (4.0, 36.0)),
            ("right_flank",   (28.0, 34.0), (4.0, 36.0)),
            ("paraspinal_l",   (8.0, 14.0), (0.0, 40.0)),
            ("paraspinal_r",   (24.0, 30.0), (0.0, 40.0)),
        ]

        for zone_name, x_range, y_range in ZONE_DEFS:
            # Filter to zone x/y bounds
            x_mask = (all_trunk_local[:, 0] >= x_range[0]) & (
                all_trunk_local[:, 0] < x_range[1]
            )
            y_mask = (all_trunk_local[:, 1] >= y_range[0]) & (
                all_trunk_local[:, 1] < y_range[1]
            )
            zone_mask = x_mask & y_mask
            zone_coords = all_trunk_local[zone_mask]

            if len(zone_coords) == 0:
                logger.warning(f"  Zone '{zone_name}': EMPTY POOL — skipping")
                self._reject_stats["zone_no_voxels"] += 1
                continue

            self._safe_zone_pools[zone_name] = zone_coords.astype(np.float64)
            vol_mm3 = len(zone_coords) * (self.voxel_size_mm ** 3)
            self._zone_volumes[zone_name] = vol_mm3
            logger.info(
                f"  Zone '{zone_name}': {len(zone_coords):,} coords "
                f"({vol_mm3:.1f} mm³ equiv), range X={x_range}, Y={y_range}"
            )

        if not self._safe_zone_pools:
            logger.warning("  ALL zones empty — check depth-filtered subcutaneous coverage")
        else:
            total = sum(len(v) for v in self._safe_zone_pools.values())
            logger.info(f"  Zone pools built: {list(self._safe_zone_pools.keys())}, total={total:,} coords")

    def _build_final_valid_pools(self) -> None:
        """Pre-compute valid positions per radius bin using is_valid_placement.

        S3: Instead of sampling from zone pools and rejecting 80%+ via is_valid_placement,
        pre-filter all zone pool positions for each radius bin and store the valid subsets.
        Sampling then becomes near-100% acceptance.

        Radius bins: small [2.0, 2.5], medium [2.5, 3.0], large [3.0, 3.5]mm.
        For each bin, run is_valid_placement with the bin's representative radius
        (midpoint) on all positions across all zone pools.
        """
        if not self._safe_zone_pools or self.merged_voxel_volume is None:
            return

        # Define radius bins and their representative radii
        radius_bins = [
            ("small",  2.0, 2.5, 2.25),
            ("medium", 2.5, 3.0, 2.75),
            ("large",  3.0, 3.5, 3.25),
        ]

        total_valid = 0
        for bin_name, r_min, r_max, r_rep in radius_bins:
            bin_valid: Dict[str, np.ndarray] = {}
            for zone_name, pool in self._safe_zone_pools.items():
                if len(pool) == 0:
                    continue

                # Test each position with representative radius
                valid_mask = np.zeros(len(pool), dtype=bool)
                for i, pos in enumerate(pool):
                    ok, _ = self.is_valid_placement(pos, r_rep, record_rejections=False)
                    valid_mask[i] = ok

                valid_positions = pool[valid_mask]
                bin_valid[zone_name] = valid_positions.astype(np.float64)
                total_valid += len(valid_positions)

            self._final_valid_pools[bin_name] = bin_valid
            bin_total = sum(len(v) for v in bin_valid.values())
            logger.info(f"  Final-valid pool '{bin_name}' (r={r_min}..{r_max}mm): {bin_total:,} positions")

        logger.info(f"  Final-valid pools total: {total_valid:,} positions across {len(self._final_valid_pools)} bins")

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
                        forced_depth=depth_mm, radius=radius
                    )
                else:
                    # Zone-aware cluster: sample from anchor's zone pool
                    anchor_center = foci[0].center if foci else None
                    candidate_center = self._sample_cluster_position(
                        anchor_center, cluster_radius, radius=radius
                    )

                if candidate_center is None:
                    continue

                # ── is_valid_placement check for anchor + cluster ───────────
                is_valid, reason = self.is_valid_placement(candidate_center, radius)
                if is_valid:
                    pass  # OK
                elif is_anchor:
                    # Anchor: try 20 fallbacks from zone pools
                    self._reject_stats["anchor_is_valid_false"] += 1
                    valid_fallback = None
                    for _ in range(20):
                        fb_center, fb_from_atlas = self._sample_position_with_source(
                            forced_depth=None, radius=radius
                        )
                        if fb_center is None:
                            continue
                        fb_valid, _ = self.is_valid_placement(fb_center, radius)
                        if fb_valid:
                            valid_fallback = (fb_center, False)
                            break
                    if valid_fallback is not None:
                        candidate_center, from_atlas = valid_fallback
                    else:
                        self._reject_stats["anchor_fallback_exhausted"] += 1
                        self._reject_stats["layer1_organ"] += 1
                        candidate_center = None
                        continue
                else:
                    # Cluster: try 20 fallbacks (same zone)
                    self._reject_stats["cluster_is_valid_false"] += 1
                    valid_fallback = None
                    for _ in range(20):
                        fb_center = self._sample_cluster_position(
                            anchor_center, cluster_radius, radius=radius
                        )
                        if fb_center is None:
                            continue
                        fb_valid, _ = self.is_valid_placement(fb_center, radius)
                        if fb_valid:
                            valid_fallback = fb_center
                            break
                    if valid_fallback is not None:
                        candidate_center = valid_fallback
                    else:
                        self._reject_stats["cluster_fallback_exhausted"] += 1
                        self._reject_stats["layer1_organ"] += 1
                        candidate_center = None
                        continue

                params = self._get_shape_params(shape, radius)
                # For atlas-sampled foci, store original atlas coords for debug serialization
                center_atlas = (
                    candidate_center + self.trunk_offset_mm
                    if from_atlas
                    else None
                )
                new_focus = AnalyticFocus(
                    center=candidate_center, shape=shape, params=params,
                    center_atlas_mm=center_atlas,
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
                    foci[0].center if foci else None, cluster_radius, radius=radius
                )
                if candidate_center is not None:
                    # zone pool guarantees safety, but check is_valid_placement
                    # with record_rejections=False since this is a last-resort fallback
                    ivp_ok, _ = self.is_valid_placement(
                        candidate_center, radius, record_rejections=False
                    )
                    if ivp_ok:
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

    def is_valid_placement(
        self, center: np.ndarray, radius: float,
        record_rejections: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """Check if tumor placement is valid; optionally record rejection reason.

        Three criteria (all must pass):
        1. No air voxels in sphere (hard constraint)
        2. body_frac >= 0.85 (≥85% of sphere in tissue, not air)
        3. At most 1 unique organ label (excluding 0=air, 1=skin)

        Parameters
        ----------
        center : np.ndarray
            Tumor center [3] in mm (trunk-local).
        radius : float
            Tumor radius in mm.
        record_rejections : bool
            If True, increment reject_{air,body_frac,multi_organ} counters
            when returning False.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, rejection_reason)
            rejection_reason is None if valid, else one of:
            "air", "body_frac", "multi_organ"
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
            return True, None

        # ── Rule 1: Hard no-air constraint ───────────────────────────────
        air_count = int(np.sum(labels_in_sphere == 0))
        if air_count > 0:
            if record_rejections:
                self._reject_stats["reject_air"] += 1
            return False, "air"

        # ── Rule 2: Body fraction ≥ 0.85 ────────────────────────────────
        body_frac = float(np.sum(labels_in_sphere != 0)) / len(labels_in_sphere)
        if body_frac < 0.85:
            if record_rejections:
                self._reject_stats["reject_body_frac"] += 1
            return False, "body_frac"

        # ── Rule 3: At most 1 unique organ ─────────────────────────────
        organ_labels_in_sphere = labels_in_sphere[
            ~np.isin(labels_in_sphere, [0, 1])
        ]
        unique_organs = len(np.unique(organ_labels_in_sphere))
        if unique_organs > 1:
            if record_rejections:
                self._reject_stats["reject_multi_organ"] += 1
            return False, "multi_organ"

        return True, None

    def _sample_cluster_position(
        self, anchor_center: Optional[np.ndarray], cluster_radius: float,
        radius: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Sample a cluster center from the pre-filtered final-valid zone pool (S3).

        S3: Uses _final_valid_pools (pre-filtered by is_valid_placement) instead of
        _safe_zone_pools. Cluster positions are sampled from the same zone as the anchor
        using the radius-appropriate bin.

        Parameters
        ----------
        anchor_center : np.ndarray or None
            Trunk-local mm center of the anchor focus.
        cluster_radius : float
            Ignored (kept for API compatibility; zone pool IS the constraint).
        radius : float
            Tumor radius for determining radius bin and MCX bbox check.

        Returns
        -------
        np.ndarray or None
            Trunk-local mm center, or None if all final-valid pools are empty.
        """
        # Determine radius bin
        if radius < 2.5:
            r_bin = "small"
        elif radius < 3.0:
            r_bin = "medium"
        else:
            r_bin = "large"

        # Fallback to general sampling if no anchor
        if anchor_center is None:
            pos, _ = self._sample_position_with_source(radius=radius)
            return pos

        # Find which zone the anchor is in (use safe_zone_pools for anchor lookup)
        anchor_zone = None
        for zone_name, pool in self._safe_zone_pools.items():
            if len(pool) == 0:
                continue
            dists = np.linalg.norm(pool[:, :2] - anchor_center[:2], axis=1)
            if dists.min() < 5.0:
                anchor_zone = zone_name
                break

        # Try final-valid pools first (S3: pre-filtered by is_valid_placement)
        if self._final_valid_pools and r_bin in self._final_valid_pools:
            fvp = self._final_valid_pools[r_bin]
            if anchor_zone is not None and anchor_zone in fvp and len(fvp[anchor_zone]) > 0:
                zone_pool = fvp[anchor_zone]
            elif fvp:
                zone_pool = self._rng.choice(list(fvp.values()))
            else:
                zone_pool = None

            if zone_pool is not None and len(zone_pool) > 0:
                idx = self._rng.integers(len(zone_pool))
                candidate = zone_pool[idx].copy()
                # MCX bbox check
                if self.mcx_bbox_mm is not None and not self._organ_constraint_disabled:
                    lo, hi = self.mcx_bbox_mm
                    if self.gt_offset_mm is not None and self.gt_shape is not None:
                        gt_hi = self.gt_offset_mm + self.gt_spacing_mm * np.array(self.gt_shape)
                        strict_lo = np.maximum(lo, self.gt_offset_mm)
                        strict_hi = np.minimum(hi, gt_hi)
                    else:
                        strict_lo, strict_hi = lo, hi
                    if np.any(candidate - radius < strict_lo) or np.any(candidate + radius > strict_hi):
                        return None
                return candidate

        # Fallback to safe_zone_pools (high rejection, but available)
        if anchor_zone is not None and len(self._safe_zone_pools.get(anchor_zone, [])) > 0:
            zone_pool = self._safe_zone_pools[anchor_zone]
        elif self._safe_zone_pools:
            zone_pool = self._rng.choice(list(self._safe_zone_pools.values()))
        else:
            return None

        if zone_pool is None or len(zone_pool) == 0:
            return None

        idx = self._rng.integers(len(zone_pool))
        candidate = zone_pool[idx].copy()

        if self.mcx_bbox_mm is not None and not self._organ_constraint_disabled:
            lo, hi = self.mcx_bbox_mm
            if self.gt_offset_mm is not None and self.gt_shape is not None:
                gt_hi = self.gt_offset_mm + self.gt_spacing_mm * np.array(self.gt_shape)
                strict_lo = np.maximum(lo, self.gt_offset_mm)
                strict_hi = np.minimum(hi, gt_hi)
            else:
                strict_lo, strict_hi = lo, hi
            if np.any(candidate - radius < strict_lo) or np.any(candidate + radius > strict_hi):
                return None

        return candidate

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
        self, forced_depth: Optional[float] = None, radius: Optional[float] = None
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
                forced_depth=forced_depth, radius=radius
            )
            return pos, used_atlas
        else:
            pos = self._sample_from_config(forced_depth=forced_depth)
            return pos, False

    def _sample_from_atlas_with_flag(
        self, forced_depth: Optional[float] = None, radius: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """Sample anchor center from pre-filtered final-valid zone pools (S3).

        S3: Uses _final_valid_pools (pre-filtered by is_valid_placement) instead of
        _safe_zone_pools. This gives near-100% acceptance instead of ~18%.

        Radius-aware: selects the appropriate radius bin to sample from.

        Parameters
        ----------
        forced_depth : float, optional
            Ignored for zone-based sampling (zone pools already guarantee depth).
        radius : float, optional
            Tumor radius in mm. Determines which radius bin to sample from.

        Returns
        -------
        Tuple[np.ndarray, bool]
            (position [3] in trunk-local mm, actually_from_atlas bool)
            Returns (config_position, False) if all final-valid pools are empty.
        """
        # Determine radius bin
        if radius is None:
            r_bin = "medium"  # default
        elif radius < 2.5:
            r_bin = "small"
        elif radius < 3.0:
            r_bin = "medium"
        else:
            r_bin = "large"

        # Try final-valid pools first (pre-filtered by is_valid_placement)
        if self._final_valid_pools and r_bin in self._final_valid_pools:
            fvp = self._final_valid_pools[r_bin]
            if fvp:
                # Weighted random zone selection by valid pool size
                zone_names = list(fvp.keys())
                zone_sizes = np.array([len(fvp[n]) for n in zone_names])
                if zone_sizes.sum() == 0:
                    return self._sample_from_config(forced_depth=forced_depth), False
                weights = zone_sizes.astype(float) / zone_sizes.sum()
                zone_name = self._rng.choice(zone_names, p=weights)
                pool = fvp[zone_name]
                if len(pool) > 0:
                    idx = self._rng.integers(len(pool))
                    return pool[idx].copy(), True

        # Fallback to safe_zone_pools (old behavior, high rejection)
        if not self._safe_zone_pools:
            return self._sample_from_config(forced_depth=forced_depth), False

        zone_names = list(self._safe_zone_pools.keys())
        zone_sizes = np.array([len(self._safe_zone_pools[n]) for n in zone_names])
        weights = zone_sizes.astype(float) / zone_sizes.sum()
        zone_name = self._rng.choice(zone_names, p=weights)
        pool = self._safe_zone_pools[zone_name]

        if len(pool) == 0:
            return self._sample_from_config(forced_depth=forced_depth), False

        idx = self._rng.integers(len(pool))
        return pool[idx].copy(), True

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
