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

        # ── U7-Fast: rejection diagnostic counters ──────────────────
        self._reject_stats = {
            "reject_air": 0,
            "reject_bone": 0,
            "reject_soft_tissue_frac": 0,
            "reject_depth": 0,
            "reject_duplicate_exact": 0,
            "reject_too_close": 0,
            "layer2_mcx_bbox": 0,
            "max_attempts_hit": 0,
            "accepted": 0,
        }

        # ── U7-Fast: diversity registry ────────────────────────────
        self._accepted_centers: list = []
        self._accepted_radii: list = []

        # ── Precompute EDT depth map from merged volume ─────────────
        # depth_edt[x, y, z] = distance in mm from nearest body surface (label != 0)
        self._edt_depth_mm: Optional[np.ndarray] = None
        if merged_voxel_volume is not None:
            self._compute_edt_depth(merged_voxel_volume, voxel_size_mm)

        # ── Zone-pool methods disabled in U7-Fast (kept as no-ops for compatibility) ──
        self._safe_zone_pools = {}
        self._zone_volumes = {}

    # ── U7-Fast: new sampling and validation methods ──────────────────────────────────

    def _compute_edt_depth(self, vol: np.ndarray, voxel_size_mm: float) -> None:
        """Precompute EDT depth map: min distance from each voxel to body surface.

        Body surface = voxels with label != 0. EDT gives distance in voxels;
        multiply by voxel_size_mm to get mm.
        """
        from scipy.ndimage import distance_transform_edt
        body_mask = vol != 0
        # EDT: distance from each voxel to nearest background (label==0)
        # We want distance FROM body TO surface = distance to nearest non-body voxel
        # But we want depth = distance from surface inward = for interior voxels,
        # distance to the nearest surface voxel (label != 0 but near boundary)
        # Simplified: use distance to nearest ZERO voxel as proxy for depth
        # (0 = outside body, non-zero = inside body)
        dist = distance_transform_edt(vol == 0, sampling=[voxel_size_mm] * 3)
        self._edt_depth_mm = dist
        logger.info(f"  EDT depth map computed: shape={dist.shape}, "
                    f"max_depth={dist.max():.1f}mm")

    def _sample_random_position(
        self, radius: float, region: str = "dorsal"
    ) -> Optional[np.ndarray]:
        """U7-Fast: random position sampling in merged voxel space.

        Samples in merged voxel indices [X, Y, Z] then converts to trunk-local mm.
        region: 'dorsal' (80%) or 'lateral' (20%) — sampling bias, not hard constraint.

        Dorsal band: X in [40, 150], Z in [15, 80]  (centered in body, mid depth)
        Lateral band: X in [5, 35] or [155, 185], Z in [20, 70]

        Returns center in trunk-local mm, or None if out of bounds.
        """
        vs = self.voxel_size_mm
        x_dim, y_dim, z_dim = self.merged_voxel_volume.shape

        # Sampling regions in voxel space
        if region == "dorsal":
            # Core dorsal: X broad, Y center, Z mid-body
            x_lo, x_hi = 30, x_dim - 30
            y_lo, y_hi = 20, y_dim - 20
            z_lo, z_hi = 15, 80
        else:  # lateral
            # Left or right flank
            side = self._rng.choice([-1, 1])
            if side == -1:
                x_lo, x_hi = 5, 35
            else:
                x_lo, x_hi = x_dim - 35, x_dim - 5
            y_lo, y_hi = 20, y_dim - 20
            z_lo, z_hi = 20, 70

        x_v = self._rng.integers(x_lo, x_hi)
        y_v = self._rng.integers(y_lo, y_hi)
        z_v = self._rng.integers(z_lo, z_hi)

        # Convert to trunk-local mm (direct indexing)
        center = np.array([x_v * vs, y_v * vs, z_v * vs], dtype=np.float64)
        return center

    def _too_close_to_any(self, center: np.ndarray, radius: float) -> Tuple[bool, float]:
        """Check min distance to previously accepted tumors.

        Separation: min_dist >= max(2 * max(r_i, r_j) + 1.0, 4.0)mm.
        """
        if not self._accepted_centers:
            return False, float('inf')
        c = np.asarray(center)
        r = radius
        min_dist = float('inf')
        for ec, er in zip(self._accepted_centers, self._accepted_radii):
            dist = np.linalg.norm(c - ec)
            required = max(2.0 * max(r, er) + 1.0, 4.0)
            if dist < required:
                return True, dist
            min_dist = min(min_dist, dist)
        return False, min_dist

    def is_valid_fast(
        self, center: np.ndarray, radius: float,
        record_rejections: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """U7-Fast 4-rule validation.

        Rules:
        A. No air:     count(label == 0) == 0
        B. No bone:    count(label == 2) == 0
        C. Soft tissue majority: frac(label == 1) >= 0.5
        D. Shallow:    2.0 <= depth_mm <= 6.0

        Parameters
        ----------
        center : np.ndarray
            Tumor center [3] in trunk-local mm.
        radius : float
            Tumor radius in mm.
        record_rejections : bool
            If True, increment rejection counters.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, rejection_reason)
        """
        if self._organ_constraint_disabled:
            return True, None
        if self.merged_voxel_volume is None:
            return True, None

        vol = self.merged_voxel_volume
        vs = self.voxel_size_mm
        vol_shape = vol.shape  # [X, Y, Z]

        # Trunk-local center → voxel indices (direct, empirically verified)
        cx, cy, cz = center[0] / vs, center[1] / vs, center[2] / vs
        r_vox = radius / vs

        # Bounding box
        x_min = max(0, int(cx - r_vox) - 1)
        x_max = min(vol_shape[0], int(cx + r_vox) + 2)
        y_min = max(0, int(cy - r_vox) - 1)
        y_max = min(vol_shape[1], int(cy + r_vox) + 2)
        z_min = max(0, int(cz - r_vox) - 1)
        z_max = min(vol_shape[2], int(cz + r_vox) + 2)

        subvol = vol[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        sx, sy, sz = np.meshgrid(
            np.arange(subvol.shape[0]),
            np.arange(subvol.shape[1]),
            np.arange(subvol.shape[2]),
            indexing='ij',
        )
        dist = np.sqrt(
            (sx - (cx - x_min)) ** 2
            + (sy - (cy - y_min)) ** 2
            + (sz - (cz - z_min)) ** 2
        )
        mask = dist <= r_vox
        labels_in_sphere = subvol[mask]

        if labels_in_sphere.size == 0:
            return True, None

        # ── Rule A: no air ───────────────────────────────────────────
        air_count = int(np.sum(labels_in_sphere == 0))
        if air_count > 0:
            if record_rejections:
                self._reject_stats["reject_air"] += 1
            return False, "air"

        # ── Rule B: no bone ──────────────────────────────────────────
        bone_count = int(np.sum(labels_in_sphere == 2))
        if bone_count > 0:
            if record_rejections:
                self._reject_stats["reject_bone"] += 1
            return False, "bone"

        # ── Rule C: soft tissue majority ─────────────────────────────
        soft_count = int(np.sum(labels_in_sphere == 1))
        soft_frac = soft_count / len(labels_in_sphere)
        if soft_frac < 0.5:
            if record_rejections:
                self._reject_stats["reject_soft_tissue_frac"] += 1
            return False, "soft_tissue_frac"

        # ── Rule D: shallow depth ────────────────────────────────────
        # EDT depth at sphere center
        xv, yv, zv = int(cx), int(cy), int(cz)
        if (0 <= xv < vol_shape[0] and 0 <= yv < vol_shape[1]
                and 0 <= zv < vol_shape[2] and self._edt_depth_mm is not None):
            depth_mm = float(self._edt_depth_mm[xv, yv, zv])
        else:
            # Fallback: use Z position relative to body surface estimate
            depth_mm = cz * vs - 1.8  # body surface at Z=1.8mm trunk

        if not (2.0 <= depth_mm <= 6.5):
            if record_rejections:
                self._reject_stats["reject_depth"] += 1
            return False, "depth"

        return True, None

    def _register_accepted(self, center: np.ndarray, radius: float) -> None:
        """Register accepted tumor in diversity registry."""
        self._accepted_centers.append(np.asarray(center, dtype=np.float64))
    def generate_sample(
        self,
        num_foci: Optional[int] = None,
        depth_mm: Optional[float] = None,
        depth_tier: Optional[str] = None,
    ) -> TumorSample:
        """U7-Fast: Generate a single tumor sample via random sampling + fast validation.

        Pipeline:
        1. Sample random candidate center (80% dorsal / 20% lateral bias)
        2. Validate with is_valid_fast (4 rules: no air, no bone, soft≥50%, depth 2-6mm)
        3. Check diversity (no exact duplicate, min distance to existing)
        4. Repeat up to 200 attempts per focus

        Parameters
        ----------
        num_foci : int, optional
            If provided, forces the exact number of foci.
        depth_mm : float, optional
            Ignored in U7-Fast (depth is validated by is_valid_fast Rule D).
        depth_tier : str, optional
            Depth tier name for reporting.

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
        organ_constraint_passed = True

        for focus_idx in range(num_foci):
            center = None
            is_anchor = focus_idx == 0
            attempts = 0
            max_attempts = 200

            while center is None and attempts < max_attempts:
                attempts += 1

                # ── Step 1: random candidate ─────────────────────────────
                if is_anchor:
                    # 80% dorsal, 20% lateral sampling bias
                    region = "dorsal" if self._rng.random() < 0.8 else "lateral"
                    candidate_center = self._sample_random_position(radius, region=region)
                else:
                    # Cluster: random offset around anchor (up to 6mm)
                    anchor = foci[0].center
                    angle = self._rng.uniform(0, 2 * np.pi)
                    dist = self._rng.uniform(1.0, 6.0)
                    offset = np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0])
                    candidate_center = (np.asarray(anchor, dtype=np.float64) + offset)

                if candidate_center is None:
                    continue

                # ── Step 2: fast validation ──────────────────────────────
                ok, reason = self.is_valid_fast(candidate_center, radius)
                if not ok:
                    continue

                # ── Step 3: diversity checks ─────────────────────────────
                # Check exact duplicate (discretized key)
                key = (round(candidate_center[0], 1), round(candidate_center[1], 1),
                       round(candidate_center[2], 1), round(radius, 1))
                if key in self._exact_keys_set():
                    self._reject_stats["reject_duplicate_exact"] += 1
                    continue

                # Check min distance to existing
                too_close, min_dist = self._too_close_to_any(candidate_center, radius)
                if too_close:
                    self._reject_stats["reject_too_close"] += 1
                    continue

                # All checks passed
                params = self._get_shape_params(shape, radius)
                new_focus = AnalyticFocus(
                    center=candidate_center, shape=shape, params=params,
                    center_atlas_mm=None,
                )
                test_foci = foci + [new_focus]

                if self._check_constraints(test_foci):
                    foci.append(new_focus)
                    center = candidate_center

            if center is None:
                if is_anchor:
                    self._reject_stats["max_attempts_hit"] += 1
                organ_constraint_passed = False
                break

        sample = TumorSample(foci=foci)
        sample._depth_tier = depth_tier
        sample._depth_mm = depth_mm
        sample._organ_constraint_passed = organ_constraint_passed

        # ── MCX bbox check ───────────────────────────────────────────
        if not self._organ_constraint_disabled and self.mcx_bbox_mm is not None:
            lo, hi = self.mcx_bbox_mm
            if self.gt_offset_mm is not None and self.gt_shape is not None:
                gt_hi = self.gt_offset_mm + self.gt_spacing_mm * np.array(self.gt_shape)
                strict_lo = np.maximum(lo, self.gt_offset_mm)
                strict_hi = np.minimum(hi, gt_hi)
            else:
                strict_lo, strict_hi = lo, hi
            for focus in sample.foci:
                c = np.asarray(focus.center)
                r = focus.params.get("radius", 1.0)
                if np.any(c - r < strict_lo) or np.any(c + r > strict_hi):
                    self._reject_stats["layer2_mcx_bbox"] += 1
                    sample._organ_constraint_passed = False
                    break

        if sample._organ_constraint_passed:
            self._reject_stats["accepted"] += 1
            # Register in diversity registry
            for focus in sample.foci:
                c = np.asarray(focus.center)
                r = focus.params.get("radius", 1.0)
                self._register_accepted(c, r)
                self._add_exact_key(c, r)

        return sample

    def _exact_keys_set(self) -> set:
        """Return the set of exact keys (creates on first call)."""
        if not hasattr(self, '_exact_keys'):
            self._exact_keys = set()
        return self._exact_keys

    def _add_exact_key(self, center: np.ndarray, radius: float) -> None:
        """Add an exact key to the dedup set."""
        key = (round(center[0], 1), round(center[1], 1),
               round(center[2], 1), round(radius, 1))
        self._exact_keys_set().add(key)

