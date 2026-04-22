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
            Kept for diagnostics only in U7-Fast (not used as placement
            hard gate).
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
        self._candidate_debug_printed = 0

        # ── Precompute EDT depth map from merged volume ─────────────
        # depth_edt[x, y, z] = inward depth (mm) for body voxels (label != 0)
        self._edt_depth_mm: Optional[np.ndarray] = None
        if merged_voxel_volume is not None:
            self._compute_edt_depth(merged_voxel_volume, voxel_size_mm)

        # ── Zone-pool methods disabled in U7-Fast (kept as no-ops for compatibility) ──
        self._safe_zone_pools = {}
        self._zone_volumes = {}

    # ── U7-Fast: new sampling and validation methods ──────────────────────────────────

    def _build_safe_zone_pools(self) -> None:
        """LEGACY - NOT USED IN U7-Fast.

        Deprecated zone-pool builder retained only for compatibility.
        """
        return

    def _build_final_valid_pools(self) -> None:
        """LEGACY - NOT USED IN U7-Fast.

        Deprecated final-valid pool builder retained only for compatibility.
        """
        return

    def _sample_from_atlas_with_flag(self, *args, **kwargs):
        """LEGACY - NOT USED IN U7-Fast.

        Deprecated atlas-guided sampling entry retained only for compatibility.
        """
        return None

    def _sample_cluster_position(self, *args, **kwargs):
        """LEGACY - NOT USED IN U7-Fast.

        Deprecated cluster sampler retained only for compatibility.
        """
        return None

    def is_valid_placement(
        self,
        center: np.ndarray,
        radius: float,
        record_rejections: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """LEGACY - NOT USED IN U7-Fast.

        Compatibility wrapper that routes to U7-Fast 4-rule validation.
        """
        return self.is_valid_fast(center, radius, record_rejections=record_rejections)

    def _get_mcx_bbox_mm(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return MCX trunk-local bbox for placement checks."""
        if self.mcx_bbox_mm is None:
            return None

        lo, hi = self.mcx_bbox_mm
        mcx_lo = np.asarray(lo, dtype=np.float64)
        mcx_hi = np.asarray(hi, dtype=np.float64)

        if np.any(mcx_lo >= mcx_hi):
            logger.warning(
                "Invalid MCX bbox: lo=%s, hi=%s",
                mcx_lo.tolist(),
                mcx_hi.tolist(),
            )
            return None
        return mcx_lo, mcx_hi

    def _compute_edt_depth(self, vol: np.ndarray, voxel_size_mm: float) -> None:
        """Precompute inward depth map in millimeters from merged tissue mask.

        Input mask:
            body_mask = (vol != 0), where non-zero labels are inside the trunk
            and zero is background/air.

        Output semantics:
            depth_map[x, y, z] is the distance (mm) from a body voxel to the
            nearest background voxel. This is inward depth from surface.

        Background voxels:
            depth is exactly 0 mm by definition.
        """
        from scipy.ndimage import distance_transform_edt

        body_mask = vol != 0
        depth_mm = distance_transform_edt(body_mask, sampling=[voxel_size_mm] * 3)
        self._edt_depth_mm = depth_mm

        logger.info(
            "  EDT depth map computed: shape=%s, min=%.3fmm, max=%.3fmm",
            depth_mm.shape,
            float(depth_mm.min()),
            float(depth_mm.max()),
        )

        body_depth = depth_mm[body_mask]
        if body_depth.size == 0:
            logger.warning("  EDT depth diagnostics skipped: no body voxels in merged volume")
            return

        p1, p5, p50, p95, p99 = np.percentile(body_depth, [1, 5, 50, 95, 99])
        logger.info(
            "  EDT body depth percentiles mm: p1=%.3f p5=%.3f p50=%.3f p95=%.3f p99=%.3f",
            float(p1),
            float(p5),
            float(p50),
            float(p95),
            float(p99),
        )

        trunk_interior_vox = np.argwhere(body_mask & (depth_mm > 0.0))
        if trunk_interior_vox.shape[0] == 0:
            logger.warning("  EDT depth probes skipped: no interior body voxels with depth>0")
            return

        n_probe = min(10, trunk_interior_vox.shape[0])
        probe_ids = self._rng.choice(trunk_interior_vox.shape[0], size=n_probe, replace=False)
        probe_logs = []
        for probe_idx in probe_ids:
            x_v, y_v, z_v = trunk_interior_vox[probe_idx]
            probe_logs.append(f"({x_v},{y_v},{z_v})->{depth_mm[x_v, y_v, z_v]:.3f}mm")
        logger.info("  EDT interior depth probes: %s", "; ".join(probe_logs))

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

        # Clamp to volume bounds first
        x_lo, x_hi = max(0, x_lo), min(x_dim, x_hi)
        y_lo, y_hi = max(0, y_lo), min(y_dim, y_hi)
        z_lo, z_hi = max(0, z_lo), min(z_dim, z_hi)

        # Enforce radius-aware MCX bbox at sampling time:
        # candidate - radius >= mcx_lo and candidate + radius <= mcx_hi.
        mcx_bbox = self._get_mcx_bbox_mm()
        if mcx_bbox is not None:
            mcx_lo, mcx_hi = mcx_bbox
            center_lo_mm = mcx_lo + radius
            center_hi_mm = mcx_hi - radius
            if np.any(center_lo_mm > center_hi_mm):
                return None

            center_lo_vox = np.ceil(center_lo_mm / vs).astype(np.int64)
            center_hi_vox = np.floor(center_hi_mm / vs).astype(np.int64)

            x_lo = max(x_lo, int(center_lo_vox[0]))
            x_hi = min(x_hi, int(center_hi_vox[0]) + 1)
            y_lo = max(y_lo, int(center_lo_vox[1]))
            y_hi = min(y_hi, int(center_hi_vox[1]) + 1)
            z_lo = max(z_lo, int(center_lo_vox[2]))
            z_hi = min(z_hi, int(center_hi_vox[2]) + 1)

        if x_lo >= x_hi or y_lo >= y_hi or z_lo >= z_hi:
            return None

        x_v = self._rng.integers(x_lo, x_hi)
        y_v = self._rng.integers(y_lo, y_hi)
        z_v = self._rng.integers(z_lo, z_hi)
        center_label = int(self.merged_voxel_volume[x_v, y_v, z_v])
        if self._candidate_debug_printed < 12:
            print(
                f"    [CAND] #{self._candidate_debug_printed + 1}: "
                f"voxel=({x_v},{y_v},{z_v}), center_label={center_label}",
                flush=True,
            )
            self._candidate_debug_printed += 1

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
        self._accepted_radii.append(float(radius))

    def _sample_num_foci(self) -> int:
        """Sample number of foci from configured distribution."""
        keys = list(self.num_foci_dist.keys())
        probs = np.asarray(list(self.num_foci_dist.values()), dtype=np.float64)
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            raise ValueError("num_foci_distribution must have positive probability sum")
        probs = probs / probs_sum
        return int(self._rng.choice(keys, p=probs))

    def _get_shape_params(self, shape: ShapeType, radius: float) -> Dict:
        """Build shape parameters for a single focus."""
        source_type = self.config.get("source_type", "gaussian")
        if shape == ShapeType.SPHERE:
            return {"radius": float(radius), "source_type": source_type}

        # Ellipsoid: anisotropy sampled from configured ratio range.
        ratio_lo, ratio_hi = self.ellipsoid_ratio
        rx_ratio = float(self._rng.uniform(ratio_lo, ratio_hi))
        rz_ratio = float(self._rng.uniform(ratio_lo, ratio_hi))
        return {
            "radius": float(radius),
            "rx": float(radius * rx_ratio),
            "ry": float(radius),
            "rz": float(radius * rz_ratio),
            "source_type": source_type,
        }

    def _check_constraints(self, foci: List[AnalyticFocus]) -> bool:
        """U7-Fast: check inter-foci distance (no overlap).

        Ensures each new focus is at least min_foci_distance_abs mm from all
        existing foci, using actual sphere radii for true non-overlap.
        """
        if len(foci) < 2:
            return True
        new_focus = foci[-1]
        c_new = np.asarray(new_focus.center)
        r_new = float(new_focus.params.get("radius", 1.0))
        for existing in foci[:-1]:
            c_ex = np.asarray(existing.center)
            r_ex = float(existing.params.get("radius", 1.0))
            dist = float(np.linalg.norm(c_new - c_ex))
            min_dist = r_new + r_ex  # true non-overlap: centers must be >= sum of radii
            min_dist = max(min_dist, self.min_foci_distance_abs)  # but respect config floor
            if dist < min_dist:
                return False
        return True

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
        3. Check inter-foci distance (no overlap) and exact dedup
        4. Accept sample

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
            Generated tumor sample with 1-N foci (multi-foci supported).
        """
        if num_foci is None:
            num_foci = self._sample_num_foci()

        foci: List[AnalyticFocus] = []
        organ_constraint_passed = True

        for focus_idx in range(num_foci):
            # Each focus gets its own shape and radius
            shape = self._rng.choice(self.shapes)
            radius = self._rng.uniform(*self.radius_range)
            center = None
            is_anchor = focus_idx == 0
            attempts = 0
            max_attempts = 500

            while center is None and attempts < max_attempts:
                attempts += 1

                # ── Step 1: random candidate ─────────────────────────────
                # 80% dorsal, 20% lateral sampling bias
                region = "dorsal" if self._rng.random() < 0.8 else "lateral"
                candidate_center = self._sample_random_position(radius, region=region)

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
        mcx_bbox = self._get_mcx_bbox_mm()
        if not self._organ_constraint_disabled and mcx_bbox is not None:
            mcx_lo, mcx_hi = mcx_bbox
            for focus in sample.foci:
                c = np.asarray(focus.center)
                r = focus.params.get("radius", 1.0)
                if np.any(c - r < mcx_lo) or np.any(c + r > mcx_hi):
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
