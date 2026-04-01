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

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ShapeType(Enum):
    """Supported tumor shape types."""

    SPHERE = "sphere"
    ELLIPSOID = "ellipsoid"


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
        if self.shape == ShapeType.SPHERE:
            return self._evaluate_sphere(coords)
        elif self.shape == ShapeType.ELLIPSOID:
            return self._evaluate_ellipsoid(coords)
        else:
            raise ValueError(f"Unknown shape type: {self.shape}")

    def _evaluate_sphere(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian sphere with sigma = radius, truncated at 4*sigma.

        d(x) = exp(-||x - center||^2 / (2 * sigma^2)) for ||x - center|| <= 4*sigma
               0 otherwise
        """
        radius = self.params.get("radius", 1.0)
        sigma = radius
        cutoff = 4.0 * sigma

        distances = np.linalg.norm(coords - self.center, axis=1)
        values = np.exp(-(distances**2) / (2.0 * sigma**2))
        values = np.where(distances <= cutoff, values, 0.0)
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
            ]
        )
        sigma = radii
        cutoff = 4.0 * np.max(sigma)

        diff = coords - self.center
        normalized_dist = diff / sigma
        distances = np.sqrt(np.sum(normalized_dist**2, axis=1))
        values = np.exp(-0.5 * np.sum(normalized_dist**2, axis=1))
        values = np.where(distances <= cutoff, values, 0.0)
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
            "foci": [
                {
                    "center": focus.center.tolist(),
                    "shape": focus.shape.value,
                    "params": focus.params,
                }
                for focus in self.foci
            ],
        }


class TumorGenerator:
    """Generate tumor samples as collections of analytic foci."""

    def __init__(self, config: Dict, atlas=None, mesh_bbox=None):
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
            - min_inter_foci_distance_ratio: float - ratio * 灶直径
            - ellipsoid_axis_ratio: List[float, float] - [rx_ratio, rz_ratio]
            - max_cluster_radius: float - max distance from anchor for multi-focus
            - min_foci_distance_abs: float - min absolute inter-foci distance (mm)
        atlas : DigimouseAtlas, optional
            Atlas for region sampling. If None, use config-based ranges.
        mesh_bbox : Dict, optional
            Mesh bounding box for depth estimation.
            {"min": [x_min, y_min, z_min], "max": [x_max, y_max, z_max]}
        """
        self.config = config
        self.atlas = atlas
        self.mesh_bbox = mesh_bbox

        self.regions = config.get("regions", ["dorsal", "lateral"])
        self.num_foci_dist = config.get(
            "num_foci_distribution", {1: 0.30, 2: 0.35, 3: 0.35}
        )
        self.shapes = [
            ShapeType(s) for s in config.get("shapes", ["sphere", "ellipsoid"])
        ]
        self.radius_range = config.get("radius_range", [1.0, 2.5])
        self.depth_range = config.get("depth_range", [1.5, 4.0])
        self.min_inter_ratio = config.get("min_inter_foci_distance_ratio", 1.5)
        self.ellipsoid_ratio = config.get("ellipsoid_axis_ratio", [1.2, 1.5])
        self.max_cluster_radius = config.get("max_cluster_radius", 8.0)
        self.min_foci_distance_abs = config.get("min_foci_distance_abs", 3.0)

        self._rng = np.random.default_rng()

    def generate_sample(self) -> TumorSample:
        """Generate a single tumor sample.

        For multi-focal samples, uses cluster-based placement where
        additional foci are placed near the anchor focus.

        Returns
        -------
        TumorSample
            Generated tumor sample with random foci.
        """
        num_foci = self._sample_num_foci()
        shape = self._rng.choice(self.shapes)
        radius = self._rng.uniform(*self.radius_range)

        foci: List[AnalyticFocus] = []
        cluster_radius = self.max_cluster_radius

        for focus_idx in range(num_foci):
            center = None
            is_anchor = focus_idx == 0
            attempts = 0
            max_attempts = 100

            while center is None and attempts < max_attempts:
                attempts += 1

                if is_anchor:
                    candidate_center, from_atlas = self._sample_position_with_source()
                else:
                    candidate_center = self._sample_cluster_position(
                        foci[0] if foci else None, cluster_radius
                    )

                if candidate_center is None:
                    continue

                if not from_atlas if is_anchor else True:
                    depth = self._get_depth_at_position(candidate_center)
                    if depth < self.depth_range[0] or depth > self.depth_range[1]:
                        if is_anchor:
                            candidate_center, from_atlas = (
                                self._sample_position_with_source()
                            )
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

        return TumorSample(foci=foci)

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

    def _sample_position_with_source(self) -> Tuple[np.ndarray, bool]:
        """Sample a position and return whether it came from atlas.

        Returns
        -------
        Tuple[np.ndarray, bool]
            (position [3] in mm, actually_from_atlas bool)
        """
        if self.atlas is not None:
            pos, used_atlas = self._sample_from_atlas_with_flag()
            return pos, used_atlas
        else:
            return self._sample_from_config(), False

    def _sample_from_atlas_with_flag(self) -> Tuple[np.ndarray, bool]:
        """Attempt to sample from atlas subcutaneous region.

        Returns
        -------
        Tuple[np.ndarray, bool]
            (position [3] in mm, actually_from_atlas bool)
            Returns (config_position, False) if atlas sampling fails.
        """
        region = self._rng.choice(self.regions)

        subq_mask = self.atlas.get_subcutaneous_region(
            depth_range_mm=tuple(self.depth_range),
            regions=[region],
        )

        if not np.any(subq_mask):
            return self._sample_from_config(), False

        voxel_coords = np.argwhere(subq_mask)
        idx = self._rng.integers(len(voxel_coords))

        x, y, z = voxel_coords[idx]
        voxel_size = self.atlas.voxel_size

        return np.array(
            [x * voxel_size, y * voxel_size, z * voxel_size], dtype=np.float64
        ), True

    def _sample_from_config(self) -> np.ndarray:
        """Sample position from config-based ranges (fallback).

        Mesh coordinate ranges (0-based physical coords):
        X: [2.4, 34.4] mm, Y: [4.8, 92.8] mm, Z: [1.6, 20.0] mm

        Dorsal side = high Z values (Z > 12mm)
        Lateral sides = extreme X values (X < 8 or X > 28)
        Trunk region = Y in [20, 70] mm (exclude head and tail)
        """
        region = self._rng.choice(self.regions) if self.regions else "dorsal"

        if region == "dorsal":
            return np.array(
                [
                    self._rng.uniform(10.0, 28.0),
                    self._rng.uniform(20.0, 70.0),
                    self._rng.uniform(15.0, 19.0),
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
            return np.array(
                [
                    x_side,
                    self._rng.uniform(20.0, 70.0),
                    self._rng.uniform(8.0, 14.0),
                ],
                dtype=np.float64,
            )
        else:
            return np.array(
                [
                    self._rng.uniform(10.0, 28.0),
                    self._rng.uniform(20.0, 70.0),
                    self._rng.uniform(2.0, 5.0),
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
        if shape == ShapeType.SPHERE:
            return {"radius": base_radius}
        elif shape == ShapeType.ELLIPSOID:
            rx_ratio, rz_ratio = self.ellipsoid_ratio
            return {
                "rx": base_radius * rx_ratio,
                "ry": base_radius,
                "rz": base_radius * rz_ratio,
            }
        else:
            return {"radius": base_radius}

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
