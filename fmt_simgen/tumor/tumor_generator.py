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
        """Evaluate Gaussian sphere: exp(-r^2 / (2 * sigma^2))."""
        r = self.params.get("radius", 1.0)
        sigma = r / 2.0

        distances = np.linalg.norm(coords - self.center, axis=1)
        return np.exp(-(distances**2) / (2.0 * sigma**2))

    def _evaluate_ellipsoid(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian ellipsoid with axis ratios."""
        radii = np.array(
            [
                self.params.get("rx", 1.0),
                self.params.get("ry", 1.0),
                self.params.get("rz", 1.0),
            ]
        )
        sigma = radii / 2.0

        diff = coords - self.center
        normalized_dist = diff / sigma
        return np.exp(-0.5 * np.sum(normalized_dist**2, axis=1))


@dataclass
class TumorSample:
    """A tumor sample containing multiple foci."""

    foci: List[AnalyticFocus] = field(default_factory=list)

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate total tumor distribution at given coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates [N×3] at which to evaluate.

        Returns
        -------
        np.ndarray
            Combined tumor values [N] (sum of all foci).
        """
        if not self.foci:
            return np.zeros(coords.shape[0])

        values = np.zeros(coords.shape[0])
        for focus in self.foci:
            values += focus.evaluate(coords)

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

    def __init__(self, config: Dict, atlas=None):
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
        atlas : DigimouseAtlas, optional
            Atlas for region sampling. If None, use config-based ranges.
        """
        self.config = config
        self.atlas = atlas

        self.regions = config.get("regions", ["dorsal", "lateral"])
        self.num_foci_dist = config.get(
            "num_foci_distribution", {1: 0.30, 2: 0.35, 3: 0.35}
        )
        self.shapes = [
            ShapeType(s) for s in config.get("shapes", ["sphere", "ellipsoid"])
        ]
        self.radius_range = config.get("radius_range", [0.3, 1.0])
        self.depth_range = config.get("depth_range", [1.0, 3.0])
        self.min_inter_ratio = config.get("min_inter_foci_distance_ratio", 1.5)
        self.ellipsoid_ratio = config.get("ellipsoid_axis_ratio", [1.2, 1.5])

        self._rng = np.random.default_rng()

    def generate_sample(self) -> TumorSample:
        """Generate a single tumor sample.

        Returns
        -------
        TumorSample
            Generated tumor sample with random foci.
        """
        num_foci = self._sample_num_foci()
        shape = self._rng.choice(self.shapes)
        radius = self._rng.uniform(*self.radius_range)

        foci: List[AnalyticFocus] = []
        attempts = 0
        max_attempts = 100

        while len(foci) < num_foci and attempts < max_attempts:
            attempts += 1
            center, from_atlas = self._sample_position_with_source()

            if not from_atlas:
                depth = self._get_depth_at_position(center)
                while depth < self.depth_range[0] or depth > self.depth_range[1]:
                    center, from_atlas = self._sample_position_with_source()
                    if from_atlas:
                        break
                    depth = self._get_depth_at_position(center)

            params = self._get_shape_params(shape, radius)

            new_focus = AnalyticFocus(center=center, shape=shape, params=params)
            test_foci = foci + [new_focus]

            if self._check_constraints(test_foci):
                foci.append(new_focus)

        return TumorSample(foci=foci)

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
            (position [3] in mm, from_atlas bool)
        """
        if self.atlas is not None:
            return self._sample_from_atlas(), True
        else:
            return self._sample_from_config(), False

    def _sample_from_atlas(self) -> np.ndarray:
        """Sample position from atlas subcutaneous region."""
        region = self._rng.choice(self.regions)

        subq_mask = self.atlas.get_subcutaneous_region(
            depth_range_mm=tuple(self.depth_range),
            regions=[region],
        )

        if not np.any(subq_mask):
            return self._sample_from_config()

        voxel_coords = np.argwhere(subq_mask)
        idx = self._rng.integers(len(voxel_coords))

        x, y, z = voxel_coords[idx]
        voxel_size = self.atlas.voxel_size

        return np.array(
            [x * voxel_size, y * voxel_size, z * voxel_size], dtype=np.float64
        )

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
            return np.array([
                self._rng.uniform(10.0, 28.0),
                self._rng.uniform(20.0, 70.0),
                self._rng.uniform(15.0, 19.0),
            ], dtype=np.float64)
        elif region == "lateral":
            x_side = self._rng.choice([
                self._rng.uniform(3.0, 8.0),
                self._rng.uniform(28.0, 33.0),
            ])
            return np.array([
                x_side,
                self._rng.uniform(20.0, 70.0),
                self._rng.uniform(8.0, 14.0),
            ], dtype=np.float64)
        else:
            return np.array([
                self._rng.uniform(10.0, 28.0),
                self._rng.uniform(20.0, 70.0),
                self._rng.uniform(2.0, 5.0),
            ], dtype=np.float64)

    def _get_depth_at_position(self, position: np.ndarray) -> float:
        """Estimate depth from surface at given position.

        Parameters
        ----------
        position : np.ndarray
            Position [3] in mm.

        Returns
        -------
        float
            Estimated depth in mm (positive value).
        """
        if self.atlas is None:
            return abs(position[2])

        return abs(position[2])

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
                min_dist = self.min_inter_ratio * (r_i + r_j)

                if dist < min_dist:
                    return False

        return True
