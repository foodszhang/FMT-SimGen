#!/usr/bin/env python3
"""Source model abstraction for E1d.

Supports:
- PointSource: single point source
- UniformEllipsoidSource: uniform intensity within ellipsoid
- GaussianEllipsoidSource: Gaussian intensity distribution

All sources support sparse sampling for efficient rendering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


SAMPLING_SCHEMES = {
    "1-point": {"n_points": 1, "description": "Single center point"},
    "7-point": {"n_points": 7, "description": "Center + 6 face centers"},
    "19-point": {"n_points": 19, "description": "Center + 6 face + 12 edge centers"},
    "27-point": {"n_points": 27, "description": "3x3x3 grid"},
}


def get_sampling_scheme(level: str) -> dict:
    """Get sampling scheme by level name."""
    if level not in SAMPLING_SCHEMES:
        raise ValueError(
            f"Unknown sampling level: {level}. Available: {list(SAMPLING_SCHEMES.keys())}"
        )
    return SAMPLING_SCHEMES[level]


@dataclass
class BaseSource(ABC):
    """Abstract base class for all source types."""

    center: np.ndarray
    alpha: float

    @abstractmethod
    def sample_points(
        self, sampling_level: str = "7-point"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from source distribution.

        Args:
            sampling_level: "1-point", "7-point", "19-point", or "27-point"

        Returns:
            points_mm: [N, 3] sampled point positions
            weights: [N] weight for each sample point
        """
        pass

    @abstractmethod
    def get_extent_mm(self) -> float:
        """Get approximate spatial extent of source in mm."""
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get all source parameters as dict."""
        pass


class PointSource(BaseSource):
    """Point source - single emission point."""

    def __init__(self, center: np.ndarray, alpha: float = 1.0):
        """
        Args:
            center: [x, y, z] position in mm
            alpha: source intensity
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.alpha = float(alpha)

    def sample_points(
        self, sampling_level: str = "1-point"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Always returns single point."""
        points = self.center.reshape(1, 3)
        weights = np.array([self.alpha])
        return points, weights

    def get_extent_mm(self) -> float:
        return 0.0

    def get_params(self) -> dict:
        return {
            "type": "point",
            "center": self.center.tolist(),
            "alpha": self.alpha,
        }


class UniformEllipsoidSource(BaseSource):
    """Uniform intensity within an ellipsoid."""

    def __init__(
        self,
        center: np.ndarray,
        axes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
    ):
        """
        Args:
            center: [x, y, z] position in mm
            axes: (ax, ay, az) semi-axis lengths in mm
            alpha: total source intensity
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.axes = tuple(axes)
        self.alpha = float(alpha)

    def _generate_sample_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample points based on sampling level.

        For uniform ellipsoid, sample points are placed at scaled positions
        within the ellipsoid. Points are in normalized [-0.5, 0.5] range,
        then scaled by axes.
        """

        if n_points == 1:
            points = np.array([[0, 0, 0]], dtype=np.float32)

        elif n_points == 7:
            offset = 0.5
            points = np.array(
                [
                    [0, 0, 0],
                    [offset, 0, 0],
                    [-offset, 0, 0],
                    [0, offset, 0],
                    [0, -offset, 0],
                    [0, 0, offset],
                    [0, 0, -offset],
                ],
                dtype=np.float32,
            )

        elif n_points == 19:
            offset = 0.5
            edge_off = offset * 0.5
            points = np.array(
                [
                    [0, 0, 0],
                    [offset, 0, 0],
                    [-offset, 0, 0],
                    [0, offset, 0],
                    [0, -offset, 0],
                    [0, 0, offset],
                    [0, 0, -offset],
                    [edge_off, edge_off, 0],
                    [edge_off, -edge_off, 0],
                    [-edge_off, edge_off, 0],
                    [-edge_off, -edge_off, 0],
                    [edge_off, 0, edge_off],
                    [edge_off, 0, -edge_off],
                    [-edge_off, 0, edge_off],
                    [-edge_off, 0, -edge_off],
                    [0, edge_off, edge_off],
                    [0, edge_off, -edge_off],
                    [0, -edge_off, edge_off],
                    [0, -edge_off, -edge_off],
                ],
                dtype=np.float32,
            )

        elif n_points == 27:
            lin = np.linspace(-0.5, 0.5, 3)
            xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
                np.float32
            )

        else:
            raise ValueError(f"Unsupported n_points: {n_points}")

        points[:, 0] *= self.axes[0]
        points[:, 1] *= self.axes[1]
        points[:, 2] *= self.axes[2]

        points = points + self.center

        weights = np.ones(len(points), dtype=np.float32)
        weights = weights / weights.sum() * self.alpha

        return points, weights

    def sample_points(
        self, sampling_level: str = "7-point"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from uniform ellipsoid."""
        scheme = get_sampling_scheme(sampling_level)
        return self._generate_sample_points(scheme["n_points"])

    def get_extent_mm(self) -> float:
        return max(self.axes)

    def get_params(self) -> dict:
        return {
            "type": "uniform",
            "center": self.center.tolist(),
            "axes": self.axes,
            "alpha": self.alpha,
        }


class GaussianEllipsoidSource(BaseSource):
    """Gaussian intensity distribution within an ellipsoid."""

    def __init__(
        self,
        center: np.ndarray,
        sigmas: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
    ):
        """
        Args:
            center: [x, y, z] position in mm
            sigmas: (sx, sy, sz) standard deviations in mm
            alpha: peak source intensity
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.sigmas = tuple(sigmas)
        self.alpha = float(alpha)

    def _compute_weights(self, points: np.ndarray) -> np.ndarray:
        """Compute Gaussian weights for sample points."""
        diff = points - self.center
        diff[:, 0] /= self.sigmas[0]
        diff[:, 1] /= self.sigmas[1]
        diff[:, 2] /= self.sigmas[2]

        dist_sq = np.sum(diff**2, axis=1)
        weights = np.exp(-0.5 * dist_sq)
        weights = weights / weights.sum() * self.alpha

        return weights

    def _generate_sample_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample points based on sampling level."""

        if n_points == 1:
            points = np.array([[0, 0, 0]], dtype=np.float32)

        elif n_points == 7:
            offset = 1.0
            points = np.array(
                [
                    [0, 0, 0],
                    [offset, 0, 0],
                    [-offset, 0, 0],
                    [0, offset, 0],
                    [0, -offset, 0],
                    [0, 0, offset],
                    [0, 0, -offset],
                ],
                dtype=np.float32,
            )

        elif n_points == 19:
            offset = 1.0
            edge_off = offset * 0.7
            points = np.array(
                [
                    [0, 0, 0],
                    [offset, 0, 0],
                    [-offset, 0, 0],
                    [0, offset, 0],
                    [0, -offset, 0],
                    [0, 0, offset],
                    [0, 0, -offset],
                    [edge_off, edge_off, 0],
                    [edge_off, -edge_off, 0],
                    [-edge_off, edge_off, 0],
                    [-edge_off, -edge_off, 0],
                    [edge_off, 0, edge_off],
                    [edge_off, 0, -edge_off],
                    [-edge_off, 0, edge_off],
                    [-edge_off, 0, -edge_off],
                    [0, edge_off, edge_off],
                    [0, edge_off, -edge_off],
                    [0, -edge_off, edge_off],
                    [0, -edge_off, -edge_off],
                ],
                dtype=np.float32,
            )

        elif n_points == 27:
            lin = np.linspace(-1.0, 1.0, 3)
            xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
                np.float32
            )

        else:
            raise ValueError(f"Unsupported n_points: {n_points}")

        points[:, 0] *= self.sigmas[0]
        points[:, 1] *= self.sigmas[1]
        points[:, 2] *= self.sigmas[2]

        points = points + self.center

        weights = self._compute_weights(points)

        return points, weights

    def sample_points(
        self, sampling_level: str = "7-point"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from Gaussian ellipsoid."""
        scheme = get_sampling_scheme(sampling_level)
        return self._generate_sample_points(scheme["n_points"])

    def get_extent_mm(self) -> float:
        return 2.0 * max(self.sigmas)

    def get_params(self) -> dict:
        return {
            "type": "gaussian",
            "center": self.center.tolist(),
            "sigmas": self.sigmas,
            "alpha": self.alpha,
        }


def create_source(source_config: dict) -> BaseSource:
    """Factory function to create source from config dict."""
    source_type = source_config["type"]

    if source_type == "point":
        return PointSource(
            center=np.array(source_config["center"]),
            alpha=source_config.get("alpha", 1.0),
        )
    elif source_type == "uniform":
        return UniformEllipsoidSource(
            center=np.array(source_config["center"]),
            axes=tuple(source_config.get("axes", (1.0, 1.0, 1.0))),
            alpha=source_config.get("alpha", 1.0),
        )
    elif source_type == "gaussian":
        return GaussianEllipsoidSource(
            center=np.array(source_config["center"]),
            sigmas=tuple(source_config.get("sigmas", (1.0, 1.0, 1.0))),
            alpha=source_config.get("alpha", 1.0),
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")
