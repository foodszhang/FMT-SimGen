"""Three-source unified interface for MVP pipeline.

Provides a single SourceSpec class that handles:
- Point source (isotropic)
- Uniform ball source
- 3D Gaussian splat

All sources are volume-normalized to total power = alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

if TYPE_CHECKING:
    from .config import OpticalParams


@dataclass
class SourceSpec:
    kind: Literal["point", "ball", "gaussian"]
    center_mm: np.ndarray
    alpha: float = 1.0
    radius_mm: float | None = None
    sigma_mm: float | np.ndarray | None = None

    def __post_init__(self):
        self.center_mm = np.asarray(self.center_mm, dtype=np.float64).reshape(3)
        if self.kind == "ball" and self.radius_mm is None:
            raise ValueError("ball source requires radius_mm")
        if self.kind == "gaussian" and self.sigma_mm is None:
            raise ValueError("gaussian source requires sigma_mm")
        if isinstance(self.sigma_mm, (int, float)):
            self.sigma_mm = np.full(3, float(self.sigma_mm), dtype=np.float64)
        if self.sigma_mm is not None:
            self.sigma_mm = np.asarray(self.sigma_mm, dtype=np.float64)

    def pattern3d(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size_mm: float,
        bbox_sigma: float = 4.0,
    ) -> np.ndarray:
        if self.kind == "point":
            return self._pattern_point(volume_shape, voxel_size_mm)
        elif self.kind == "ball":
            return self._pattern_ball(volume_shape, voxel_size_mm)
        elif self.kind == "gaussian":
            return self._pattern_gaussian(volume_shape, voxel_size_mm, bbox_sigma)
        else:
            raise ValueError(f"Unknown source kind: {self.kind}")

    def _pattern_point(
        self, volume_shape: Tuple[int, int, int], voxel_size_mm: float
    ) -> np.ndarray:
        nx, ny, nz = volume_shape
        pattern = np.zeros((nx, ny, nz), dtype=np.float32)
        # Point source is placed at pattern center
        # center_mm is relative to pattern center, but for point source
        # we place it at the center of the pattern
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        pattern[cx, cy, cz] = 1.0
        return pattern

    def _pattern_ball(
        self, volume_shape: Tuple[int, int, int], voxel_size_mm: float
    ) -> np.ndarray:
        nx, ny, nz = volume_shape
        # Pattern coordinates centered at (0,0,0)
        x = (np.arange(nx) - (nx - 1) / 2) * voxel_size_mm
        y = (np.arange(ny) - (ny - 1) / 2) * voxel_size_mm
        z = (np.arange(nz) - (nz - 1) / 2) * voxel_size_mm
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        # Source center is relative to pattern center (0,0,0)
        # For pattern3d, we ignore center_mm and place ball at origin
        r = np.sqrt(X**2 + Y**2 + Z**2)
        pattern = (r <= self.radius_mm).astype(np.float32)
        total = np.sum(pattern)
        if total > 0:
            pattern *= self.alpha / total
        return pattern

    def _pattern_gaussian(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size_mm: float,
        bbox_sigma: float,
    ) -> np.ndarray:
        nx, ny, nz = volume_shape
        x = (np.arange(nx) - (nx - 1) / 2) * voxel_size_mm
        y = (np.arange(ny) - (ny - 1) / 2) * voxel_size_mm
        z = (np.arange(nz) - (nz - 1) / 2) * voxel_size_mm
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        sigma = self.sigma_mm
        # Gaussian centered at pattern origin
        r2 = (X / sigma[0]) ** 2 + (Y / sigma[1]) ** 2 + (Z / sigma[2]) ** 2
        pattern = np.exp(-0.5 * r2)
        cutoff = bbox_sigma**2
        pattern[r2 > cutoff] = 0.0
        total = np.sum(pattern)
        if total > 0:
            pattern *= self.alpha / total
        return pattern.astype(np.float32)

    def to_dict(self) -> dict:
        result = {
            "kind": self.kind,
            "center_mm": self.center_mm.tolist(),
            "alpha": self.alpha,
        }
        if self.radius_mm is not None:
            result["radius_mm"] = self.radius_mm
        if self.sigma_mm is not None:
            result["sigma_mm"] = self.sigma_mm.tolist()
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "SourceSpec":
        return cls(
            kind=d["kind"],
            center_mm=np.array(d["center_mm"]),
            alpha=d.get("alpha", 1.0),
            radius_mm=d.get("radius_mm"),
            sigma_mm=d.get("sigma_mm"),
        )

    def __repr__(self) -> str:
        if self.kind == "point":
            return f"SourceSpec(point, center={self.center_mm}, alpha={self.alpha})"
        elif self.kind == "ball":
            return (
                f"SourceSpec(ball, center={self.center_mm}, "
                f"R={self.radius_mm}mm, alpha={self.alpha})"
            )
        else:
            return (
                f"SourceSpec(gaussian, center={self.center_mm}, "
                f"sigma={self.sigma_mm}mm, alpha={self.alpha})"
            )
