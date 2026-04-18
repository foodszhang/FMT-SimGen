"""Configuration for MVP pipeline."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class OpticalParams:
    mu_a: float = 0.087
    mus_p: float = 4.3
    g: float = 0.9
    n: float = 1.37

    @property
    def D(self) -> float:
        return 1.0 / (3.0 * self.mus_p)

    @property
    def delta(self) -> float:
        return (self.D / self.mu_a) ** 0.5

    @property
    def mu_eff(self) -> float:
        return 1.0 / self.delta

    def to_dict(self) -> dict:
        return {
            "mu_a": self.mu_a,
            "mus_p": self.mus_p,
            "g": self.g,
            "n": self.n,
            "D": self.D,
            "delta": self.delta,
            "mu_eff": self.mu_eff,
        }


@dataclass
class MVPConfig:
    voxel_size_mm: float = 0.2
    detector_resolution: tuple = (256, 256)
    fov_mm: float = 40.0
    camera_distance_mm: float = 80.0
    mcx_photons_gt: int = 10**9
    mcx_photons_fast: int = 10**8
    views: tuple = (0, 90, -90, 180)
    optical: OpticalParams = None

    def __post_init__(self):
        if self.optical is None:
            self.optical = OpticalParams()

    def to_dict(self) -> dict:
        return {
            "voxel_size_mm": self.voxel_size_mm,
            "detector_resolution": self.detector_resolution,
            "fov_mm": self.fov_mm,
            "camera_distance_mm": self.camera_distance_mm,
            "mcx_photons_gt": self.mcx_photons_gt,
            "mcx_photons_fast": self.mcx_photons_fast,
            "views": self.views,
            "optical": self.optical.to_dict() if self.optical else None,
        }


OPTICAL = OpticalParams()
