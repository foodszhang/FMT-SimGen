"""Configuration for E-C atlas surface experiment."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..shared.surface_coords import (
    SurfacePosition,
    get_surface_positions,
    get_position_names,
)


@dataclass
class ECConfig:
    """Configuration for E-C experiment.

    Attributes
    ----------
    y_mm : float
        Y coordinate for all positions. Default 10.0 (corrected).
        Y=2.4 is out-of-scope.
    n_photons : int
        Number of MCX photons.
    voxel_size_mm : float
        Voxel size in mm (downsampled 2x).
    volume_shape : tuple
        Atlas volume shape (NX, NY, NZ).
    camera_distance_mm : float
        Camera distance.
    fov_mm : float
        Field of view.
    detector_resolution : tuple
        (width, height) in pixels.
    tissue_params : dict
        Soft tissue optical parameters.
    """

    y_mm: float = 10.0
    n_photons: int = 1_000_000_000

    voxel_size_mm: float = 0.4
    volume_shape: tuple = (95, 100, 52)

    camera_distance_mm: float = 80.0
    fov_mm: float = 50.0
    detector_resolution: tuple = (128, 128)

    tissue_params: dict = field(
        default_factory=lambda: {
            "mua_mm": 0.087,
            "mus_prime_mm": 1.0,
            "g": 0.9,
            "n": 1.37,
        }
    )

    out_of_scope_y: float = 2.4

    @property
    def is_out_of_scope(self) -> bool:
        """Check if current Y is out-of-scope (Y=2.4)."""
        return abs(self.y_mm - self.out_of_scope_y) < 0.1

    def get_positions(self) -> Dict[str, SurfacePosition]:
        """Get all surface positions with current Y."""
        return get_surface_positions(self.y_mm)

    def get_mcx_volume_path(self) -> Path:
        """Path to MCX trunk volume."""
        return Path("output/shared/mcx_volume_trunk.bin")

    def get_mcx_material_path(self) -> Path:
        """Path to MCX material YAML."""
        return Path("output/shared/mcx_material.yaml")


def get_default_config() -> ECConfig:
    """Get default configuration (Y=10)."""
    return ECConfig()


def get_out_of_scope_config() -> ECConfig:
    """Get out-of-scope configuration (Y=2.4)."""
    return ECConfig(y_mm=2.4)
