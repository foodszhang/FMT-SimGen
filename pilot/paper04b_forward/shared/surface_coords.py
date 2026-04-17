"""Surface position definitions for E-C experiment."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class SurfacePosition:
    """A source position on the atlas surface.

    Attributes
    ----------
    name : str
        Position identifier (e.g., "P1-dorsal").
    xyz_mm : np.ndarray
        (X, Y, Z) coordinates in mm.
    best_angle : int
        Optimal viewing angle in degrees.
    description : str
        Human-readable description.
    """

    name: str
    xyz_mm: np.ndarray
    best_angle: int
    description: str


def get_surface_positions(y_mm: float = 10.0) -> Dict[str, SurfacePosition]:
    """Get all 5 surface positions for E-C experiment.

    Parameters
    ----------
    y_mm : float
        Y coordinate in mm. Default 10.0 (corrected).
        Y=2.4 is out-of-scope (sources in liver).

    Returns
    -------
    dict mapping position name to SurfacePosition.
    """
    positions = {
        "P1": SurfacePosition(
            name="P1-dorsal",
            xyz_mm=np.array([0.0, y_mm, 8.0], dtype=np.float32),
            best_angle=0,
            description="Dorsal midline (back)",
        ),
        "P2": SurfacePosition(
            name="P2-left",
            xyz_mm=np.array([-8.0, y_mm, 0.0], dtype=np.float32),
            best_angle=60,
            description="Left lateral",
        ),
        "P3": SurfacePosition(
            name="P3-right",
            xyz_mm=np.array([8.0, y_mm, 0.0], dtype=np.float32),
            best_angle=-60,
            description="Right lateral",
        ),
        "P4": SurfacePosition(
            name="P4-dorsal-lat",
            xyz_mm=np.array([-6.0, y_mm, 6.0], dtype=np.float32),
            best_angle=30,
            description="Dorsal-left oblique",
        ),
        "P5": SurfacePosition(
            name="P5-ventral",
            xyz_mm=np.array([0.0, y_mm, -6.0], dtype=np.float32),
            best_angle=0,
            description="Ventral (belly)",
        ),
    }
    return positions


def get_position_names() -> List[str]:
    """Get list of position names in order."""
    return ["P1", "P2", "P3", "P4", "P5"]
