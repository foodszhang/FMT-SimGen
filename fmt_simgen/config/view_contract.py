"""
View / camera contract: turntable imaging geometry.

These are locked by the hardware specification and should not vary per experiment.
"""
from typing import Final

# Turntable angles in degrees
VIEW_ANGLES: Final[list[int]] = [-90, -60, -30, 0, 30, 60, 90]

# Field of view in mm (square detector)
VIEW_FOV_MM: Final[float] = 80.0

# Camera to rotation-center distance in mm
VIEW_CAMERA_DISTANCE_MM: Final[float] = 200.0

# Detector resolution (width, height) in pixels
VIEW_DETECTOR_RESOLUTION: Final[tuple[int, int]] = (256, 256)

# Pose: "prone" or "supine"
VIEW_POSE: Final[str] = "prone"

# Platform occlusion enabled (always True for real setup)
VIEW_PLATFORM_OCCLUSION: Final[bool] = True

# Platform Z center in trunk-local mm (dorsal surface reference)
VIEW_PLATFORM_Z_CENTER_MM: Final[float] = 4.0
