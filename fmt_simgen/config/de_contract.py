"""
DE (Diffusion Equation) channel contract.

Source type (gaussian vs uniform) and sigma are experiment-specific
and come from config/default.yaml at runtime.
This file holds only truly universal DE constants.

Note: sigma values are per-experiment; see config/default.yaml::de
"""
from typing import Final

# Source types
DE_SOURCE_GAUSSIAN: Final[str] = "gaussian"
DE_SOURCE_UNIFORM: Final[str] = "uniform"

# Default Gaussian sigma in mm (used as runtime default, not a locked contract)
DE_DEFAULT_SIGMA_MM: Final[float] = 3.0

# Surface node discretization: FEM mesh node spacing target
DE_FEM_MAX_EDGE_MM: Final[float] = 1.5
