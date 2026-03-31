"""
FMT-SimGen: Fluorescence Molecular Tomography Simulation Dataset Generator

A Python library for generating FMT simulation datasets with dual-ground-truth
sampling (FEM nodes + voxel grid) from analytic tumor definitions.
"""

__version__ = "0.1.0"

from fmt_simgen.atlas.digimouse import DigimouseAtlas
from fmt_simgen.mesh.mesh_generator import MeshGenerator
from fmt_simgen.physics.fem_solver import FEMSolver
from fmt_simgen.physics.optical_params import OpticalParameterManager
from fmt_simgen.tumor.tumor_generator import TumorGenerator, TumorSample, AnalyticFocus
from fmt_simgen.sampling.dual_sampler import DualSampler
from fmt_simgen.dataset.builder import DatasetBuilder

__all__ = [
    "DigimouseAtlas",
    "MeshGenerator",
    "FEMSolver",
    "OpticalParameterManager",
    "TumorGenerator",
    "TumorSample",
    "AnalyticFocus",
    "DualSampler",
    "DatasetBuilder",
]
