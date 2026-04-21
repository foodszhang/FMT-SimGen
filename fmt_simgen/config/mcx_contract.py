"""
MCX contract: pure simulation constants.

These values are fixed for the FMT-SimGen forward model.
Do not put runtime-configured values here (e.g. trunk_offset_mm varies per experiment).
"""

# Wavelength in nm (FMT typical)
MCX_WAVELENGTH_NM: int = 700

# Number of photons per simulation
MCX_PHOTONS: int = 10_000_000

# Time gate: start, end, step (seconds)
MCX_TIME_GATE: tuple[float, float, float] = (0.0, 5.0e-8, 5.0e-8)

# GPU executable name (mcx for CUDA, mcxcl for OpenCL/CPU fallback)
MCX_GPUExecutable: str = "mcx"

# Media list material file (relative to output/shared/)
MCX_MATERIAL_FILE: str = "output/shared/mcx_material.yaml"
