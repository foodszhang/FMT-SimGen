"""
FMT-SimGen configuration contracts.

Each submodule is pure constants with no side effects.
All frame/MCX/DE/view geometry values come from here — no literals elsewhere.

version: v2-2026-04-21
"""
from .frame_contract import *
from .mcx_contract import *
from .de_contract import *
from .view_contract import *

__all__ = [
    # frame (from .frame_contract)
    "TRUNK_OFFSET_ATLAS_MM",
    "TRUNK_SIZE_MM",
    "VOXEL_SIZE_MM",
    "TRUNK_GRID_SHAPE",
    "VOLUME_CENTER_WORLD",
    "CAMERA_DISTANCE_MM",
    "FOV_MM",
    "DETECTOR_RESOLUTION",
    "ANGLES",
    "atlas_corner_to_trunk",
    "trunk_to_atlas_corner",
    "world_to_volume_voxel",
    "volume_voxel_to_world",
    "assert_in_trunk_bbox",
    # MCX
    "MCX_WAVELENGTH_NM",
    "MCX_PHOTONS",
    "MCX_TIME_GATE",
    "MCX_GPUExecutable",
    "MCX_MATERIAL_FILE",
    # DE
    "DE_SOURCE_GAUSSIAN",
    "DE_SOURCE_UNIFORM",
    "DE_DEFAULT_SIGMA_MM",
    "DE_FEM_MAX_EDGE_MM",
    # view
    "VIEW_ANGLES",
    "VIEW_FOV_MM",
    "VIEW_CAMERA_DISTANCE_MM",
    "VIEW_DETECTOR_RESOLUTION",
    "VIEW_POSE",
    "VIEW_PLATFORM_OCCLUSION",
    "VIEW_PLATFORM_Z_CENTER_MM",
]

# ── CONFIG_HASH ──────────────────────────────────────────────────────────────────
import hashlib
import json

_ALL_CONSTANTS: dict = {
    # frame
    "TRUNK_OFFSET_ATLAS_MM": TRUNK_OFFSET_ATLAS_MM.tolist(),
    "TRUNK_SIZE_MM": TRUNK_SIZE_MM.tolist(),
    "VOXEL_SIZE_MM": VOXEL_SIZE_MM,
    "TRUNK_GRID_SHAPE": list(TRUNK_GRID_SHAPE),
    "VOLUME_CENTER_WORLD": VOLUME_CENTER_WORLD.tolist(),
    "CAMERA_DISTANCE_MM": CAMERA_DISTANCE_MM,
    "FOV_MM": FOV_MM,
    "DETECTOR_RESOLUTION": list(DETECTOR_RESOLUTION),
    "ANGLES": ANGLES,
    # MCX
    "MCX_WAVELENGTH_NM": MCX_WAVELENGTH_NM,
    "MCX_PHOTONS": MCX_PHOTONS,
    "MCX_TIME_GATE": list(MCX_TIME_GATE),  # tuple → list for JSON
    # DE
    "DE_DEFAULT_SIGMA_MM": DE_DEFAULT_SIGMA_MM,
    "DE_FEM_MAX_EDGE_MM": DE_FEM_MAX_EDGE_MM,
    # view
    "VIEW_ANGLES": VIEW_ANGLES,
    "VIEW_FOV_MM": VIEW_FOV_MM,
    "VIEW_CAMERA_DISTANCE_MM": VIEW_CAMERA_DISTANCE_MM,
    "VIEW_DETECTOR_RESOLUTION": list(VIEW_DETECTOR_RESOLUTION),
    "VIEW_POSE": VIEW_POSE,
    "VIEW_PLATFORM_OCCLUSION": VIEW_PLATFORM_OCCLUSION,
    "VIEW_PLATFORM_Z_CENTER_MM": VIEW_PLATFORM_Z_CENTER_MM,
}

CONFIG_HASH: str = hashlib.sha256(
    json.dumps(_ALL_CONSTANTS, sort_keys=True).encode()
).hexdigest()[:16]

CONTRACT_VERSION: str = "v2-2026-04-21"

if __name__ == "__main__":
    print(f"CONFIG_HASH={CONFIG_HASH}")
    print(f"CONTRACT_VERSION={CONTRACT_VERSION}")
