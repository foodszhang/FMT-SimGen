"""Frame transforms — single source of truth."""
import json
import numpy as np
from pathlib import Path


def load_frame_manifest(shared_dir: str | Path) -> dict:
    """Load frame_manifest.json from shared directory."""
    shared_dir = Path(shared_dir)
    with open(shared_dir / "frame_manifest.json") as f:
        return json.load(f)


def world_mm_to_mcx_voxel(world_mm: np.ndarray, manifest: dict) -> np.ndarray:
    """trunk-local mm → MCX voxel index (float, corner-aligned)."""
    vx = manifest["mcx_volume"]["voxel_size_mm"]
    return np.asarray(world_mm, dtype=np.float64) / vx


def atlas_mm_to_world_mm(atlas_mm: np.ndarray, manifest: dict) -> np.ndarray:
    """atlas-corner mm → trunk-local mm."""
    off = np.asarray(manifest["atlas_to_world_offset_mm"], dtype=np.float64)
    return np.asarray(atlas_mm, dtype=np.float64) - off


def gt_voxels_world_to_index(world_mm: np.ndarray, manifest: dict) -> np.ndarray:
    """Look up gt_voxels grid index from world coord.

    Uses voxel-grid center alignment (consistent with dual_sampler).
    """
    g = manifest["voxel_grid_gt"]
    off = np.asarray(g["offset_world_mm"], dtype=np.float64)
    sp = g["spacing_mm"]
    # center-aligned: index = (world - offset - spacing/2) / spacing
    return (np.asarray(world_mm) - off - sp / 2) / sp
