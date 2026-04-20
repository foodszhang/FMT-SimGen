#!/usr/bin/env python3
"""
U3: Build canonical trunk volume from atlas.

Produces output/shared/trunk_volume.npz:
    trunk_volume: [X=190, Y=200, Z=104] uint8 labels
    voxel_size_mm: 0.2
    trunk_offset_atlas_mm: [0, 34, 0]

Then regenerates output/shared/mcx_volume_trunk.bin from the same array.

Downsample: majority-vote (not mean) — labels are discrete classes.
Atlas is 0.1mm, trunk is 0.2mm → 2×2×2 block mode.
"""
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
ATLAS_PATH = REPO_ROOT / "output/shared/atlas_full.npz"
TRUNK_VOL_PATH = REPO_ROOT / "output/shared/trunk_volume.npz"
MCX_BIN_PATH = REPO_ROOT / "output/shared/mcx_volume_trunk.bin"

# ── Constants ─────────────────────────────────────────────────────────────────
VS_ATLAS = 0.1   # mm, atlas voxel size
VS_TRUNK = 0.2   # mm, trunk/MCX voxel size (2× downsample)

from fmt_simgen.frame_contract import (
    TRUNK_OFFSET_ATLAS_MM,
    TRUNK_SIZE_MM,
    TRUNK_GRID_SHAPE,
    VOXEL_SIZE_MM,
)
from fmt_simgen.mcx_volume import ORIGINAL_TO_CC_LABEL

assert VS_TRUNK == VOXEL_SIZE_MM  # sanity check

# ── Helpers ───────────────────────────────────────────────────────────────────

def majority_vote_downsample(vol: np.ndarray, block_size: int = 2) -> np.ndarray:
    """Downsample by block-wise majority vote.

    vol: input array of shape (X, Y, Z), all dims must be divisible by block_size.
    block_size: integer, default 2

    Returns downsampled array of shape (X//bs, Y//bs, Z//bs).
    """
    from scipy.stats import mode
    shape = vol.shape
    bs = block_size

    # Reshape: (X//bs, bs, Y//bs, bs, Z//bs, bs)
    new_shape = (shape[0] // bs, bs, shape[1] // bs, bs, shape[2] // bs, bs)
    vol_blocks = vol.reshape(new_shape)

    # Transpose to (X//bs, Y//bs, Z//bs, bs, bs, bs) then flatten block → (X//bs, Y//bs, Z//bs, bs^3)
    vol_blocks = vol_blocks.transpose(0, 2, 4, 1, 3, 5)  # → (Xb, Yb, Zb, bs, bs, bs)
    blocks_flat = vol_blocks.reshape(vol_blocks.shape[:3] + (bs**3,))

    # Mode over the last axis (the 8 block values)
    mode_result, _ = mode(blocks_flat, axis=-1, keepdims=False)
    return mode_result.astype(np.uint8)


def crop_and_downsample(atlas_labels: np.ndarray) -> np.ndarray:
    """Crop atlas to trunk bbox, then majority-vote downsample to 0.2mm."""
    offset = TRUNK_OFFSET_ATLAS_MM  # [0, 34, 0] in atlas mm

    # atlas voxels: shape (380, 992, 208) @ 0.1mm
    # Crop window in atlas voxel indices
    x0 = int(np.round(offset[0] / VS_ATLAS))
    y0 = int(np.round(offset[1] / VS_ATLAS))
    z0 = int(np.round(offset[2] / VS_ATLAS))

    nx_out = int(np.round(TRUNK_SIZE_MM[0] / VS_TRUNK))   # 190
    ny_out = int(np.round(TRUNK_SIZE_MM[1] / VS_TRUNK))   # 200
    nz_out = int(np.round(TRUNK_SIZE_MM[2] / VS_TRUNK))   # 104

    # Atlas crop size in voxels at atlas resolution
    nx_at = int(np.round(TRUNK_SIZE_MM[0] / VS_ATLAS))   # 380
    ny_at = int(np.round(TRUNK_SIZE_MM[1] / VS_ATLAS))   # 400
    nz_at = int(np.round(TRUNK_SIZE_MM[2] / VS_ATLAS))   # 208

    atlas_crop = atlas_labels[
        x0 : x0 + nx_at,
        y0 : y0 + ny_at,
        z0 : z0 + nz_at,
    ]
    logger.info(f"Atlas crop shape: {atlas_crop.shape}, dtype={atlas_crop.dtype}")

    # Majority-vote downsample 2×2×2
    trunk = majority_vote_downsample(atlas_crop, block_size=2)
    logger.info(f"Trunk volume shape: {trunk.shape}, dtype={trunk.dtype}")
    assert trunk.shape == TRUNK_GRID_SHAPE, (
        f"Expected {TRUNK_GRID_SHAPE}, got {trunk.shape}"
    )
    return trunk


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== U3: Build Canonical Trunk Volume ===")

    # Load atlas
    af = np.load(ATLAS_PATH, allow_pickle=True)
    atlas_labels = af["original_labels"]
    logger.info(f"Atlas: shape={atlas_labels.shape}, dtype={atlas_labels.dtype}")

    # Crop + downsample
    trunk = crop_and_downsample(atlas_labels)

    # Apply label mapping: Digimouse original labels → CC Task Packet classes (0-9)
    trunk_cc = np.zeros_like(trunk)
    for orig_label, cc_label in ORIGINAL_TO_CC_LABEL.items():
        trunk_cc[trunk == orig_label] = cc_label
    trunk = trunk_cc

    # Save trunk_volume.npz
    np.savez_compressed(
        TRUNK_VOL_PATH,
        trunk_volume=trunk,
        voxel_size_mm=VS_TRUNK,
        trunk_offset_atlas_mm=TRUNK_OFFSET_ATLAS_MM,
        trunk_size_mm=TRUNK_SIZE_MM,
        trunk_grid_shape=TRUNK_GRID_SHAPE,
    )
    logger.info(f"Saved {TRUNK_VOL_PATH}: shape={trunk.shape}")

    # ── Regenerate mcx_volume_trunk.bin from same array ─────────────────────
    # mcx_volume_trunk.bin stores (Z, Y, X) = shape (104, 200, 190)
    trunk_zyx = trunk.transpose(2, 1, 0)  # XYZ → ZYX
    assert trunk_zyx.shape == (104, 200, 190), f"ZYX shape mismatch: {trunk_zyx.shape}"
    trunk_zyx.tofile(MCX_BIN_PATH)
    logger.info(f"Saved {MCX_BIN_PATH}: shape={trunk_zyx.shape} (ZYX)")

    # ── Per-label stats ─────────────────────────────────────────────────────
    logger.info("\n=== Per-label voxel counts in trunk_volume ===")
    for lb in sorted(np.unique(trunk)):
        count = (trunk == lb).sum()
        total = (atlas_labels == lb).sum()
        frac = count / total if total > 0 else 0
        atlas_frac_in_window = frac  # same metric as U2
        logger.info(f"  label {lb:2d}: {count:7d} voxels, {atlas_frac_in_window*100:.1f}% of atlas ({total:7d})")

    # ── Compare with old mcx_volume_trunk.bin ─────────────────────────────────
    old_bin = Path("output/shared/mcx_volume_trunk.bin")
    if old_bin.exists():
        old = np.fromfile(old_bin, dtype=np.uint8).reshape((104, 200, 190))
        new_zyx = trunk_zyx
        mismatch = (old != new_zyx).sum()
        total = old.size
        logger.info(f"\n=== Old vs New mcx_volume_trunk.bin ===")
        logger.info(f"Old shape (ZYX): {old.shape}")
        logger.info(f"New shape (ZYX): {new_zyx.shape}")
        logger.info(f"Voxel-wise mismatch: {mismatch}/{total} = {mismatch/total*100:.1f}%")
        # Show Y-slice distribution of mismatches
        y_mismatch = (old != new_zyx).sum(axis=(0, 2))  # per Y slice
        for y in [0, 1, 2, 3, 4, 50, 100, 150, 198, 199]:
            if y < y_mismatch.size:
                logger.info(f"  Y slice {y:3d}: {y_mismatch[y]:6d} mismatches")
    else:
        logger.info("No old bin found — skipping comparison")

    logger.info("\n=== U3 complete ===")
    logger.info(f"trunk_volume.npz: {TRUNK_VOL_PATH}")
    logger.info(f"mcx_volume_trunk.bin: {MCX_BIN_PATH}")


if __name__ == "__main__":
    main()
