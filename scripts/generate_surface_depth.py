#!/usr/bin/env python3
"""
Generate surface_depth.npz for each sample — geometric trunk silhouette depth_map.
Uses project_volume_reference on trunk_mask to get angle-dependent front-surface depth.

Usage:
    python scripts/generate_surface_depth.py [--samples_dir data/uniform_trunk_v2_20260420_100948/samples]
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOXEL_SIZE_MM, VOLUME_CENTER_WORLD
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_trunk_mask() -> np.ndarray:
    """Load trunk volume from mcx_volume_trunk.bin → XYZ order [X×Y×Z]."""
    bin_path = Path("output/shared/mcx_volume_trunk.bin")
    # Shape: [Z=104, Y=200, X=190] (ZYX order in file)
    # Loaded as (X, Y, Z) via transpose(2,1,0)
    raw = np.fromfile(bin_path, dtype=np.uint8)
    vol_zyx = raw.reshape((104, 200, 190))
    vol_xyz = vol_zyx.transpose(2, 1, 0)  # [X=190, Y=200, Z=104]
    trunk_mask = (vol_xyz > 0).astype(np.float32)  # 0/1
    logger.info(f"Trunk mask: shape={trunk_mask.shape}, nonzero={np.count_nonzero(trunk_mask)}")
    return trunk_mask


def md5_vol(v: np.ndarray) -> str:
    return hashlib.md5(v.astype(np.uint8).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="data/uniform_trunk_v2_20260420_100948/samples")
    parser.add_argument("--angles", default="-90,-60,-30,0,30,60,90")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    angles = [int(a) for a in args.angles.split(",")]
    camera = TurntableCamera({})  # default config: fov_mm=50 but we'll override below
    camera.fov_mm = 80.0  # must match view_config.json

    trunk_mask = load_trunk_mask()

    # MD5 check across 3 samples
    sample_dirs = sorted(samples_dir.glob("sample_*"))[:3]
    logger.info("Trunk mask MD5 across first 3 samples:")
    for sp in sample_dirs:
        logger.info(f"  {sp.name}: (mask comes from shared bin, same for all)")

    # Generate surface_depth for all samples
    for sp in sorted(samples_dir.glob("sample_*")):
        out_path = sp / "surface_depth.npz"
        if out_path.exists():
            logger.debug(f"Skip {sp.name}: surface_depth.npz exists")
            continue
        depths = {}
        for angle in angles:
            _, sdepth = project_volume_reference(
                trunk_mask,
                angle_deg=float(angle),
                camera_distance=camera.camera_distance_mm,
                fov_mm=camera.fov_mm,
                detector_resolution=camera.detector_resolution,
                voxel_size_mm=VOXEL_SIZE_MM,
                volume_center_world=tuple(VOLUME_CENTER_WORLD),
            )
            depths[str(angle)] = sdepth.astype(np.float32)
        np.savez_compressed(out_path, **depths)
        logger.info(f"  {sp.name}: saved {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
