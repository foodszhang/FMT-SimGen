#!/usr/bin/env python3
"""Generate surface_depth.npz using body_mask (original_labels > 0) from full atlas.
Uses project_volume_reference with vcw=(0,0,0) which gives 20581 finite px at angle=0."""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("__file__").parent.parent))
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="data/uniform_trunk_v2_20260420_100948/samples")
    parser.add_argument("--view_config", default="output/shared/view_config.json")
    args = parser.parse_args()

    view_cfg = json.load(open(args.view_config))
    camera = TurntableCamera(view_cfg)
    angles = view_cfg["angles"]

    # Load full atlas labels (380x992x208 @ 0.1mm)
    af = np.load("output/shared/atlas_full.npz", allow_pickle=True)
    orig = af["original_labels"]
    body_mask = (orig > 0).astype(np.float32)
    print(f"body_mask shape={body_mask.shape}, nnz={np.count_nonzero(body_mask)}")

    # Pre-compute all 7 depth maps (shared across samples)
    print("Computing depth maps from body_mask (this is the same for all samples)...")
    depth_maps = {}
    for angle in angles:
        _, sdepth = project_volume_reference(
            body_mask,
            angle_deg=float(angle),
            camera_distance=camera.camera_distance_mm,
            fov_mm=camera.fov_mm,
            detector_resolution=camera.detector_resolution,
            voxel_size_mm=0.1,   # atlas is 0.1mm
            volume_center_world=(0.0, 0.0, 0.0),  # atlas frame center
        )
        depth_maps[str(angle)] = sdepth.astype(np.float32)
        f = np.isfinite(sdepth)
        print(f"  angle={angle:4d}: finite={f.sum()}, range=[{sdepth[f].min():.1f}, {sdepth[f].max():.1f}]")

    # Save per sample (same depth maps, but save to each sample dir)
    samples_dir = Path(args.samples_dir)
    for sp in sorted(samples_dir.glob("sample_*")):
        out_path = sp / "surface_depth.npz"
        if out_path.exists():
            continue
        np.savez_compressed(out_path, **depth_maps)
        print(f"  {sp.name}: saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
