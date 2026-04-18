#!/usr/bin/env python3
"""Hard gates: 6-point frame consistency check."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify frame consistency")
    parser.add_argument(
        "--dataset", required=True,
        help="e.g. data/uniform_1000_v3")
    parser.add_argument(
        "--shared", default="assets/mesh",
        help="Shared assets directory (contains mesh.npz and frame_manifest.json)")
    parser.add_argument(
        "--max_samples", type=int, default=10,
        help="Maximum samples to check")
    args = parser.parse_args()

    shared = Path(args.shared)
    manifest = json.load(open(shared / "frame_manifest.json"))
    trunk_offset = np.array(manifest["atlas_to_world_offset_mm"])
    mcx_bbox_max = np.array(manifest["mcx_volume"]["bbox_world_mm"]["max"])
    mcx_bbox_min = np.array(manifest["mcx_volume"]["bbox_world_mm"]["min"])
    mesh_frame = manifest["fem_mesh"]["frame"]

    # Gate 1: mesh.npz frame matches manifest
    print("Checking mesh frame...")
    mesh = np.load(shared / "mesh.npz")
    nodes_disk = mesh["nodes"]

    # Convert to trunk-local for checks
    if mesh_frame == "atlas_corner_mm":
        nodes = nodes_disk.astype(np.float64) - trunk_offset
    else:
        nodes = nodes_disk.astype(np.float64)

    mesh_bbox_min = manifest["fem_mesh"]["bbox_world_mm"]["min"]
    mesh_bbox_max = manifest["fem_mesh"]["bbox_world_mm"]["max"]
    assert nodes_disk.min() >= -1.0, \
        f"Gate1 FAIL: mesh nodes out of reasonable range"
    assert (np.array(mesh_bbox_max) - nodes_disk.max(axis=0) < 5.0).all(), \
        f"Gate1 FAIL: mesh bbox mismatch"
    print(f"  Gate1 PASS  (mesh.npz is {mesh_frame}, trunk-offset={trunk_offset})")

    # Gate 2: ≥50% of trunk-local nodes inside MCX bbox (full-body mesh has head/tail outside)
    in_mcx = np.all(
        (nodes >= mcx_bbox_min - 1.0) & (nodes <= mcx_bbox_max + 1.0), axis=1
    ).mean()
    assert in_mcx >= 0.50, f"Gate2 FAIL: only {in_mcx*100:.1f}% nodes in MCX bbox"
    print(f"  Gate2 PASS  ({in_mcx*100:.1f}% trunk-local nodes inside MCX bbox)")

    # Gate 3-4: per-sample checks
    samples_dir = Path(args.dataset) / "samples"
    sids = sorted(
        p.name for p in samples_dir.iterdir() if p.is_dir()
    )[: args.max_samples]

    if not sids:
        print("  No samples found, skipping Gate 3-4")
    else:
        print(f"Checking {len(sids)} samples...")
        g = manifest["voxel_grid_gt"]
        gt_off = np.array(g["offset_world_mm"])
        gt_sp = g["spacing_mm"]

        for sid in sids:
            tp = json.load(open(samples_dir / sid / "tumor_params.json"))
            for f in tp["foci"]:
                c = np.asarray(f["center"])
                # Gate 3: focus center must be inside MCX bbox
                assert np.all((c >= mcx_bbox_min - 1.0) & (c <= mcx_bbox_max + 1.0)), \
                    f"Gate3 FAIL {sid}: focus center {c} outside MCX bbox"

            # Gate 4: gt_voxels centroid matches foci center
            gt_v = np.load(samples_dir / sid / "gt_voxels.npy")
            if gt_v.sum() > 1e-6:
                idx = np.argwhere(gt_v > 0.05 * gt_v.max())
                ctr_world = idx.mean(0) * gt_sp + gt_off + gt_sp / 2
                foci_ctr = np.mean(
                    [np.asarray(f["center"]) for f in tp["foci"]], axis=0
                )
                err = np.linalg.norm(ctr_world - foci_ctr)
                assert err < 3.0, \
                    f"Gate4 FAIL {sid}: gt_voxels center off by {err:.2f}mm"

        print(f"  Gate3-4 PASS  (checked {len(sids)} samples)")

    # Gate 5: MCX jnii bbox contains foci (requires MCX output)
    # Gate 6: proj.npz non-zero ratio (requires projections)
    print("\nNote: Gate 5-6 require MCX output — skipped in this check.")
    print("\nALL GATES PASS (Gate 1-4)")


if __name__ == "__main__":
    main()
