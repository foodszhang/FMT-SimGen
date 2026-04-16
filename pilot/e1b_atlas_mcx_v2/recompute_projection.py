#!/usr/bin/env python3
"""Recompute projection using run_stage1_5_surface_aware.py function."""

import numpy as np
import jdata as jd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_stage1_5_surface_aware import project_mcx_to_detector


def main():
    # Load MCX output
    work_data = jd.loadjd("results/stage1_5_surface/S1.5-D4mm/S1.5-D4mm.jnii")
    work_fluence = np.array(work_data["NIFTIData"], dtype=np.float32)
    if work_fluence.ndim > 3:
        work_fluence = work_fluence[..., 0, 0]
    # JNII is already in XYZ order - NO transpose needed
    # work_fluence = work_fluence.transpose(2, 1, 0)  # <-- WRONG!

    print(f"Loaded fluence: {work_fluence.shape}, max={work_fluence.max():.4e}")

    # Project using the function from run_stage1_5_surface_aware.py
    proj = project_mcx_to_detector(
        work_fluence,
        0.0,  # angle
        200.0,  # camera_distance
        50.0,  # fov
        (256, 256),  # resolution
        0.2,  # voxel_size
    )

    print(
        f"Projected: max={proj.max():.4e}, peak at {np.unravel_index(np.argmax(proj), proj.shape)}"
    )

    # Save for comparison
    Path("results/multiposition").mkdir(parents=True, exist_ok=True)
    np.save("results/multiposition/work_recomputed.npy", proj)
    print("Saved to results/multiposition/work_recomputed.npy")

    # Compare with saved
    saved = np.load("results/stage1_5_surface/S1.5-D4mm/mcx_projection_a0.npy")
    print(
        f"Saved: max={saved.max():.4e}, peak at {np.unravel_index(np.argmax(saved), saved.shape)}"
    )

    diff = np.abs(proj - saved)
    print(f"Difference max: {diff.max():.4e}")
    print(
        f"Difference at peak: {diff[np.unravel_index(np.argmax(saved), saved.shape)]:.4e}"
    )

    # Find where they differ most
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Max difference at: {max_diff_idx}")
    print(f"  Recomputed: {proj[max_diff_idx]:.4e}")
    print(f"  Saved:      {saved[max_diff_idx]:.4e}")


if __name__ == "__main__":
    main()
