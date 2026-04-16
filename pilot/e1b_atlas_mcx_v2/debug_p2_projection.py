#!/usr/bin/env python3
"""Debug P2-left projection from different angles."""

import numpy as np
import jdata as jd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from run_multiposition_bestview import project_mcx_to_detector


def main():
    # Load P2-left MCX
    data = jd.loadjd("results/multiposition/P2-left/P2-left.jnii")
    fluence = np.array(data["NIFTIData"], dtype=np.float32)[..., 0, 0]

    print(f"Loaded P2-left: {fluence.shape}, max={fluence.max():.4e}")
    print(
        f"Source expected at: X=-8.3, Y=2.4, Z=1.0 (left side, 4mm from left surface)"
    )

    angles = [-90, -60, -30, 0, 30, 60, 90]
    projections = {}

    for angle in angles:
        proj = project_mcx_to_detector(
            fluence, float(angle), 200.0, 50.0, (256, 256), 0.2
        )
        projections[angle] = proj
        peak = np.unravel_index(np.argmax(proj), proj.shape)
        print(
            f"Angle {angle:3d}°: max={proj.max():.4e}, peak at {peak}, nonzero={np.sum(proj > 0)}"
        )

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, angle in enumerate(angles):
        proj = projections[angle]
        im = axes[i].imshow(proj, cmap="hot")
        axes[i].set_title(f"{angle}°: max={proj.max():.4e}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    axes[-1].axis("off")
    plt.suptitle(
        "P2-left (source on left side) - Projections from Different Angles", fontsize=14
    )
    plt.tight_layout()

    output_path = Path("results/multiposition/P2_left_all_angles.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
