#!/usr/bin/env python3
"""Plot side-by-side projection comparison: Stage 1 (Cube) vs Stage 1.5 (Atlas)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmt_simgen.mcx_projection import project_volume_reference


def load_and_project(sample_dir, angle=0):
    """Load comparison.npz and project at given angle."""
    data = np.load(sample_dir / "comparison.npz")
    fluence_mcx = data["fluence_mcx"]
    fluence_green = data["fluence_green"]

    proj_mcx, _ = project_volume_reference(
        fluence_mcx, angle, 200, 50, (256, 256), 0.2
    )
    proj_green, _ = project_volume_reference(
        fluence_green, angle, 200, 50, (256, 256), 0.2
    )

    return proj_mcx, proj_green


def main():
    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    depths = [2, 4, 6]  # Show 3 representative depths
    angle = 0  # Frontal view

    fig, axes = plt.subplots(len(depths), 4, figsize=(16, 4.5 * len(depths)))

    for row_idx, depth in enumerate(depths):
        # Stage 1 (Cube)
        s1_dir = results_dir / "stage1" / f"S1-D{depth:.0f}mm"
        proj_mcx_s1, proj_green_s1 = load_and_project(s1_dir, angle)

        # Stage 1.5 (Atlas)
        s1a_dir = results_dir / "stage1_atlas" / f"S1A-D{depth:.0f}mm"
        proj_mcx_s1a, proj_green_s1a = load_and_project(s1a_dir, angle)

        # Normalize each to its own peak
        p1_mcx = proj_mcx_s1 / proj_mcx_s1.max()
        p1_green = proj_green_s1 / proj_green_s1.max()
        p1a_mcx = proj_mcx_s1a / proj_mcx_s1a.max()
        p1a_green = proj_green_s1a / proj_green_s1a.max()

        # Plot
        vmax = 1.0

        # Column 1: Stage 1 MCX
        im0 = axes[row_idx, 0].imshow(p1_mcx.T, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[row_idx, 0].set_title(f'Stage 1 (Cube) - MCX\nDepth {depth}mm', fontsize=10)
        axes[row_idx, 0].axis('off')

        # Column 2: Stage 1 Green
        im1 = axes[row_idx, 1].imshow(p1_green.T, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[row_idx, 1].set_title(f'Stage 1 (Cube) - Green\nNCC ≈ 0.99', fontsize=10)
        axes[row_idx, 1].axis('off')

        # Column 3: Stage 1.5 MCX
        im2 = axes[row_idx, 2].imshow(p1a_mcx.T, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        axes[row_idx, 2].set_title(f'Stage 1.5 (Atlas) - MCX\nDepth {depth}mm', fontsize=10)
        axes[row_idx, 2].axis('off')

        # Column 4: Stage 1.5 Green
        im3 = axes[row_idx, 3].imshow(p1a_green.T, origin='lower', cmap='hot', vmin=0, vmax=vmax)
        if depth == 2:
            axes[row_idx, 3].set_title(f'Stage 1.5 (Atlas) - Green\nNCC ≈ 0.80', fontsize=10)
        elif depth == 4:
            axes[row_idx, 3].set_title(f'Stage 1.5 (Atlas) - Green\nNCC ≈ 0.57', fontsize=10)
        else:
            axes[row_idx, 3].set_title(f'Stage 1.5 (Atlas) - Green\nNCC ≈ 0.31', fontsize=10)
        axes[row_idx, 3].axis('off')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)

    plt.suptitle('Projection Comparison: Stage 1 (Cube) vs Stage 1.5 (Atlas)\n'
                 'Angle = 0° (Frontal View)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.savefig(figures_dir / "projection_comparison_cube_vs_atlas.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {figures_dir / 'projection_comparison_cube_vs_atlas.png'}")


if __name__ == "__main__":
    main()
