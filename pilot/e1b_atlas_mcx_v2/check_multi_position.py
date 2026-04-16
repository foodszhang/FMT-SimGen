#!/usr/bin/env python3
"""Check multi-position results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_projection import project_get_surface_coords


def main():
    results_dir = Path(__file__).parent / "results/multi_position"

    # Load atlas for outline
    atlas_bin_path = "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary, 0.0, 200.0, 50.0, (256, 256), 0.2
    )

    positions = ["center", "left", "right", "anterior", "posterior"]

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for row, pos_id in enumerate(positions):
        result_dir = results_dir / f"MP-{pos_id}-D4mm"

        try:
            mcx = np.load(result_dir / "mcx_projection_a0.npy")
            green = np.load(result_dir / "green_projection_a0.npy")

            # Separate normalization
            mcx_norm = mcx / mcx.max()
            green_norm = green / green.max()

            # MCX
            im0 = axes[row, 0].imshow(mcx_norm, cmap="hot", vmin=0, vmax=1)
            axes[row, 0].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
            axes[row, 0].set_title(f"{pos_id}: MCX")
            axes[row, 0].axis("off")

            # Green
            im1 = axes[row, 1].imshow(green_norm, cmap="hot", vmin=0, vmax=1)
            axes[row, 1].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
            axes[row, 1].set_title(f"{pos_id}: Green")
            axes[row, 1].axis("off")

            # Difference
            diff = np.abs(mcx_norm - green_norm)
            im2 = axes[row, 2].imshow(diff, cmap="hot", vmin=0, vmax=1)
            axes[row, 2].contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
            axes[row, 2].set_title(f"{pos_id}: |Diff|")
            axes[row, 2].axis("off")

            print(
                f"{pos_id}: MCX peak at {np.unravel_index(np.argmax(mcx), mcx.shape)}, Green peak at {np.unravel_index(np.argmax(green), green.shape)}"
            )

        except Exception as e:
            print(f"{pos_id}: Error - {e}")
            axes[row, 0].text(0.5, 0.5, f"Error: {pos_id}", ha="center", va="center")

    plt.tight_layout()
    output_path = results_dir / "multi_position_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
