#!/usr/bin/env python3
"""Check Stage 1.5 projections."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import render_green_surface_projection


def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load atlas binary mask
    atlas_bin_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    )
    volume = np.fromfile(atlas_bin_path, dtype=np.uint8).reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    atlas_binary = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    # Load results
    result_dir = Path(__file__).parent / "results/stage1_5_surface/S1.5-D2mm"
    mcx_proj = np.load(result_dir / "mcx_projection_a0.npy")
    green_proj = np.load(result_dir / "green_projection_a0.npy")

    print(
        f"MCX projection: shape={mcx_proj.shape}, max={mcx_proj.max():.6e}, min={mcx_proj.min():.6e}"
    )
    print(
        f"Green projection: shape={green_proj.shape}, max={green_proj.max():.6e}, min={green_proj.min():.6e}"
    )

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # MCX
    im0 = axes[0, 0].imshow(mcx_proj, cmap="hot")
    axes[0, 0].set_title(f"MCX (linear)\nmax={mcx_proj.max():.4e}")
    plt.colorbar(im0, ax=axes[0, 0])

    mcx_log = np.log10(mcx_proj + 1e-10)
    im1 = axes[0, 1].imshow(mcx_log, cmap="hot", vmin=-5, vmax=np.log10(mcx_proj.max()))
    axes[0, 1].set_title("MCX (log10)")
    plt.colorbar(im1, ax=axes[0, 1])

    # MCX profile
    center_y = mcx_proj.shape[0] // 2
    axes[0, 2].plot(mcx_proj[center_y, :], label="MCX")
    axes[0, 2].set_title(f"MCX Profile (row {center_y})")
    axes[0, 2].set_xlabel("X pixel")
    axes[0, 2].set_ylabel("Intensity")

    # Green
    im3 = axes[1, 0].imshow(green_proj, cmap="hot")
    axes[1, 0].set_title(f"Green (linear)\nmax={green_proj.max():.4e}")
    plt.colorbar(im3, ax=axes[1, 0])

    green_log = np.log10(green_proj + 1e-20)
    im4 = axes[1, 1].imshow(green_log, cmap="hot", vmin=-20, vmax=-15)
    axes[1, 1].set_title("Green (log10)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Green profile
    axes[1, 2].plot(green_proj[center_y, :], label="Green")
    axes[1, 2].set_title(f"Green Profile (row {center_y})")
    axes[1, 2].set_xlabel("X pixel")
    axes[1, 2].set_ylabel("Intensity")

    plt.tight_layout()
    output_path = Path(__file__).parent / "_check_stage1_5_projection.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")

    # Find peak locations
    mcx_peak_idx = np.unravel_index(np.argmax(mcx_proj), mcx_proj.shape)
    green_peak_idx = np.unravel_index(np.argmax(green_proj), green_proj.shape)
    print(f"\nMCX peak location: {mcx_peak_idx}")
    print(f"Green peak location: {green_peak_idx}")


if __name__ == "__main__":
    main()
