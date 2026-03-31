#!/usr/bin/env python3
"""
Step 0a: Load Digimouse Atlas and Analyze Tissue Labels

This script:
1. Loads the full Digimouse atlas (380×992×208)
2. Computes statistics for each unique label
3. Saves merged atlas and tumor region mask
4. Generates visualization images

Usage:
    python scripts/run_step0a_atlas.py

Output:
    output/shared/atlas_full.npz
    output/shared/atlas_vis/*.png
"""

import sys
from pathlib import Path
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from fmt_simgen.atlas.digimouse import DigimouseAtlas

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ATLAS_PATH = "/home/foods/pro/mcx_simulation/ct_data/atlas_380x992x208.hdr"
OUTPUT_DIR = Path("output/shared")
VIS_DIR = OUTPUT_DIR / "atlas_vis"


TISSUE_COLORS = {
    0: "black",
    1: "saddlebrown",  # skin
    2: "ivory",  # skeleton/bone
    3: "lightcoral",  # brain
    4: "indianred",  # medulla
    5: "firebrick",  # cerebellum
    6: "darkred",  # olfactory_bulb
    7: "salmon",  # external_brain
    8: "coral",  # striatum
    9: "orangered",  # heart
    10: "tomato",  # brain_other
    11: "peachpuff",  # muscle
    12: "khaki",  # fat
    13: "lavender",  # cartilage
    14: "plum",  # tongue
    15: "greenyellow",  # stomach
    16: "limegreen",  # spleen
    17: "mediumseagreen",  # pancreas
    18: "darkgreen",  # liver
    19: "teal",  # kidney
    20: "turquoise",  # adrenal
    21: "cyan",  # lung
}


def create_labeled_colorbar(ax, labels, names, colormap):
    """Create a colorbar with labels."""
    patches = [
        Patch(color=colormap(l), label=f"{l}: {names.get(l, 'unknown')}")
        for l in labels
        if l in colormap
    ]
    ax.legend(handles=patches, loc="center", fontsize=6, framealpha=0.9)


def visualize_slice(volume, slice_idx, axis, title, output_path, colormap=None):
    """Visualize a single slice with colormap."""
    if axis == "z":
        data = volume[:, :, slice_idx].T
    elif axis == "y":
        data = volume[:, slice_idx, :].T
    elif axis == "x":
        data = volume[slice_idx, :, :].T

    fig, ax = plt.subplots(figsize=(12, 10))

    if colormap is None:
        ax.imshow(data, cmap="nipy_spectral", origin="lower")
    else:
        cmap = mcolors.ListedColormap(
            [colormap.get(i, "black") for i in range(int(volume.max()) + 1)]
        )
        ax.imshow(data, cmap=cmap, vmin=0, vmax=len(colormap), origin="lower")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def visualize_with_labels(volume, slice_idx, axis, title, output_path, label_names):
    """Visualize slice with tissue labels colored."""
    if axis == "z":
        data = volume[:, :, slice_idx].T
    elif axis == "y":
        data = volume[:, slice_idx, :].T
    elif axis == "x":
        data = volume[slice_idx, :, :].T

    fig, ax = plt.subplots(figsize=(14, 12))

    unique_vals = np.unique(data)
    max_val = int(volume.max()) + 1 if volume.max() < 100 else 30

    cmap = plt.cm.nipy_spectral
    bounds = np.arange(-0.5, max_val + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(data, cmap=cmap, norm=norm, origin="lower")

    cbar = plt.colorbar(im, ax=ax, ticks=range(max_val), shrink=0.8)
    cbar.ax.set_ylabel("Tissue Label", fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def visualize_tumor_region(volume, tumor_mask, slice_idx, axis, title, output_path):
    """Visualize tumor region overlay on atlas."""
    if axis == "z":
        vol_slc = volume[:, :, slice_idx].T
        mask_slc = tumor_mask[:, :, slice_idx].T
    elif axis == "y":
        vol_slc = volume[:, slice_idx, :].T
        mask_slc = tumor_mask[:, slice_idx, :]
    elif axis == "x":
        vol_slc = volume[slice_idx, :, :].T
        mask_slc = tumor_mask[slice_idx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap = plt.cm.nipy_spectral
    bounds = np.arange(-0.5, volume.max() + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    axes[0].imshow(vol_slc, cmap=cmap, norm=norm, origin="lower")
    axes[0].set_title("Original Atlas", fontsize=12)
    axes[0].set_xlabel("Pixel")
    axes[0].set_ylabel("Pixel")

    axes[1].imshow(vol_slc, cmap="gray", origin="lower", alpha=0.7)
    axes[1].imshow(
        np.ma.masked_where(~mask_slc, mask_slc), cmap="Reds", alpha=0.7, origin="lower"
    )
    axes[1].set_title("Tumor Region (red overlay)", fontsize=12)

    axes[2].imshow(
        np.ma.masked_where(~mask_slc, mask_slc), cmap="hot", alpha=0.8, origin="lower"
    )
    axes[2].set_title("Tumor Region Only", fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 0a: Load and analyze Digimouse atlas"
    )
    parser.add_argument("--atlas_path", default=ATLAS_PATH, help="Path to atlas file")
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR), help="Output directory"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 0a: Digimouse Atlas Loading and Analysis")
    logger.info("=" * 60)

    logger.info(f"\nLoading atlas from: {args.atlas_path}")
    atlas = DigimouseAtlas(args.atlas_path)
    atlas.load()

    print("\n" + "=" * 60)
    print("LABEL STATISTICS (from atlas loading)")
    print("=" * 60)
    for stats in atlas.info.label_stats:
        print(
            f"  Label {stats.label:2d} ({stats.name:15s}): "
            f"{stats.voxel_count:8d} voxels, "
            f"centroid=({stats.centroid[0]:7.1f}, {stats.centroid[1]:7.1f}, {stats.centroid[2]:7.1f})mm"
        )

    logger.info("\nComputing subcutaneous region...")
    logger.info("Coordinate system: X=left-right, Y=ant-post, Z=inf-sup (dorsal=+Z)")
    logger.info("NOTE: Subcutaneous muscle/fat tissue is on VENTRAL side (z < center)")
    tumor_region = atlas.get_subcutaneous_region(
        depth_range_mm=(1.0, 3.0),
        regions=["ventral"],
        exclude_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21],
        torso_only=True,
    )

    x_dim, y_dim, z_dim = atlas.shape
    voxel_size = atlas.voxel_size

    logger.info("\nGenerating visualizations...")

    mid_z = z_dim // 2
    mid_y = y_dim // 2
    mid_x = x_dim // 2

    visualize_with_labels(
        atlas.volume,
        mid_z,
        "z",
        f"Digimouse Atlas - Axial Slice at Z={mid_z} (position {mid_z / z_dim:.1%})",
        VIS_DIR / "atlas_axial.png",
        atlas.DEFAULT_TAG_MAPPING,
    )

    visualize_with_labels(
        atlas.volume,
        mid_y,
        "y",
        f"Digimouse Atlas - Coronal Slice at Y={mid_y} (position {mid_y / y_dim:.1%})",
        VIS_DIR / "atlas_coronal.png",
        atlas.DEFAULT_TAG_MAPPING,
    )

    visualize_with_labels(
        atlas.volume,
        mid_x,
        "x",
        f"Digimouse Atlas - Sagittal Slice at X={mid_x} (position {mid_x / x_dim:.1%})",
        VIS_DIR / "atlas_sagittal.png",
        atlas.DEFAULT_TAG_MAPPING,
    )

    visualize_tumor_region(
        atlas.volume,
        tumor_region,
        mid_z,
        "z",
        f"Tumor Region (1-3mm subcutaneous) - Axial at Z={mid_z}",
        VIS_DIR / "atlas_tumor_region_axial.png",
    )

    visualize_tumor_region(
        atlas.volume,
        tumor_region,
        mid_y,
        "y",
        f"Tumor Region (1-3mm subcutaneous) - Coronal at Y={mid_y}",
        VIS_DIR / "atlas_tumor_region_coronal.png",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(tumor_region[:, :, mid_z].T, cmap="hot", origin="lower")
    ax.set_title(f"Tumor Region Mask - Axial Z={mid_z} (white=available)", fontsize=12)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "tumor_region_binary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {VIS_DIR / 'tumor_region_binary.png'}")

    logger.info("\nSaving atlas data...")
    atlas.save(
        str(OUTPUT_DIR / "atlas_full.npz"),
        merged_volume=None,
        tumor_region_mask=tumor_region,
    )

    logger.info("\n" + "=" * 60)
    logger.info("ATLAS ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  Data: {OUTPUT_DIR / 'atlas_full.npz'}")
    logger.info(f"  Visualizations: {VIS_DIR / '*.png'}")
    logger.info(f"\nAtlas info:")
    logger.info(f"  Shape: {atlas.shape}")
    logger.info(f"  Voxel size: {atlas.voxel_size} mm")
    logger.info(
        f"  Physical size: {atlas.shape[0] * atlas.voxel_size:.1f} x "
        f"{atlas.shape[1] * atlas.voxel_size:.1f} x "
        f"{atlas.shape[2] * atlas.voxel_size:.1f} mm"
    )
    logger.info(f"  Unique labels: {len(atlas.info.unique_labels)}")
    logger.info(
        f"  Tumor region voxels: {np.sum(tumor_region)} "
        f"({np.sum(tumor_region) * atlas.voxel_size**3:.2f} mm³)"
    )


if __name__ == "__main__":
    main()
