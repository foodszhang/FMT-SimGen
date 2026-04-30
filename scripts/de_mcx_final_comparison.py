#!/usr/bin/env python3
"""
Compare DE gt_voxels with MCX fluence using the SAME projection function.
Both DE and MCX volumes are projected using project_volume_reference
with identical camera geometry.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
import jdata as jd
from fmt_simgen.mcx_projection import project_volume_reference
from fmt_simgen.view_config import TurntableCamera
from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD

import matplotlib.pyplot as plt


def load_de_volume(sample_dir: Path) -> np.ndarray:
    """Load DE gt_voxels [X=190, Y=200, Z=104] at 0.2mm."""
    return np.load(sample_dir / "gt_voxels.npy")


def load_mcx_volume(sample_dir: Path) -> np.ndarray:
    """Load MCX fluence volume and transpose to [X=190, Y=200, Z=104].

    JNII NIFTIData shape is (X=104, Y=200, Z=190) at 0.2mm.
    DE shape is (X=190, Y=200, Z=104) at 0.2mm.
    Physical: DE (38mm, 40mm, 20.8mm), MCX (20.8mm, 40mm, 38mm) -- X and Z swapped.
    Transpose(2,1,0): result[x,y,z] = nifti[z,y,x] -> (Z=190,Y=200,X=104) = (190,200,104).
    """
    data = jd.load(str(list(sample_dir.glob("*.jnii"))[0]))
    nifti = data["NIFTIData"][:, :, :, 0, 0]  # (X=104, Y=200, Z=190)
    # Transpose to match DE axis ordering: (Z=190, Y=200, X=104) = (190, 200, 104)
    return np.transpose(nifti, (2, 1, 0))


def project_comparison(
    de_vol: np.ndarray,
    mcx_vol: np.ndarray,
    angle: float,
    cam: TurntableCamera,
) -> tuple:
    """Project both volumes using project_volume_reference."""
    mcx_norm = mcx_vol / max(mcx_vol.max(), 1e-6)

    proj_de, _ = project_volume_reference(
        de_vol, angle, cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
    )
    proj_mcx, _ = project_volume_reference(
        mcx_norm, angle, cam.camera_distance_mm, cam.fov_mm,
        cam.detector_resolution, 0.2, VOLUME_CENTER_WORLD
    )

    return proj_de, proj_mcx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="sample_0000")
    parser.add_argument("--samples_dir", type=str, default="data/small_uniform_5samples/samples")
    parser.add_argument("--output_dir", type=str, default="output/verification")
    args = parser.parse_args()

    sample_dir = Path(args.samples_dir) / args.sample
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.sample}...")

    de_vol = load_de_volume(sample_dir)
    print(f"DE volume: shape={de_vol.shape}, max={de_vol.max():.4f}")

    mcx_vol = load_mcx_volume(sample_dir)
    print(f"MCX volume: shape={mcx_vol.shape}, max={mcx_vol.max():.2f}")

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cam = TurntableCamera(cfg["view_config"])

    import json
    tp = json.load(open(sample_dir / "tumor_params.json"))
    n_foci = tp.get("num_foci", 0)
    depth_tier = tp.get("depth_tier", "unknown")

    angles = [-90, -60, -30, 0, 30, 60, 90]
    n = len(angles)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))

    for col, angle in enumerate(angles):
        proj_de, proj_mcx = project_comparison(de_vol, mcx_vol, float(angle), cam)

        ax = axes[0, col]
        vmax = proj_de.max() * 0.8 if proj_de.max() > 0 else 0.01
        ax.imshow(proj_de, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"DE {angle}°")
        ax.axis("off")

        ax = axes[1, col]
        vmax = proj_mcx.max() * 0.8 if proj_mcx.max() > 0 else 0.01
        ax.imshow(proj_mcx, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"MCX {angle}°")
        ax.axis("off")

        ax = axes[2, col]
        diff = proj_de - proj_mcx
        v = max(abs(diff.min()), abs(diff.max()), 0.05)
        ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"Diff {angle}°")
        ax.axis("off")

    for row, label in enumerate(["DE voxels", "MCX fluence", "Difference"]):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=8)

    fig.suptitle(f"{args.sample}: {n_foci} foci, depth={depth_tier}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"{args.sample}_de_mcx_final.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
