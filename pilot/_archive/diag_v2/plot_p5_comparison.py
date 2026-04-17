#!/usr/bin/env python3
"""Generate P5 corrected comparison plot."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p5_dir = Path(__file__).parent / "results" / "P5_corrected"

    mcx = np.load(p5_dir / "mcx_a60.npy")
    green = np.load(p5_dir / "green_a60.npy")

    mcx_norm = mcx / mcx.max()
    green_norm = green / green.max()

    diff = mcx_norm - green_norm

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    im = ax.imshow(mcx_norm, cmap="hot", origin="lower")
    ax.set_title("MCX (normalized)", fontsize=14)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(green_norm, cmap="hot", origin="lower")
    ax.set_title("Green (normalized)", fontsize=14)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(diff, cmap="RdBu", origin="lower", vmin=-0.5, vmax=0.5)
    ax.set_title("MCX - Green", fontsize=14)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    mask = (mcx > 0) & (green > 0)
    ax.scatter(mcx_norm[mask], green_norm[mask], alpha=0.3, s=1)
    ax.plot([0, 1], [0, 1], "r--", label="1:1")
    ax.set_xlabel("MCX (normalized)")
    ax.set_ylabel("Green (normalized)")
    ax.set_title("Pixel scatter", fontsize=14)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ratio = mcx / (green + 1e-30)
    ratio_valid = ratio[mask]
    ax.hist(ratio_valid, bins=50, edgecolor="black")
    ax.set_xlabel("MCX / Green")
    ax.set_ylabel("Count")
    ax.set_title("Pixel ratio distribution", fontsize=14)
    ax.axvline(
        ratio_valid.mean(),
        color="r",
        linestyle="--",
        label=f"mean={ratio_valid.mean():.2e}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    profiles = []
    for i in [mcx_norm, green_norm]:
        h, w = i.shape
        profile = i[h // 2, :]
        profiles.append(profile)
    ax.plot(profiles[0], label="MCX", alpha=0.7)
    ax.plot(profiles[1], label="Green", alpha=0.7)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Horizontal profile (Y=center)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    mcx_flat = mcx.flatten().astype(np.float64)
    green_flat = green.flatten().astype(np.float64)
    mcx_mean = mcx_flat.mean()
    green_mean = green_flat.mean()
    mcx_centered = mcx_flat - mcx_mean
    green_centered = green_flat - green_mean
    ncc = np.dot(mcx_centered, green_centered) / np.sqrt(
        np.dot(mcx_centered, mcx_centered) * np.dot(green_centered, green_centered)
    )
    k = mcx.sum() / green.sum()

    fig.suptitle(
        f"P5 Corrected (Y=10mm): NCC={ncc:.4f}, k={k:.4e}\n"
        f"Source in soft_tissue (not liver)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = p5_dir / "p5_corrected_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # Also create original P5 comparison for reference
    orig_dir = (
        Path(__file__).parent.parent
        / "results"
        / "stage2_multiposition_v2"
        / "S2-Vol-P5-ventral-r2.0"
    )

    mcx_orig = np.load(orig_dir / "mcx_a60.npy")
    green_orig = np.load(orig_dir / "green_a60.npy")

    mcx_orig_norm = mcx_orig / mcx_orig.max()
    green_orig_norm = green_orig / green_orig.max()

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes2[0, 0]
    im = ax.imshow(mcx_orig_norm, cmap="hot", origin="lower")
    ax.set_title("MCX (normalized)", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[0, 1]
    im = ax.imshow(green_orig_norm, cmap="hot", origin="lower")
    ax.set_title("Green (normalized)", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[0, 2]
    im = ax.imshow(
        mcx_orig_norm - green_orig_norm,
        cmap="RdBu",
        origin="lower",
        vmin=-0.5,
        vmax=0.5,
    )
    ax.set_title("MCX - Green", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[1, 0]
    mask_orig = (mcx_orig > 0) & (green_orig > 0)
    ax.scatter(mcx_orig_norm[mask_orig], green_orig_norm[mask_orig], alpha=0.3, s=1)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("MCX (normalized)")
    ax.set_ylabel("Green (normalized)")
    ax.set_title("Pixel scatter", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    ratio_orig = mcx_orig / (green_orig + 1e-30)
    ax.hist(ratio_orig[mask_orig], bins=50, edgecolor="black")
    ax.set_xlabel("MCX / Green")
    ax.set_ylabel("Count")
    ax.set_title("Pixel ratio distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 2]
    ax.text(
        0.5,
        0.7,
        "ORIGINAL P5 (Y=2.4mm)",
        fontsize=14,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.5,
        "Source was INSIDE LIVER",
        fontsize=12,
        ha="center",
        color="red",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.3, "NCC = 0.23 (FAIL)", fontsize=12, ha="center", transform=ax.transAxes
    )
    ax.text(
        0.5,
        0.1,
        "k = 1.09e4 (1000× smaller)",
        fontsize=12,
        ha="center",
        transform=ax.transAxes,
    )
    ax.axis("off")

    mcx_flat_orig = mcx_orig.flatten().astype(np.float64)
    green_flat_orig = green_orig.flatten().astype(np.float64)
    mcx_mean_orig = mcx_flat_orig.mean()
    green_mean_orig = green_flat_orig.mean()
    mcx_centered_orig = mcx_flat_orig - mcx_mean_orig
    green_centered_orig = green_flat_orig - green_mean_orig
    ncc_orig = np.dot(mcx_centered_orig, green_centered_orig) / np.sqrt(
        np.dot(mcx_centered_orig, mcx_centered_orig)
        * np.dot(green_centered_orig, green_centered_orig)
    )

    fig2.suptitle(
        f"P5 Original (Y=2.4mm): NCC={ncc_orig:.4f}\n"
        f"Source was INSIDE LIVER (geometry error)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path2 = p5_dir / "p5_original_comparison.png"
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path2}")


if __name__ == "__main__":
    main()
