"""Figure 1: E0 2D Surface Projection Comparison.

Shows MCX vs Analytic Green's function 2D surface images side-by-side
with residual maps. Much more visually convincing than 1D profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color, get_label, format_number


def radial_to_2d(rho, intensity, image_size, pixel_size_mm):
    """Generate 2D rotationally symmetric image from 1D radial profile.

    Args:
        rho: radial coordinates (mm)
        intensity: intensity values at rho
        image_size: output image size (square)
        pixel_size_mm: pixel size in mm

    Returns:
        2D image array
    """
    # Create coordinate grid
    coords = (np.arange(image_size) - image_size / 2) * pixel_size_mm
    xx, yy = np.meshgrid(coords, coords)
    r = np.sqrt(xx**2 + yy**2)

    # Interpolate to grid
    image = np.interp(r.flatten(), rho, intensity, right=0)
    return image.reshape(image_size, image_size)


def load_e0_surface_data(config_id, profile_dir):
    """Load MCX and analytic surface data for a config.

    Returns:
        dict with mcx_img, green_img, rho, pixel_size, etc.
    """
    mcx_file = profile_dir / f"{config_id}_mcx.npz"
    analytic_file = profile_dir / f"{config_id}_analytic.npz"

    if not mcx_file.exists() or not analytic_file.exists():
        return None

    mcx_data = np.load(mcx_file)
    analytic_data = np.load(analytic_file)

    # Get MCX data
    rho_mcx = mcx_data["rho"]
    I_mcx = mcx_data["intensity"]

    # Get analytic data
    rho_analytic = analytic_data["rho"]
    I_semi = analytic_data["I_semi"]

    # Determine image parameters
    # MCX data: assume 151 points from 0-15mm, pixel size ~0.1mm
    # For visualization, create 200x200 image covering ~20mm FOV
    image_size = 200
    fov_mm = 20.0
    pixel_size = fov_mm / image_size

    # Generate 2D images from radial profiles
    mcx_img = radial_to_2d(rho_mcx, I_mcx, image_size, pixel_size)
    green_img = radial_to_2d(rho_analytic, I_semi, image_size, pixel_size)

    # Compute NCC on central region
    center = image_size // 2
    radius = int(10.0 / pixel_size)  # 10mm radius
    y, x = np.ogrid[:image_size, :image_size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2

    mcx_masked = mcx_img[mask]
    green_masked = green_img[mask]

    # Normalize for NCC
    mcx_norm = (mcx_masked - mcx_masked.mean()) / (mcx_masked.std() + 1e-10)
    green_norm = (green_masked - green_masked.mean()) / (green_masked.std() + 1e-10)
    ncc = np.corrcoef(mcx_norm, green_norm)[0, 1]

    return {
        "mcx_img": mcx_img,
        "green_img": green_img,
        "rho_mcx": rho_mcx,
        "I_mcx": I_mcx,
        "rho_green": rho_analytic,
        "I_green": I_semi,
        "pixel_size": pixel_size,
        "image_size": image_size,
        "ncc": ncc,
        "tissue": str(mcx_data.get("tissue_type", "Unknown")),
        "depth": float(mcx_data.get("depth_mm", 0)),
    }


def plot_figure1_2d_comparison(output_dir):
    """Generate Figure 1: 2D surface comparison.

    Layout: 3 rows (depths) x 4 cols (MCX, Green, Residual, Profile)
    """
    set_paper_style()

    configs = [
        ("C01", "Muscle", 1.5),
        ("C02", "Muscle", 3.0),
        ("C03", "Muscle", 5.0),
    ]

    profile_dir = Path(
        "/home/foods/pro/FMT-SimGen/pilot/e0_psf_validation/results/profiles"
    )

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    for row, (cid, tissue, depth) in enumerate(configs):
        data = load_e0_surface_data(cid, profile_dir)

        if data is None:
            for col in range(4):
                axes[row, col].text(
                    0.5,
                    0.5,
                    "Data N/A",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                )
            continue

        mcx_img = data["mcx_img"]
        green_img = data["green_img"]
        ncc = data["ncc"]

        # Shared color scale
        vmax = max(mcx_img.max(), green_img.max())
        vmin = 0

        extent = [-10, 10, -10, 10]  # mm

        # Panel 1: MCX
        ax = axes[row, 0]
        im1 = ax.imshow(
            mcx_img, cmap="inferno", vmin=vmin, vmax=vmax, extent=extent, origin="lower"
        )
        ax.set_title(f"MCX ({tissue}, d={depth}mm)", fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        # Panel 2: Green's function
        ax = axes[row, 1]
        im2 = ax.imshow(
            green_img,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower",
        )
        ax.set_title("Analytic Green", fontsize=10)
        ax.set_xlabel("x (mm)")

        # Panel 3: Residual
        ax = axes[row, 2]
        residual = np.abs(mcx_img - green_img)
        vmax_res = residual.max()
        im3 = ax.imshow(residual, cmap="hot", extent=extent, origin="lower")
        ax.set_title(f"|Residual|, max={vmax_res:.2e}", fontsize=10)
        ax.set_xlabel("x (mm)")

        # Panel 4: Central profile
        ax = axes[row, 3]
        mid = data["image_size"] // 2
        rho = (np.arange(data["image_size"]) - mid) * data["pixel_size"]

        # Normalize for comparison
        mcx_profile = mcx_img[mid, :] / mcx_img.max()
        green_profile = green_img[mid, :] / green_img.max()

        ax.plot(rho, mcx_profile, "-", color=get_color("mcx"), lw=2, label="MCX")
        ax.plot(
            rho,
            green_profile,
            "--",
            color=get_color("green_halfspace"),
            lw=1.5,
            label="Green",
        )
        ax.set_title(f"Profile, NCC={ncc:.4f}", fontsize=10)
        ax.set_xlabel("ρ (mm)")
        ax.set_ylabel("Normalized intensity")
        ax.set_xlim(-10, 10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.02, 0.55, 0.01, 0.35])
    plt.colorbar(im1, cax=cbar_ax1, label="Intensity")

    cbar_ax2 = fig.add_axes([0.02, 0.15, 0.01, 0.35])
    plt.colorbar(im3, cax=cbar_ax2, label="|Residual|")

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    output_path = output_dir / "fig1_e0_2d_mcx_vs_green"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E0 2D comparison figure."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Generating Figure 1: E0 2D comparison...")
    plot_figure1_2d_comparison(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
