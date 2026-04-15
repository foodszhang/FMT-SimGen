"""Figure 1 & 2: E0 PSF validation - Analytic Green's function vs MCX.

Figure 1: 2x3 grid comparing MCX, Green's function, and Gaussian PSF
Figure 2: Residual analysis and metrics vs depth
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import set_paper_style, get_color, get_label, format_number


def resample_to_common_rho(rho_source, intensity_source, rho_target):
    """Resample intensity to match target rho grid."""
    if len(rho_source) == len(rho_target) and np.allclose(rho_source, rho_target):
        return intensity_source

    # Use linear interpolation
    interp = interp1d(
        rho_source,
        intensity_source,
        kind="linear",
        bounds_error=False,
        fill_value=0,
    )
    return interp(rho_target)


def load_e0_data(results_dir: Path) -> Dict:
    """Load E0 PSF comparison data from npz files."""
    data = {}
    profile_dir = results_dir / "profiles"

    if not profile_dir.exists():
        print(f"Warning: Profile directory not found: {profile_dir}")
        return data

    configs = [
        ("C01", "Muscle", 1.5),
        ("C02", "Muscle", 3.0),
        ("C03", "Muscle", 5.0),
        ("C04", "Liver", 1.5),
        ("C05", "Liver", 3.0),
        ("C06", "Liver", 5.0),
    ]

    for cid, tissue, depth in configs:
        mcx_file = profile_dir / f"{cid}_mcx.npz"
        analytic_file = profile_dir / f"{cid}_analytic.npz"

        if not mcx_file.exists() or not analytic_file.exists():
            print(f"Warning: Missing data for {cid}")
            continue

        try:
            mcx_data = np.load(mcx_file)
            analytic_data = np.load(analytic_file)

            # Get MCX data (reference grid)
            rho_mcx = mcx_data.get("rho", mcx_data.get("r", None))
            I_mcx = mcx_data.get("intensity", mcx_data.get("I", None))

            # Get analytic data
            rho_analytic = analytic_data.get("rho", None)
            I_semi = analytic_data.get("I_semi", None)
            I_inf = analytic_data.get("I_inf", None)
            I_gauss = analytic_data.get("I_gauss", None)

            if rho_mcx is None or I_mcx is None:
                print(f"Warning: Missing MCX data for {cid}")
                continue

            # Resample analytic data to match MCX rho grid
            if rho_analytic is not None and I_semi is not None:
                I_semi_resampled = resample_to_common_rho(rho_analytic, I_semi, rho_mcx)
            else:
                I_semi_resampled = None

            if rho_analytic is not None and I_inf is not None:
                I_inf_resampled = resample_to_common_rho(rho_analytic, I_inf, rho_mcx)
            else:
                I_inf_resampled = None

            if rho_analytic is not None and I_gauss is not None:
                I_gauss_resampled = resample_to_common_rho(
                    rho_analytic, I_gauss, rho_mcx
                )
            else:
                I_gauss_resampled = None

            data[cid] = {
                "tissue": tissue,
                "depth": depth,
                "rho": rho_mcx,
                "I_mcx": I_mcx,
                "I_semi": I_semi_resampled,
                "I_inf": I_inf_resampled,
                "I_gauss": I_gauss_resampled,
            }

            # Load metrics if available
            for key in [
                "ncc_mcx_semi",
                "ncc_mcx_inf",
                "ncc_mcx_gauss",
                "fwhm_mcx",
                "fwhm_semi",
                "fwhm_gauss",
            ]:
                if key in analytic_data:
                    data[cid][key] = float(analytic_data[key])

        except Exception as e:
            print(f"Warning: Failed to load {cid}: {e}")
            continue

    return data


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation.

    Handles arrays of different sizes by using the shorter length.
    """
    # Ensure same length for comparison
    min_len = min(len(a), len(b))
    a_trim = a[:min_len]
    b_trim = b[:min_len]

    a_norm = (a_trim - a_trim.mean()) / (a_trim.std() + 1e-10)
    b_norm = (b_trim - b_trim.mean()) / (b_trim.std() + 1e-10)
    return float(np.corrcoef(a_norm, b_norm)[0, 1])


def compute_fwhm(rho: np.ndarray, intensity: np.ndarray) -> float:
    """Compute FWHM from radial profile."""
    half_max = intensity.max() / 2
    above_half = rho[intensity >= half_max]
    if len(above_half) < 2:
        return 0.0
    return float(above_half[-1] - above_half[0])


def plot_figure1(data: Dict, output_dir: Path):
    """Generate Figure 1: PSF comparison 2x3 grid."""
    set_paper_style()

    configs = [
        ("C01", "Muscle", 1.5),
        ("C02", "Muscle", 3.0),
        ("C03", "Muscle", 5.0),
        ("C04", "Liver", 1.5),
        ("C05", "Liver", 3.0),
        ("C06", "Liver", 5.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharey="row")
    fig.patch.set_facecolor("white")

    for idx, (cid, tissue, depth) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]

        if cid not in data:
            ax.text(
                0.5, 0.5, "Data N/A", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{tissue}, d={depth}mm")
            continue

        d = data[cid]
        rho = d["rho"]
        I_mcx = d["I_mcx"]
        I_semi = d["I_semi"]
        I_gauss = d.get("I_gauss")

        if rho is None or I_mcx is None or I_semi is None:
            ax.text(
                0.5,
                0.5,
                "Data incomplete",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{tissue}, d={depth}mm")
            continue

        # Normalize to peak
        I_mcx_norm = I_mcx / I_mcx.max()
        I_semi_norm = I_semi / I_semi.max()

        # Plot curves
        ax.plot(rho, I_mcx_norm, "-", color=get_color("mcx"), lw=2, label="MCX (GT)")
        ax.plot(
            rho,
            I_semi_norm,
            "--",
            color=get_color("green_halfspace"),
            lw=1.5,
            label="Green (half-space)",
        )

        if I_gauss is not None:
            I_gauss_norm = I_gauss / I_gauss.max()
            ax.plot(
                rho,
                I_gauss_norm,
                ":",
                color=get_color("gaussian"),
                lw=1.5,
                label="Gaussian fit",
            )

        # Compute and annotate NCC
        ncc = d.get("ncc_mcx_semi")
        if ncc is None and I_mcx is not None and I_semi is not None:
            ncc = compute_ncc(I_mcx, I_semi)

        if ncc is not None:
            ax.text(
                0.97,
                0.95,
                f"NCC={format_number(ncc, 3)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Styling
        ax.set_xlim(0, min(15, rho.max()))
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel(get_label("rho"))
        if idx % 3 == 0:
            ax.set_ylabel(get_label("intensity"))
        ax.set_title(f"{tissue}, d={depth}mm", fontsize=10)

        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "fig1_e0_psf_vs_mcx"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def plot_figure2(data: Dict, output_dir: Path):
    """Generate Figure 2: Residual analysis and metrics vs depth."""
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("white")

    # Panel A: Relative error profiles
    ax = axes[0]
    muscle_configs = ["C01", "C02", "C03"]
    depths_muscle = [1.5, 3.0, 5.0]

    for cid, depth in zip(muscle_configs, depths_muscle):
        if cid not in data:
            continue
        d = data[cid]
        rho = d["rho"]
        I_mcx = d["I_mcx"]
        I_semi = d["I_semi"]

        if rho is None or I_mcx is None or I_semi is None:
            continue

        # Compute relative error
        rel_error = (I_semi - I_mcx) / (I_mcx + 1e-10)

        ax.plot(rho, rel_error, "-", lw=1.5, label=f"d={depth}mm")

    ax.axhline(y=0, color="k", linestyle="--", lw=0.8)
    ax.set_xlabel(get_label("rho"))
    ax.set_ylabel("Relative error (Green-MCX)/MCX")
    ax.set_title("Panel A: Relative error (Muscle)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Summary metrics vs depth
    ax = axes[1]

    tissues = ["Muscle", "Liver"]
    configs_by_tissue = {
        "Muscle": ["C01", "C02", "C03"],
        "Liver": ["C04", "C05", "C06"],
    }
    depths = [1.5, 3.0, 5.0]
    markers = ["o", "s"]

    for tissue, marker in zip(tissues, markers):
        nccs = []
        fwhm_ratios = []
        valid_depths = []

        for cid, depth in zip(configs_by_tissue[tissue], depths):
            if cid not in data:
                continue
            d = data[cid]

            ncc = d.get("ncc_mcx_semi")
            if ncc is None:
                I_mcx = d["I_mcx"]
                I_semi = d["I_semi"]
                if I_mcx is not None and I_semi is not None:
                    ncc = compute_ncc(I_mcx, I_semi)

            fwhm_mcx = d.get("fwhm_mcx")
            fwhm_semi = d.get("fwhm_semi")
            if fwhm_mcx is None and d["I_mcx"] is not None:
                fwhm_mcx = compute_fwhm(d["rho"], d["I_mcx"])
            if fwhm_semi is None and d["I_semi"] is not None:
                fwhm_semi = compute_fwhm(d["rho"], d["I_semi"])

            if ncc is not None:
                nccs.append(ncc)
                valid_depths.append(depth)
            if fwhm_mcx is not None and fwhm_semi is not None and fwhm_mcx > 0:
                fwhm_ratios.append(fwhm_semi / fwhm_mcx)

        if valid_depths:
            ax.plot(
                valid_depths,
                nccs,
                marker=marker,
                linestyle="-",
                color=get_color("green_halfspace"),
                lw=1.5,
                label=f"{tissue} (NCC)",
            )

    ax.set_xlabel(get_label("depth"))
    ax.set_ylabel("NCC")
    ax.set_title("Panel B: NCC vs depth")
    ax.set_ylim(0.95, 1.005)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "fig2_e0_residual_vs_depth"
    plt.savefig(
        output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Saved: {output_path}.pdf/.png")
    plt.close()


def main():
    """Generate E0 figures."""
    base_dir = Path("/home/foods/pro/FMT-SimGen/pilot/e0_psf_validation")
    results_dir = base_dir / "results"
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading E0 data...")
    data = load_e0_data(results_dir)

    if not data:
        print("Warning: No E0 data found. Skipping Figure 1 & 2.")
        return

    print(f"Loaded {len(data)} configurations")

    print("Generating Figure 1: PSF comparison...")
    plot_figure1(data, output_dir)

    print("Generating Figure 2: Residual analysis...")
    plot_figure2(data, output_dir)

    print("E0 figures complete!")


if __name__ == "__main__":
    main()
