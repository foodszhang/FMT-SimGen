"""Generate LaTeX and CSV tables for paper.

Table 1: E0 summary
Table 2: E1d summary
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.interpolate import interp1d
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import format_number, verdict_to_symbol


def resample_to_common_rho(rho_source, intensity_source, rho_target):
    """Resample intensity to match target rho grid."""
    if len(rho_source) == len(rho_target) and np.allclose(rho_source, rho_target):
        return intensity_source

    interp = interp1d(
        rho_source, intensity_source, kind="linear", bounds_error=False, fill_value=0
    )
    return interp(rho_target)


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation."""
    a_norm = (a - a.mean()) / (a.std() + 1e-10)
    b_norm = (b - b.mean()) / (b.std() + 1e-10)
    return float(np.corrcoef(a_norm, b_norm)[0, 1])


def load_e0_summary(e0_dir: Path) -> List[Dict]:
    """Load E0 experiment summary data."""
    results = []

    # Config definitions
    configs = [
        ("C01", "Muscle", 1.5),
        ("C02", "Muscle", 3.0),
        ("C03", "Muscle", 5.0),
        ("C04", "Liver", 1.5),
        ("C05", "Liver", 3.0),
        ("C06", "Liver", 5.0),
        ("C07", "Muscle", 1.5),  # Different source size
        ("C08", "Liver", 3.0),
        ("C09", "Bilayer", 5.0),
    ]

    profile_dir = e0_dir / "profiles"

    for cid, tissue, depth in configs:
        row = {
            "config": cid,
            "tissue": tissue,
            "depth_mm": depth,
            "ncc_green": None,
            "ncc_gauss": None,
            "fwhm_ratio": None,
            "peak_ratio": None,
            "verdict": "N/A",
        }

        # Try to load from npz files
        mcx_file = profile_dir / f"{cid}_mcx.npz"
        analytic_file = profile_dir / f"{cid}_analytic.npz"

        if mcx_file.exists() and analytic_file.exists():
            try:
                mcx_data = np.load(mcx_file)
                analytic_data = np.load(analytic_file)

                # Get data
                rho_mcx = mcx_data.get("rho", None)
                I_mcx = mcx_data.get("intensity", None)
                rho_analytic = analytic_data.get("rho", None)
                I_semi = analytic_data.get("I_semi", None)
                I_gauss = analytic_data.get("I_gauss", None)

                if rho_mcx is not None and I_mcx is not None:
                    # Resample analytic data to MCX grid
                    if rho_analytic is not None and I_semi is not None:
                        I_semi_resampled = resample_to_common_rho(
                            rho_analytic, I_semi, rho_mcx
                        )
                        row["ncc_green"] = compute_ncc(I_mcx, I_semi_resampled)

                    if rho_analytic is not None and I_gauss is not None:
                        I_gauss_resampled = resample_to_common_rho(
                            rho_analytic, I_gauss, rho_mcx
                        )
                        row["ncc_gauss"] = compute_ncc(I_mcx, I_gauss_resampled)

                    # Compute FWHM ratio
                    fwhm_mcx = compute_fwhm(rho_mcx, I_mcx)
                    if rho_analytic is not None and I_semi is not None:
                        fwhm_semi = compute_fwhm(rho_analytic, I_semi)
                        if fwhm_mcx > 0:
                            row["fwhm_ratio"] = fwhm_semi / fwhm_mcx

            except Exception as e:
                print(f"Warning: Could not load {cid}: {e}")

        # Determine verdict
        if row["ncc_green"] is not None and row["ncc_green"] > 0:
            if row["ncc_green"] > 0.995:
                row["verdict"] = "GO"
            elif row["ncc_green"] > 0.99:
                row["verdict"] = "CAUTION"
            else:
                row["verdict"] = "NO-GO"

        results.append(row)

    return results


def compute_fwhm(rho: np.ndarray, intensity: np.ndarray) -> float:
    """Compute FWHM from radial profile."""
    half_max = intensity.max() / 2
    above_half = rho[intensity >= half_max]
    if len(above_half) < 2:
        return 0.0
    return float(above_half[-1] - above_half[0])


def generate_table1(e0_data: List[Dict], output_dir: Path):
    """Generate Table 1: E0 summary."""

    # CSV output
    csv_path = output_dir / "table1_e0_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Config",
                "Tissue",
                "Depth (mm)",
                "NCC (Green)",
                "NCC (Gauss)",
                "FWHM Ratio",
                "Verdict",
            ]
        )
        for row in e0_data:
            writer.writerow(
                [
                    row["config"],
                    row["tissue"],
                    row["depth_mm"],
                    format_number(row["ncc_green"], 3),
                    format_number(row["ncc_gauss"], 3),
                    format_number(row["fwhm_ratio"], 3),
                    row["verdict"],
                ]
            )
    print(f"Saved: {csv_path}")

    # LaTeX output
    tex_path = output_dir / "table1_e0_summary.tex"
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{E0 PSF Validation Summary: Analytic Green's Function vs MCX Monte Carlo}
\label{tab:e0_summary}
\begin{tabular}{clcccc}
\toprule
Config & Tissue & Depth (mm) & NCC (Green) & NCC (Gauss) & Verdict \\
\midrule
""")

        for row in e0_data:
            f.write(
                f"{row['config']} & {row['tissue']} & {row['depth_mm']} & "
                f"{format_number(row['ncc_green'], 3)} & "
                f"{format_number(row['ncc_gauss'], 3)} & "
                f"{row['verdict']} \\\\\n"
            )

        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Saved: {tex_path}")


def load_e1d_summary(e1d_dir: Path) -> List[Dict]:
    """Load E1d experiment summary data."""
    results = []

    summary_file = e1d_dir / "e1d_r2_summary.json"
    if not summary_file.exists():
        print(f"Warning: E1d summary not found: {summary_file}")
        return results

    try:
        with open(summary_file, "r") as f:
            data = json.load(f)

        # Part A: Geometry results
        part_a = data.get("part_a_geometry_results", {})

        experiments_a = [
            (
                "A1-shallow",
                "A1_atlas_self_consistent_shallow",
                "Atlas self-consistent (shallow)",
            ),
            (
                "A1-deep",
                "A1_atlas_self_consistent_deep",
                "Atlas self-consistent (deep)",
            ),
            (
                "A2-shallow",
                "A2_atlas_vs_flat_shallow",
                "Atlas GT $\rightarrow$ Flat inverse (shallow)",
            ),
            (
                "A2-deep",
                "A2_atlas_vs_flat_deep",
                "Atlas GT $\rightarrow$ Flat inverse (deep)",
            ),
            ("A3", "A3_lateral_source", "Lateral region source"),
        ]

        for exp_id, key, desc in experiments_a:
            exp_data = part_a.get(key, {})
            pos_err = exp_data.get("position_error_mm", None)
            size_err = exp_data.get("size_error_mm", None)
            ncc = exp_data.get("flat_vs_gt_ncc", None)

            # Determine verdict
            if pos_err is not None:
                if pos_err < 0.5:
                    verdict = "PASS"
                else:
                    verdict = "FAIL"
            else:
                verdict = "N/A"

            results.append(
                {
                    "experiment": exp_id,
                    "setting": desc,
                    "position_error_mm": pos_err,
                    "size_error_mm": size_err,
                    "ncc": ncc,
                    "verdict": verdict,
                    "section": "A",
                }
            )

        # Part B: Quadrature results
        part_b = data.get("part_b_quadrature_results", {})
        gauss_quad = part_b.get("gaussian_source", {})

        for scheme in ["1-point", "sr-6", "ut-7", "grid-27"]:
            if scheme in gauss_quad:
                quad_data = gauss_quad[scheme]
                results.append(
                    {
                        "experiment": f"B1-{scheme}",
                        "setting": f"vs MC-512 reference ({scheme})",
                        "position_error_mm": None,
                        "size_error_mm": None,
                        "ncc": quad_data.get("ncc", None),
                        "verdict": "GOOD" if quad_data.get("ncc", 0) > 0.998 else "OK",
                        "section": "B",
                    }
                )

        # Part C: Inverse degradation
        part_c = data.get("part_c_inverse_degradation", {})

        experiments_c = [
            ("C1", "C1_gaussian_to_gaussian", "Gaussian $\rightarrow$ Gaussian"),
            ("C2", "C2_uniform_to_uniform", "Uniform $\rightarrow$ Uniform"),
            (
                "C3",
                "C3_uniform_to_gaussian",
                "Uniform $\rightarrow$ Gaussian (mismatch)",
            ),
        ]

        for exp_id, key, desc in experiments_c:
            exp_data = part_c.get(key, {})
            pos_err = exp_data.get("position_error_mm", None)
            size_err = exp_data.get("size_error_mm", None)

            if pos_err is not None:
                if pos_err < 0.6:
                    verdict = "PASS"
                elif pos_err < 1.0:
                    verdict = "CAUTION"
                else:
                    verdict = "FAIL"
            else:
                verdict = "N/A"

            results.append(
                {
                    "experiment": exp_id,
                    "setting": desc,
                    "position_error_mm": pos_err,
                    "size_error_mm": size_err,
                    "ncc": None,
                    "verdict": verdict,
                    "section": "C",
                }
            )

    except Exception as e:
        print(f"Warning: Failed to parse E1d summary: {e}")

    return results


def generate_table2(e1d_data: List[Dict], output_dir: Path):
    """Generate Table 2: E1d summary."""

    # CSV output
    csv_path = output_dir / "table2_e1d_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Experiment",
                "Setting",
                "Position Error (mm)",
                "Size Error (mm)",
                "NCC",
                "Verdict",
            ]
        )
        for row in e1d_data:
            writer.writerow(
                [
                    row["experiment"],
                    row["setting"],
                    format_number(row["position_error_mm"], 3),
                    format_number(row["size_error_mm"], 3),
                    format_number(row["ncc"], 3),
                    row["verdict"],
                ]
            )
    print(f"Saved: {csv_path}")

    # LaTeX output
    tex_path = output_dir / "table2_e1d_summary.tex"
    with open(tex_path, "w") as f:
        f.write(r"""\begin{table*}[t]
\centering
\caption{E1d Atlas Surface Experiments Summary: Geometry, Quadrature, and Inverse Degradation}
\label{tab:e1d_summary}
\begin{tabular}{llcccc}
\toprule
Experiment & Setting & Pos. Error (mm) & Size Error (mm) & NCC & Verdict \\
\midrule
""")

        current_section = ""
        for row in e1d_data:
            # Add section header
            if row["section"] != current_section:
                if current_section != "":
                    f.write(r"\midrule" + "\n")
                f.write(
                    r"\multicolumn{6}{l}{\textit{Part " + row["section"] + "}} \\\\\n"
                )
                current_section = row["section"]

            pos_err = format_number(row["position_error_mm"], 3)
            size_err = format_number(row["size_error_mm"], 3)
            ncc = format_number(row["ncc"], 3)

            f.write(
                f"{row['experiment']} & {row['setting']} & {pos_err} & {size_err} & {ncc} & {row['verdict']} \\\\\n"
            )

        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    print(f"Saved: {tex_path}")


def main():
    """Generate all tables."""
    pilot_dir = Path("/home/foods/pro/FMT-SimGen/pilot")
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Table 1: E0 Summary")
    print("=" * 60)
    e0_dir = pilot_dir / "e0_psf_validation" / "results"
    e0_data = load_e0_summary(e0_dir)
    if e0_data:
        generate_table1(e0_data, output_dir)
    else:
        print("Warning: No E0 data available")

    print("\n" + "=" * 60)
    print("Generating Table 2: E1d Summary")
    print("=" * 60)
    e1d_dir = pilot_dir / "e1d_finite_source_local_surface" / "results"
    e1d_data = load_e1d_summary(e1d_dir)
    if e1d_data:
        generate_table2(e1d_data, output_dir)
    else:
        print("Warning: No E1d data available")

    print("\nAll tables generated!")


if __name__ == "__main__":
    main()
