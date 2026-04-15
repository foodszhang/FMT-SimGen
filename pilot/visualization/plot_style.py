"""Unified plotting style for FMT-SimGen paper figures.

Provides consistent styling for TMI/IEEE paper format.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any


def set_paper_style():
    """Set TMI/IEEE paper style with serif fonts."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


# Color palette - colorblind-friendly
COLORS: Dict[str, str] = {
    # Ground truth / reference
    "mcx": "#1a1a1a",  # Black - ground truth
    "reference": "#1a1a1a",
    "gt": "#1a1a1a",
    # Green's function variants
    "green_halfspace": "#2166ac",  # Dark blue - main method
    "green_half": "#2166ac",
    "green_semi": "#2166ac",
    "green_infinite": "#67a9cf",  # Light blue - ablation
    "green_inf": "#67a9cf",
    # Baseline methods
    "gaussian_psf": "#ef8a62",  # Orange-red - baseline
    "gaussian": "#ef8a62",
    "gauss": "#ef8a62",
    # Atlas vs Flat
    "atlas": "#2166ac",  # Blue - atlas (correct)
    "flat": "#ef8a62",  # Red/orange - flat (wrong)
    # Quadrature schemes
    "sr6": "#2166ac",  # Dark blue - recommended
    "ut7": "#67a9cf",  # Light blue
    "grid27": "#b2182b",  # Dark red
    "stratified_33": "#d6604d",  # Light red
    "mc512": "#1a1a1a",  # Black - reference
    "one_point": "#999999",  # Gray
    # Status colors
    "pass": "#1a9850",  # Green
    "fail": "#d73027",  # Red
    "caution": "#fc8d59",  # Orange
    "info": "#4575b4",  # Blue
}

# Line styles
LINESTYLES: Dict[str, str] = {
    "mcx": "-",
    "reference": "-",
    "gt": "-",
    "green_halfspace": "--",
    "green_half": "--",
    "green_semi": "--",
    "green_infinite": "-.",
    "green_inf": "-.",
    "gaussian_psf": ":",
    "gaussian": ":",
    "gauss": ":",
    "atlas": "-",
    "flat": "--",
}

# Marker styles
MARKERS: Dict[str, str] = {
    "muscle": "o",
    "liver": "s",
    "bilayer": "^",
    "atlas": "o",
    "flat": "s",
    "sr6": "*",
    "ut7": "D",
    "grid27": "v",
    "one_point": "x",
}


def get_color(name: str, default: str = "#333333") -> str:
    """Get color by name, with fallback."""
    return COLORS.get(name.lower(), default)


def get_linestyle(name: str, default: str = "-") -> str:
    """Get linestyle by name, with fallback."""
    return LINESTYLES.get(name.lower(), default)


def get_marker(name: str, default: str = "o") -> str:
    """Get marker by name, with fallback."""
    return MARKERS.get(name.lower(), default)


# Figure size presets
FIGURE_SIZES = {
    "single_column": (3.5, 2.5),  # Single column width
    "single_column_tall": (3.5, 4.0),
    "double_column": (7.2, 4.5),  # Double column width
    "double_column_large": (7.2, 6.0),
    "triple_panel": (10.0, 3.5),  # 3 panels in a row
    "six_panel": (10.0, 7.0),  # 2x3 panels
}


def get_figsize(preset: str = "single_column") -> tuple:
    """Get figure size by preset name."""
    return FIGURE_SIZES.get(preset, (6.0, 4.0))


# Label presets
LABELS = {
    "rho": r"Radial distance $\rho$ (mm)",
    "intensity": "Normalized intensity",
    "depth": "Depth (mm)",
    "ncc": "NCC",
    "fwhm_ratio": "FWHM ratio",
    "position_error": "Position error (mm)",
    "size_error": "Size error (%)",
    "n_points": "Number of points",
    "render_time": "Render time (ms)",
    "loss": "Loss",
    "iteration": "Iteration",
}


def get_label(key: str, default: str = None) -> str:
    """Get axis label by key."""
    return LABELS.get(key, default or key)


# LaTeX table style
LATEX_TABLE_PREAMBLE = r"""\begin{table}[t]
\centering
\caption{%s}
\label{%s}
\begin{tabular}{%s}
\toprule
"""

LATEX_TABLE_POSTAMBLE = r"""\bottomrule
\end{tabular}
\end{table}
"""


def format_number(val: float, precision: int = 3) -> str:
    """Format number for display, handling None/NaN."""
    if val is None:
        return "N/A"
    try:
        import numpy as np

        if np.isnan(val):
            return "N/A"
        if abs(val) < 0.001:
            return f"{val:.{precision}e}"
        return f"{val:.{precision}f}"
    except:
        return str(val)


def verdict_to_symbol(verdict: str) -> str:
    """Convert verdict string to LaTeX symbol."""
    v = verdict.upper()
    if "PASS" in v or "GO" in v:
        return r"\checkmark"
    elif "FAIL" in v:
        return r"\texttimes"
    elif "CAUTION" in v or "WARN" in v:
        return r"\textexclamdown"
    elif "INFO" in v:
        return r"\circ"
    else:
        return "-"
