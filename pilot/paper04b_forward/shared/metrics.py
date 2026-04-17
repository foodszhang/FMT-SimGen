"""Metrics for MCX vs Green comparison."""

import numpy as np


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cross-correlation."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    a_mean = a_flat.mean()
    b_mean = b_flat.mean()

    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean

    num = np.dot(a_centered, b_centered)
    denom = np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered))

    if denom < 1e-10:
        return 0.0

    return float(num / denom)


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_scale_factor(mcx: np.ndarray, green: np.ndarray) -> float:
    """Compute scale factor k = sum(MCX) / sum(Green)."""
    green_sum = np.sum(green)
    if green_sum < 1e-10:
        return 0.0
    return float(np.sum(mcx) / green_sum)


def compute_peak_ratio(mcx: np.ndarray, green: np.ndarray) -> float:
    """Compute peak ratio = max(MCX) / max(Green)."""
    green_max = np.max(green)
    if green_max < 1e-10:
        return 0.0
    return float(np.max(mcx) / green_max)


def compute_all_metrics(mcx: np.ndarray, green: np.ndarray) -> dict:
    """Compute all comparison metrics.

    Returns
    -------
    dict with keys: ncc, rmse, scale_factor, peak_ratio
    """
    return {
        "ncc": compute_ncc(mcx, green),
        "rmse": compute_rmse(mcx, green),
        "scale_factor": compute_scale_factor(mcx, green),
        "peak_ratio": compute_peak_ratio(mcx, green),
    }
