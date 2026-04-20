"""Metrics for comparing MCX vs closed-form forward models."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Linear-space Pearson correlation coefficient (primary metric)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


def ncc_log(a: np.ndarray, b: np.ndarray, eps: float = 1e-20) -> float:
    """Log-space NCC (auxiliary metric only). Do NOT use for paper primary tables."""
    a = np.log10(np.asarray(a).flatten().astype(np.float64) + eps)
    b = np.log10(np.asarray(b).flatten().astype(np.float64) + eps)
    a_mean, b_mean = np.mean(a), np.mean(b)
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    return float(num / den) if den > 1e-12 else 0.0


def scale_factor_k(a: np.ndarray, b: np.ndarray) -> float:
    """Linear scale factor: sum(a) / sum(b). Optimal for linear-MSE loss."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_b < 1e-12:
        return 0.0
    return float(sum_a / sum_b)


def scale_factor_logmse(
    meas: np.ndarray, forward: np.ndarray, eps: float = 1e-20
) -> float:
    """Optimal scale for log-MSE loss (geomean).

    For loss = mean((log(m) - log(s*f))^2), optimal s = 10^mean(log(m) - log(f)).
    """
    meas = np.asarray(meas).flatten().astype(np.float64)
    forward = np.asarray(forward).flatten().astype(np.float64)
    valid = (meas > 0) & (forward > 0)
    if np.sum(valid) < 10:
        return 0.0
    return float(
        10 ** np.mean(np.log10(meas[valid] + eps) - np.log10(forward[valid] + eps))
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def peak_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    peak_a = np.max(np.abs(a))
    peak_b = np.max(np.abs(b))
    if peak_b < 1e-12:
        return 0.0
    return float(peak_a / peak_b)


def compute_all_metrics(mcx_proj: np.ndarray, closed_proj: np.ndarray) -> dict:
    mcx_valid = mcx_proj.flatten()
    closed_valid = closed_proj.flatten()
    mask = (mcx_valid > 0) | (closed_valid > 0)
    if np.sum(mask) < 10:
        return {
            "ncc": 0.0,
            "k": 0.0,
            "rmse": float("inf"),
            "peak_ratio": 0.0,
            "n_valid": int(np.sum(mask)),
        }
    mcx_roi = mcx_valid[mask]
    closed_roi = closed_valid[mask]
    return {
        "ncc": ncc(mcx_roi, closed_roi),
        "k": scale_factor_k(mcx_roi, closed_roi),
        "rmse": rmse(mcx_roi, closed_roi),
        "peak_ratio": peak_ratio(mcx_roi, closed_roi),
        "n_valid": int(np.sum(mask)),
    }


def metrics_summary(metrics: dict) -> str:
    return (
        f"NCC={metrics['ncc']:.4f}, "
        f"k={metrics['k']:.2e}, "
        f"RMSE={metrics['rmse']:.2e}, "
        f"peak_ratio={metrics['peak_ratio']:.2f}"
    )
