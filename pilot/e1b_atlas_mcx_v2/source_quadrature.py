#!/usr/bin/env python3
"""Source quadrature schemes for E1d.

Implements proper integration schemes for finite-size sources:
- SR-6: 3D Spherical-Radial cubature (6 points, moment-matched)
- UT-7: Unscented Transform (7 points, center + 6 axial)
- Grid-27: 3x3x3 grid (reference)
- Stratified-33/57: Stratified sampling for uniform ellipsoid
- MC-128/256: Monte Carlo reference
"""

from typing import Tuple, Optional
import numpy as np


SAMPLING_SCHEMES = {
    "1-point": {"n_points": 1, "description": "Single center point"},
    "sr-6": {
        "n_points": 6,
        "description": "Spherical-Radial cubature (moment-matched)",
    },
    "ut-7": {"n_points": 7, "description": "Unscented Transform (center + 6 axial)"},
    "7-point": {"n_points": 7, "description": "Legacy 7-point (center + 6 face)"},
    "19-point": {"n_points": 19, "description": "Center + 6 face + 12 edge centers"},
    "grid-27": {"n_points": 27, "description": "3x3x3 grid"},
    "stratified-33": {
        "n_points": 33,
        "description": "Stratified uniform (center + 6 + 12 + 8 + 6)",
    },
    "stratified-71": {
        "n_points": 71,
        "description": "Stratified uniform (center + 18 axial + 36 edge + 16 body)",
    },
    "mc-128": {"n_points": 128, "description": "Monte Carlo 128 samples"},
    "mc-256": {"n_points": 256, "description": "Monte Carlo 256 samples"},
    "mc-512": {"n_points": 512, "description": "Monte Carlo 512 samples"},
}


def sample_gaussian_sr6(
    center: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """SR-6: 3D Spherical-Radial cubature for Gaussian integral.

    Uses 6 symmetric points along principal axes:
        x = μ ± sqrt(3) * σ_i * e_i

    This exactly integrates all 2nd-order polynomials under Gaussian measure.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity

    Returns:
        points: [6, 3] sample positions
        weights: [6] equal weights, sum = alpha
    """
    sqrt3 = np.sqrt(3.0)

    points = np.zeros((6, 3), dtype=np.float32)

    for i in range(3):
        points[2 * i, i] = sqrt3 * sigmas[i]
        points[2 * i + 1, i] = -sqrt3 * sigmas[i]

    points = points + center

    weights = np.ones(6, dtype=np.float32) / 6.0 * alpha

    return points, weights


def sample_gaussian_ut7(
    center: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
    kappa: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """UT-7: Unscented Transform sigma points for Gaussian.

    Uses center + 6 axial points:
        x_0 = μ
        x_i = μ + sqrt(3 + κ) * σ_i * e_i    (i = 1..6)

    Weights:
        w_0 = κ / (3 + κ)
        w_i = 1 / (2 * (3 + κ))              (i = 1..6)

    For κ=0, this reduces to SR-6 with center point added.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity
        kappa: scaling parameter (default 0)

    Returns:
        points: [7, 3] sample positions
        weights: [7] weights, sum = alpha
    """
    n = 3
    gamma = np.sqrt(n + kappa)

    points = np.zeros((7, 3), dtype=np.float32)

    points[0] = center

    for i in range(3):
        points[1 + 2 * i, i] = gamma * sigmas[i]
        points[2 + 2 * i, i] = -gamma * sigmas[i]

    points[1:] = points[1:] + center

    weights = np.zeros(7, dtype=np.float32)
    weights[0] = kappa / (n + kappa)
    weights[1:] = 1.0 / (2.0 * (n + kappa))

    if weights[0] < 0:
        weights[0] = 0
        weights = weights / weights.sum()

    weights = weights * alpha

    return points, weights


def sample_gaussian_grid27(
    center: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid-27: 3x3x3 grid for Gaussian integral.

    Uses points at μ ± σ along each axis.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity

    Returns:
        points: [27, 3] sample positions
        weights: [27] Gaussian weights, sum = alpha
    """
    lin = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    points_norm = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    points = points_norm * sigmas + center

    diff = points - center
    diff = diff / sigmas
    dist_sq = np.sum(diff**2, axis=1)

    weights = np.exp(-0.5 * dist_sq)
    weights = weights / weights.sum() * alpha

    return points.astype(np.float32), weights.astype(np.float32)


def sample_gaussian_mc(
    center: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
    n_samples: int = 128,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo sampling for Gaussian integral.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity
        n_samples: number of MC samples
        seed: random seed for reproducibility

    Returns:
        points: [n_samples, 3] sample positions
        weights: [n_samples] equal weights, sum = alpha
    """
    if seed is not None:
        np.random.seed(seed)

    points = np.random.randn(n_samples, 3).astype(np.float32)
    points = points * sigmas + center

    weights = np.ones(n_samples, dtype=np.float32) / n_samples * alpha

    return points, weights


def sample_uniform_stratified33(
    center: np.ndarray,
    axes: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified-33: 33-point scheme for uniform ellipsoid integral.

    Layout:
        - 1 center point
        - 6 face centers (±0.5 along axes)
        - 12 edge centers (±0.35 along 2 axes)
        - 8 body centers (±0.25 along all 3 axes)
        - 6 outer points (±0.7 along axes)

    Args:
        center: [3] center position
        axes: [3] semi-axis lengths
        alpha: total intensity

    Returns:
        points: [33, 3] sample positions
        weights: [33] volume-matched weights, sum = alpha
    """
    points_norm = np.zeros((33, 3), dtype=np.float32)

    idx = 0

    points_norm[idx] = [0, 0, 0]
    idx += 1

    for i in range(3):
        points_norm[idx] = [0, 0, 0]
        points_norm[idx, i] = 0.5
        idx += 1
        points_norm[idx] = [0, 0, 0]
        points_norm[idx, i] = -0.5
        idx += 1

    edge_r = 0.35
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    points_norm[idx] = [0, 0, 0]
                    points_norm[idx, i] = s1 * edge_r
                    points_norm[idx, j] = s2 * edge_r
                    idx += 1

    body_r = 0.25
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                points_norm[idx] = [s1 * body_r, s2 * body_r, s3 * body_r]
                idx += 1

    outer_r = 0.7
    for i in range(3):
        points_norm[idx] = [0, 0, 0]
        points_norm[idx, i] = outer_r
        idx += 1
        points_norm[idx] = [0, 0, 0]
        points_norm[idx, i] = -outer_r
        idx += 1

    points = points_norm * axes + center

    weights = np.ones(33, dtype=np.float32)
    weights[0] = 0.06
    weights[1:7] = 0.035
    weights[7:19] = 0.025
    weights[19:27] = 0.018
    weights[27:33] = 0.012

    weights = weights / weights.sum() * alpha

    return points, weights


def sample_uniform_stratified71(
    center: np.ndarray,
    axes: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified-71: 71-point scheme for uniform ellipsoid integral.

    Layout:
        - 1 center point
        - 3 radii × 6 axial = 18 axial points
        - 3 radii × 12 edge = 36 edge points
        - 2 radii × 8 body = 16 body points
        - Total = 1 + 18 + 36 + 16 = 71

    Args:
        center: [3] center position
        axes: [3] semi-axis lengths
        alpha: total intensity

    Returns:
        points: [71, 3] sample positions
        weights: [71] volume-matched weights, sum = alpha
    """
    n_points = 71
    points_norm = np.zeros((n_points, 3), dtype=np.float32)

    idx = 0

    points_norm[idx] = [0, 0, 0]
    idx += 1

    for r in [0.3, 0.6, 0.9]:
        for i in range(3):
            points_norm[idx] = [0, 0, 0]
            points_norm[idx, i] = r
            idx += 1
            points_norm[idx] = [0, 0, 0]
            points_norm[idx, i] = -r
            idx += 1

    for r in [0.25, 0.5, 0.75]:
        for i in range(3):
            for j in range(i + 1, 3):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        points_norm[idx] = [0, 0, 0]
                        points_norm[idx, i] = s1 * r
                        points_norm[idx, j] = s2 * r
                        idx += 1

    for r in [0.3, 0.6]:
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    points_norm[idx] = [s1 * r, s2 * r, s3 * r]
                    idx += 1

    points = points_norm * axes + center

    weights = np.ones(n_points, dtype=np.float32)
    weights[0] = 0.03
    weights[1:19] = 0.018
    weights[19:55] = 0.012
    weights[55:71] = 0.014

    weights = weights / weights.sum() * alpha

    return points, weights


def sample_uniform_mc(
    center: np.ndarray,
    axes: np.ndarray,
    alpha: float,
    n_samples: int = 256,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo sampling for uniform ellipsoid integral.

    Rejection sampling to ensure points are inside ellipsoid.

    Args:
        center: [3] center position
        axes: [3] semi-axis lengths
        alpha: total intensity
        n_samples: number of MC samples
        seed: random seed

    Returns:
        points: [n_samples, 3] sample positions
        weights: [n_samples] equal weights, sum = alpha
    """
    if seed is not None:
        np.random.seed(seed)

    points = np.zeros((n_samples, 3), dtype=np.float32)
    count = 0

    while count < n_samples:
        batch = np.random.uniform(-1, 1, (n_samples, 3)).astype(np.float32)

        dist_sq = (
            (batch[:, 0] / axes[0]) ** 2
            + (batch[:, 1] / axes[1]) ** 2
            + (batch[:, 2] / axes[2]) ** 2
        )
        inside = dist_sq <= 1.0

        n_inside = min(inside.sum(), n_samples - count)
        if n_inside > 0:
            points[count : count + n_inside] = batch[inside][:n_inside] * axes + center
            count += n_inside

    weights = np.ones(n_samples, dtype=np.float32) / n_samples * alpha

    return points, weights


def sample_gaussian(
    center: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
    scheme: str = "sr-6",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample Gaussian source with specified scheme.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity
        scheme: sampling scheme name
        seed: random seed for MC schemes

    Returns:
        points: [N, 3] sample positions
        weights: [N] weights, sum = alpha
    """
    if scheme == "1-point":
        points = center.reshape(1, 3).astype(np.float32)
        weights = np.array([alpha], dtype=np.float32)
        return points, weights

    elif scheme == "sr-6":
        return sample_gaussian_sr6(center, sigmas, alpha)

    elif scheme == "ut-7":
        return sample_gaussian_ut7(center, sigmas, alpha)

    elif scheme == "7-point":
        points_norm = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )
        points = points_norm * sigmas + center
        dist_sq = np.sum(points_norm**2, axis=1)
        weights = np.exp(-0.5 * dist_sq)
        weights = weights / weights.sum() * alpha
        return points, weights.astype(np.float32)

    elif scheme == "19-point":
        offset = 1.0
        edge_off = 0.7
        points_norm = np.array(
            [
                [0, 0, 0],
                [offset, 0, 0],
                [-offset, 0, 0],
                [0, offset, 0],
                [0, -offset, 0],
                [0, 0, offset],
                [0, 0, -offset],
                [edge_off, edge_off, 0],
                [edge_off, -edge_off, 0],
                [-edge_off, edge_off, 0],
                [-edge_off, -edge_off, 0],
                [edge_off, 0, edge_off],
                [edge_off, 0, -edge_off],
                [-edge_off, 0, edge_off],
                [-edge_off, 0, -edge_off],
                [0, edge_off, edge_off],
                [0, edge_off, -edge_off],
                [0, -edge_off, edge_off],
                [0, -edge_off, -edge_off],
            ],
            dtype=np.float32,
        )
        points = points_norm * sigmas + center
        diff = points_norm
        dist_sq = np.sum(diff**2, axis=1)
        weights = np.exp(-0.5 * dist_sq)
        weights = weights / weights.sum() * alpha
        return points, weights.astype(np.float32)

    elif scheme == "grid-27":
        return sample_gaussian_grid27(center, sigmas, alpha)

    elif scheme.startswith("mc-"):
        n_samples = int(scheme.split("-")[1])
        return sample_gaussian_mc(center, sigmas, alpha, n_samples, seed)

    else:
        raise ValueError(f"Unknown Gaussian sampling scheme: {scheme}")


def sample_uniform(
    center: np.ndarray,
    axes: np.ndarray,
    alpha: float,
    scheme: str = "stratified-33",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample uniform ellipsoid source with specified scheme.

    Args:
        center: [3] center position
        axes: [3] semi-axis lengths
        alpha: total intensity
        scheme: sampling scheme name
        seed: random seed for MC schemes

    Returns:
        points: [N, 3] sample positions
        weights: [N] weights, sum = alpha
    """
    if scheme == "1-point":
        points = center.reshape(1, 3).astype(np.float32)
        weights = np.array([alpha], dtype=np.float32)
        return points, weights

    elif scheme == "7-point":
        points_norm = np.array(
            [
                [0, 0, 0],
                [0.5, 0, 0],
                [-0.5, 0, 0],
                [0, 0.5, 0],
                [0, -0.5, 0],
                [0, 0, 0.5],
                [0, 0, -0.5],
            ],
            dtype=np.float32,
        )
        points = points_norm * axes + center
        weights = np.ones(7, dtype=np.float32) / 7.0 * alpha
        return points, weights

    elif scheme == "19-point":
        offset = 0.5
        edge_off = 0.35
        points_norm = np.array(
            [
                [0, 0, 0],
                [offset, 0, 0],
                [-offset, 0, 0],
                [0, offset, 0],
                [0, -offset, 0],
                [0, 0, offset],
                [0, 0, -offset],
                [edge_off, edge_off, 0],
                [edge_off, -edge_off, 0],
                [-edge_off, edge_off, 0],
                [-edge_off, -edge_off, 0],
                [edge_off, 0, edge_off],
                [edge_off, 0, -edge_off],
                [-edge_off, 0, edge_off],
                [-edge_off, 0, -edge_off],
                [0, edge_off, edge_off],
                [0, edge_off, -edge_off],
                [0, -edge_off, edge_off],
                [0, -edge_off, -edge_off],
            ],
            dtype=np.float32,
        )
        points = points_norm * axes + center
        weights = np.ones(19, dtype=np.float32) / 19.0 * alpha
        return points, weights

    elif scheme == "grid-27":
        lin = np.linspace(-0.5, 0.5, 3)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
        points_norm = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
            np.float32
        )
        points = points_norm * axes + center
        weights = np.ones(27, dtype=np.float32) / 27.0 * alpha
        return points, weights

    elif scheme == "stratified-33":
        return sample_uniform_stratified33(center, axes, alpha)

    elif scheme == "stratified-71":
        return sample_uniform_stratified71(center, axes, alpha)

    elif scheme.startswith("mc-"):
        n_samples = int(scheme.split("-")[1])
        return sample_uniform_mc(center, axes, alpha, n_samples, seed)

    else:
        raise ValueError(f"Unknown uniform sampling scheme: {scheme}")


def get_sampling_scheme(scheme: str) -> dict:
    """Get sampling scheme info."""
    if scheme not in SAMPLING_SCHEMES:
        raise ValueError(
            f"Unknown scheme: {scheme}. Available: {list(SAMPLING_SCHEMES.keys())}"
        )
    return SAMPLING_SCHEMES[scheme]


def select_adaptive_scheme(
    source_extent_mm: float,
    depth_mm: float,
    mu_eff_mm: float,
    mode: str = "gaussian",
) -> str:
    """Select sampling scheme adaptively based on source/geometry properties.

    Args:
        source_extent_mm: approximate source extent
        depth_mm: source depth from surface
        mu_eff_mm: effective attenuation coefficient
        mode: "gaussian" or "uniform"

    Returns:
        recommended scheme name
    """
    extent_ratio = source_extent_mm / max(depth_mm, 0.1)

    mu_source = mu_eff_mm * source_extent_mm

    if extent_ratio < 0.15 and mu_source < 0.3:
        return "1-point"
    elif extent_ratio < 0.25 and mu_source < 0.6:
        return "sr-6" if mode == "gaussian" else "7-point"
    elif extent_ratio < 0.35 and mu_source < 1.0:
        return "ut-7" if mode == "gaussian" else "stratified-33"
    else:
        return "grid-27" if mode == "gaussian" else "stratified-57"
