"""Cubature schemes for ball/Gaussian source integration.

Schemes:
- SR6_7pt: Stroud-Rule degree-5 (center + 6 face centers)
- grid_27pt: 3×3×3 uniform grid
- strat_33pt: Stratified random
- lebedev_50x4: Lebedev-50 (angular) × Gauss-4 (radial)
- halton_64pt: Halton quasi-random sequence
"""

import numpy as np
from scipy.special import roots_legendre

from .config import OPTICAL


def sr6_7pt() -> tuple:
    """Stroud-Rule degree-5, 7-point scheme.

    Center + 6 face centers of cube.
    Weights optimized for degree-5 polynomial accuracy.

    Returns
    -------
    points : (7, 3) array
    weights : (7,) array
    """
    a = np.sqrt(14.0 / 15.0)
    w_center = 2.0 / 7.0
    w_face = 5.0 / 42.0

    points = np.zeros((7, 3))
    points[0] = [0, 0, 0]
    points[1] = [a, 0, 0]
    points[2] = [-a, 0, 0]
    points[3] = [0, a, 0]
    points[4] = [0, -a, 0]
    points[5] = [0, 0, a]
    points[6] = [0, 0, -a]

    weights = np.array([w_center] + [w_face] * 6)

    return points, weights


def grid_27pt() -> tuple:
    """3×3×3 uniform grid with equal weights.

    Returns
    -------
    points : (27, 3) array
    weights : (27,) array
    """
    coords = np.array([-1.0, 0.0, 1.0])
    points = []
    for x in coords:
        for y in coords:
            for z in coords:
                points.append([x, y, z])
    points = np.array(points)
    weights = np.ones(27) / 27.0

    return points, weights


def stratified_33pt(seed: int = 42) -> tuple:
    """Stratified random sampling, 33 points.

    Returns
    -------
    points : (33, 3) array
    weights : (33,) array
    """
    rng = np.random.default_rng(seed)
    n = 33
    n_per_dim = int(np.ceil(n ** (1 / 3)))
    n_total = n_per_dim**3

    points = []
    step = 2.0 / n_per_dim
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                x = -1 + step * (i + rng.random())
                y = -1 + step * (j + rng.random())
                z = -1 + step * (k + rng.random())
                points.append([x, y, z])

    points = np.array(points[:n])
    weights = np.ones(n) / n

    return points, weights


def lebedev_50x4() -> tuple:
    """Lebedev-50 (angular) × Gauss-4 (radial) product rule.

    Returns
    -------
    points : (200, 3) array
    weights : (200,) array
    """
    n_angular = 50
    n_radial = 4

    theta = np.linspace(0, np.pi, int(np.sqrt(n_angular)))
    phi = np.linspace(0, 2 * np.pi, int(np.sqrt(n_angular)))
    THETA, PHI = np.meshgrid(theta, phi)
    THETA = THETA.ravel()
    PHI = PHI.ravel()

    ang_points = np.zeros((len(THETA), 3))
    ang_points[:, 0] = np.sin(THETA) * np.cos(PHI)
    ang_points[:, 1] = np.sin(THETA) * np.sin(PHI)
    ang_points[:, 2] = np.cos(THETA)
    ang_weights = np.ones(len(THETA)) / len(THETA)

    r_nodes, r_weights = roots_legendre(n_radial)
    r = 0.5 * (r_nodes + 1.0)
    r_weights = 0.5 * r_weights

    points = []
    weights = []
    for ri, wi in zip(r, r_weights):
        for j, (ang_p, ang_w) in enumerate(zip(ang_points, ang_weights)):
            points.append(ri * ang_p)
            weights.append(wi * ang_w * ri**2)

    points = np.array(points)
    weights = np.array(weights)
    weights /= weights.sum()

    return points, weights


def halton_64pt(seed: int = 0) -> tuple:
    """Halton quasi-random sequence, 64 points.

    Returns
    -------
    points : (64, 3) array
    weights : (64,) array
    """
    n = 64

    def halton(b, n, skip=seed):
        seq = np.zeros(n)
        for i in range(n):
            i_skip = i + skip + 1
            f = 1.0 / b
            val = 0.0
            while i_skip > 0:
                val += f * (i_skip % b)
                i_skip = i_skip // b
                f /= b
            seq[i] = val
        return seq

    x = halton(2, n)
    y = halton(3, n)
    z = halton(5, n)

    points = np.column_stack([x, y, z]) * 2 - 1
    weights = np.ones(n) / n

    return points, weights


SCHEME_FUNCTIONS = {
    "SR6_7pt": sr6_7pt,
    "grid_27pt": grid_27pt,
    "strat_33pt": stratified_33pt,
    "lebedev_50x4": lebedev_50x4,
    "halton_64pt": halton_64pt,
}


def get_scheme(name: str, **kwargs) -> tuple:
    """Get cubature scheme by name.

    Parameters
    ----------
    name : str
        Scheme name.
    **kwargs
        Additional arguments (e.g., seed).

    Returns
    -------
    points, weights : tuple
    """
    if name not in SCHEME_FUNCTIONS:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(SCHEME_FUNCTIONS.keys())}"
        )
    return SCHEME_FUNCTIONS[name](**kwargs)


def map_to_ball(points: np.ndarray, weights: np.ndarray, R: float) -> tuple:
    """Map unit cube [-1,1]³ points to ball of radius R.

    Uses proper uniform sampling in ball:
    - Map cube to sphere surface (normalize)
    - Scale radius by cube root for uniform volume density

    Parameters
    ----------
    points : (N, 3) array
        Points in unit cube.
    weights : (N,) array
        Weights (should sum to 1).
    R : float
        Ball radius.

    Returns
    -------
    ball_points : (N, 3) array
    ball_weights : (N,) array
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    unit_vecs = points / norms

    r_cube = norms.ravel()
    r_ball = np.cbrt(r_cube) * R

    ball_points = unit_vecs * r_ball[:, np.newaxis]

    volume = 4.0 / 3.0 * np.pi * R**3
    ball_weights = weights * volume

    return ball_points, ball_weights
