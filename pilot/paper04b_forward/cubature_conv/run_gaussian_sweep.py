"""Run Gaussian source sweep: cubature vs closed-form comparison."""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import OPTICAL, GAUSSIAN_SIGMAS_MM, GAUSSIAN_D_DELTAS
from .green import G_inf
from .closed_form_gaussian_fft import closed_form_gaussian_point
from .run_ball_sweep import compute_ncc, compute_rel_l1


def gaussian_sample(sigma: float, n_points: int, seed: int = None) -> np.ndarray:
    """Generate samples from 3D Gaussian distribution.

    Returns points sampled from N(0, σ²I₃).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, (n_points, 3))


def uniform_ball_sample(R: float, n_points: int, seed: int = None) -> np.ndarray:
    """Generate uniform random samples in ball of radius R."""
    rng = np.random.default_rng(seed)

    directions = rng.normal(0, 1, (n_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    u = rng.uniform(0, 1, n_points)
    radii = np.cbrt(u) * R

    return directions * radii[:, np.newaxis]


def cubature_gaussian_response(
    sigma: float,
    d: float,
    scheme_name: str,
    optical,
) -> float:
    """Compute Gaussian source response using cubature.

    Uses importance sampling from Gaussian distribution.

    Parameters
    ----------
    sigma : float
        Gaussian sigma (mm).
    d : float
        Distance from source center.
    scheme_name : str
    optical : OpticalParams

    Returns
    -------
    float
        ∫ G(r) * N(r) dV where N is the normalized 3D Gaussian.
    """
    D = optical.D
    delta = optical.delta

    if scheme_name == "SR6_7pt":
        n_points = 7
        seed = 7
    elif scheme_name == "grid_27pt":
        n_points = 27
        seed = 100
    elif scheme_name == "strat_33pt":
        n_points = 33
        seed = 42
    elif scheme_name == "lebedev_50x4":
        n_radial = 4
        n_theta = 8
        n_phi = 8

        r_nodes, r_weights = leggauss(n_radial)
        r = 0.5 * sigma * 2 * (r_nodes + 1.0)
        r_weights = 0.5 * sigma * 2 * r_weights

        theta = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / n_theta / 2
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False) + np.pi / n_phi

        response = 0.0
        for ri, wi in zip(r, r_weights):
            for t in theta:
                for p in phi:
                    x = ri * np.sin(t) * np.cos(p)
                    y = ri * np.sin(t) * np.sin(p)
                    z = ri * np.cos(t)

                    dist = np.sqrt(d**2 + ri**2 - 2 * d * z)
                    G = G_inf(dist, optical)

                    gauss_weight = np.exp(-(ri**2) / (2 * sigma**2))
                    gauss_weight /= (2 * np.pi * sigma**2) ** 1.5

                    dtheta = np.pi / n_theta
                    dphi = 2 * np.pi / n_phi
                    response += (
                        wi * ri**2 * np.sin(t) * dtheta * dphi * G * gauss_weight
                    )

        return response

    elif scheme_name == "halton_64pt":
        n_points = 64
        seed = 12345
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")

    pts = gaussian_sample(sigma, n_points, seed=seed)

    response = 0.0
    for p in pts:
        dist = np.sqrt(d**2 + np.sum(p**2) - 2 * d * p[2])
        G = G_inf(dist, optical)
        response += G

    return response / n_points


SCHEMES = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]


def run_gaussian_sweep(
    sigmas: list = None,
    d_deltas: list = None,
    schemes: list = None,
    optical=None,
    output_path: Path = None,
) -> dict:
    """Run Gaussian sweep and compute metrics."""
    if sigmas is None:
        sigmas = GAUSSIAN_SIGMAS_MM
    if d_deltas is None:
        d_deltas = GAUSSIAN_D_DELTAS
    if schemes is None:
        schemes = SCHEMES
    if optical is None:
        optical = OPTICAL

    results = []
    delta = optical.delta

    for sigma in sigmas:
        sigma_over_delta = sigma / delta

        for d_delta in d_deltas:
            d = d_delta * delta

            gt = closed_form_gaussian_point(d, sigma, optical)

            row = {
                "sigma_mm": sigma,
                "sigma_over_delta": sigma_over_delta,
                "d_delta": d_delta,
                "d_mm": d,
                "closed_form": gt,
            }

            for scheme in schemes:
                try:
                    approx = cubature_gaussian_response(sigma, d, scheme, optical)
                    rel_l1 = abs(approx - gt) / gt
                    ncc = 1.0 - rel_l1

                    row[f"{scheme}_response"] = approx
                    row[f"{scheme}_rel_l1"] = rel_l1
                    row[f"{scheme}_ncc"] = ncc
                except Exception as e:
                    row[f"{scheme}_response"] = None
                    row[f"{scheme}_rel_l1"] = None
                    row[f"{scheme}_ncc"] = None

            results.append(row)

    output = {
        "delta_mm": delta,
        "table": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

    return output


def main():
    output_path = Path(__file__).parent / "results" / "table_gaussian_sigmaxN.json"
    run_gaussian_sweep(output_path=output_path)


if __name__ == "__main__":
    main()
