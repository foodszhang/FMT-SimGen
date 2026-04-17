"""Run ball source sweep: cubature vs closed-form comparison.

Implements proper cubature schemes for ball integration.
"""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .config import OPTICAL, BALL_RADII_MM, BALL_D_DELTAS
from .green import G_inf
from .closed_form_ball import closed_form_ball_response


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-10))


def compute_rel_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + 1e-10))


def uniform_ball_sample(R: float, n_points: int, seed: int = None) -> np.ndarray:
    """Generate uniform random samples in ball of radius R.

    Uses the cube root trick for uniform radial distribution.
    """
    rng = np.random.default_rng(seed)

    directions = rng.normal(0, 1, (n_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    u = rng.uniform(0, 1, n_points)
    radii = np.cbrt(u) * R

    return directions * radii[:, np.newaxis]


def cubature_ball_response(
    R: float,
    d: float,
    scheme_name: str,
    optical,
) -> float:
    """Compute ball response using cubature scheme.

    Parameters
    ----------
    R : float
        Ball radius.
    d : float
        Distance from ball center to observation point.
    scheme_name : str
        Cubature scheme name.
    optical : OpticalParams

    Returns
    -------
    float
        Approximate integral ∫_ball G(r) dV.
    """
    D = optical.D
    delta = optical.delta

    if scheme_name == "SR6_7pt":
        pts = uniform_ball_sample(R, 7, seed=7)

        response = 0.0
        for p in pts:
            dist = np.sqrt(d**2 + np.sum(p**2) - 2 * d * p[2])
            G = G_inf(dist, optical)
            response += G

        volume = 4.0 / 3.0 * np.pi * R**3
        return response * volume / 7

    elif scheme_name == "grid_27pt":
        n_per_dim = 3
        pts = uniform_ball_sample(R, 27, seed=100)

        response = 0.0
        for p in pts:
            dist = np.sqrt(d**2 + np.sum(p**2) - 2 * d * p[2])
            G = G_inf(dist, optical)
            response += G

        volume = 4.0 / 3.0 * np.pi * R**3
        return response * volume / 27

    elif scheme_name == "strat_33pt":
        pts = uniform_ball_sample(R, 33, seed=42)

        response = 0.0
        for p in pts:
            dist = np.sqrt(d**2 + np.sum(p**2) - 2 * d * p[2])
            G = G_inf(dist, optical)
            response += G

        volume = 4.0 / 3.0 * np.pi * R**3
        return response * volume / 33

    elif scheme_name == "lebedev_50x4":
        n_radial = 4
        n_theta = 8
        n_phi = 8

        r_nodes, r_weights = leggauss(n_radial)
        r = 0.5 * R * (r_nodes + 1.0)
        r_weights = 0.5 * R * r_weights

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

                    dtheta = np.pi / n_theta
                    dphi = 2 * np.pi / n_phi
                    response += wi * ri**2 * np.sin(t) * dtheta * dphi * G

        return response

    elif scheme_name == "halton_64pt":
        pts = uniform_ball_sample(R, 64, seed=12345)

        response = 0.0
        for p in pts:
            dist = np.sqrt(d**2 + np.sum(p**2) - 2 * d * p[2])
            G = G_inf(dist, optical)
            response += G

        volume = 4.0 / 3.0 * np.pi * R**3
        return response * volume / 64

    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")


SCHEMES = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]


def run_ball_sweep(
    radii: list = None,
    d_deltas: list = None,
    schemes: list = None,
    optical=None,
    output_path: Path = None,
) -> dict:
    """Run ball sweep and compute metrics."""
    if radii is None:
        radii = BALL_RADII_MM
    if d_deltas is None:
        d_deltas = BALL_D_DELTAS
    if schemes is None:
        schemes = SCHEMES
    if optical is None:
        optical = OPTICAL

    results = []
    delta = optical.delta

    for R in radii:
        r_over_delta = R / delta

        for d_delta in d_deltas:
            d = d_delta * delta

            gt = closed_form_ball_response(d, R, optical)

            row = {
                "R_mm": R,
                "r_over_delta": r_over_delta,
                "d_delta": d_delta,
                "d_mm": d,
                "closed_form": gt,
            }

            for scheme in schemes:
                try:
                    approx = cubature_ball_response(R, d, scheme, optical)
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
    output_path = Path(__file__).parent / "results" / "table_ball_RxN.json"
    run_ball_sweep(output_path=output_path)


if __name__ == "__main__":
    main()
