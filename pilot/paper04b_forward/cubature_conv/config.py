"""Configuration for cubature convergence study (Part C, §4.A extension).

Physical parameters: soft_tissue NIR-II baseline.
"""

from dataclasses import dataclass


@dataclass
class OpticalParams:
    mu_a: float = 0.087
    mus_p: float = 4.3
    g: float = 0.9
    n: float = 1.37

    @property
    def D(self) -> float:
        return 1.0 / (3.0 * self.mus_p)

    @property
    def delta(self) -> float:
        return (self.D / self.mu_a) ** 0.5

    @property
    def mu_eff(self) -> float:
        return 1.0 / self.delta


OPTICAL = OpticalParams()

BALL_RADII_MM = [0.5, 1.0, 2.0, 3.0, 5.0]
BALL_D_DELTAS = [3, 5, 10]

GAUSSIAN_SIGMAS_MM = [0.3, 0.5, 1.0, 2.0, 3.0]
GAUSSIAN_D_DELTAS = [3, 5]

GRID_SIZE = 256
WINDOW_DELTA = 10.0

SCHEMES = ["SR6_7pt", "grid_27pt", "strat_33pt", "lebedev_50x4", "halton_64pt"]
