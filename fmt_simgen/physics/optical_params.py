"""
OpticalParameterManager: Manage optical properties for each tissue type.

Handles:
- Loading optical parameters from config
- Computing derived quantities (D, mu_sp, etc.)
- Providing lookup by tissue label
"""

import numpy as np
from typing import Dict, List, Optional


class TissueOpticalParams:
    """Optical parameters for a single tissue type."""

    def __init__(
        self,
        label: int,
        name: str,
        mu_a: float,
        mu_s_prime: float,
        g: float,
        n: float = 1.37,
    ):
        """Initialize tissue optical parameters.

        Parameters
        ----------
        label : int
            Tissue label ID.
        name : str
            Tissue name.
        mu_a : float
            Absorption coefficient (mm^-1).
        mu_s_prime : float
            Reduced scattering coefficient (mm^-1).
        g : float
            Anisotropy factor.
        n : float, default 1.37
            Refractive index.
        """
        self.label = label
        self.name = name
        self.mu_a = mu_a
        self.mu_s_prime = mu_s_prime
        self.g = g
        self.n = n

        if mu_a < 1e-10 and mu_s_prime < 1e-10:
            self._D = 1e10
        else:
            self._D = 1.0 / (3.0 * (mu_a + mu_s_prime))

    @property
    def mu_sp_prime(self) -> float:
        """Get reduced scattering coefficient (mm^-1)."""
        return self._mu_sp_prime

    @property
    def D(self) -> float:
        """Get diffusion coefficient (mm)."""
        return self._D

    def __repr__(self) -> str:
        return (
            f"TissueOpticalParams(label={self.label}, name='{self.name}', "
            f"mu_a={self.mu_a:.4f}, mu_s_prime={self.mu_s_prime:.4f}, "
            f"g={self.g}, n={self.n}, D={self.D:.4f})"
        )


class OpticalParameterManager:
    """Manage optical parameters for all tissue types."""

    def __init__(self, config: Dict, n: float = 1.37):
        """Initialize optical parameter manager.

        Parameters
        ----------
        config : Dict
            Configuration dictionary with tissue parameters.
            Expected format:
            {
                "muscle": {"label": 1, "mu_a": 0.08697, "mu_sp": 4.29071, "g": 0.90},
                ...
            }
        n : float, default 1.37
            Default refractive index.
        """
        self.n = n
        self._tissues: Dict[int, TissueOpticalParams] = {}
        self._name_map: Dict[str, int] = {}

        for tissue_name, params in config.items():
            mu_s_prime = params.get("mu_s_prime", params.get("mu_sp", 0.0))
            tissue = TissueOpticalParams(
                label=params["label"],
                name=tissue_name,
                mu_a=params["mu_a"],
                mu_s_prime=mu_s_prime,
                g=params["g"],
                n=n,
            )
            self._tissues[tissue.label] = tissue
            self._name_map[tissue_name] = tissue.label

    def get_by_label(self, label: int) -> TissueOpticalParams:
        """Get optical parameters by tissue label.

        Parameters
        ----------
        label : int
            Tissue label ID.

        Returns
        -------
        TissueOpticalParams
            Optical parameters for the tissue.
        """
        if label not in self._tissues:
            raise ValueError(f"Unknown tissue label: {label}")
        return self._tissues[label]

    def get_by_name(self, name: str) -> TissueOpticalParams:
        """Get optical parameters by tissue name.

        Parameters
        ----------
        name : str
            Tissue name (e.g., 'muscle', 'liver').

        Returns
        -------
        TissueOpticalParams
            Optical parameters for the tissue.
        """
        if name not in self._name_map:
            raise ValueError(f"Unknown tissue name: {name}")
        return self.get_by_label(self._name_map[name])

    def get_multi_params(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Get arrays of optical parameters for multiple elements.

        Parameters
        ----------
        labels : np.ndarray
            Array of tissue labels [M] for M elements.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with arrays: 'mu_a', 'mu_s_prime', 'D', 'g', 'n'.
        """
        mu_a = np.array([self.get_by_label(l).mu_a for l in labels])
        mu_s_prime = np.array([self.get_by_label(l).mu_s_prime for l in labels])
        D = np.array([self.get_by_label(l).D for l in labels])
        g = np.array([self.get_by_label(l).g for l in labels])
        n = np.array([self.get_by_label(l).n for l in labels])

        return {"mu_a": mu_a, "mu_s_prime": mu_s_prime, "D": D, "g": g, "n": n}

    @staticmethod
    def compute_ro_and_an(n: float) -> tuple:
        """Compute reflection coefficient and An for Robin boundary condition.

        Parameters
        ----------
        n : float
            Refractive index.

        Returns
        -------
        Tuple[float, float]
            (R, An) where R is reflection coefficient and An = (1+R)/(1-R).
        """
        R = -1.4399 / (n**2) + 0.7099 / n + 0.6681 + 0.0636 * n
        An = (1.0 + R) / (1.0 - R)
        return R, An

    def list_tissues(self) -> List[TissueOpticalParams]:
        """List all registered tissue optical parameters.

        Returns
        -------
        List[TissueOpticalParams]
            List of all tissue parameters.
        """
        return list(self._tissues.values())
