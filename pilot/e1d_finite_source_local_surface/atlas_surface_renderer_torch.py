#!/usr/bin/env python3
"""Differentiable atlas surface renderer for E1d-R2.

PyTorch implementation supporting gradient-based optimization.
"""

from typing import Tuple, Optional
import torch
import numpy as np


def diffusion_params_torch(tissue_params: dict, device: torch.device) -> dict:
    """Compute diffusion parameters in PyTorch."""
    mua = tissue_params["mua_mm"]
    mus = tissue_params["mus_mm"]
    g = tissue_params.get("g", 0.9)
    n = tissue_params.get("n", 1.37)

    mus_prime = mus * (1 - g)
    D = 1.0 / (3.0 * (mua + mus_prime))
    mu_eff = torch.sqrt(torch.tensor(3.0 * mua * (mua + mus_prime), device=device))

    R_eff = 0.493
    A = (1 + R_eff) / (1 - R_eff)
    zb = 2 * A * D

    return {
        "D": torch.tensor(D, device=device),
        "mu_eff": mu_eff,
        "zb": torch.tensor(zb, device=device),
    }


def sample_gaussian_torch(
    center: torch.Tensor,
    sigmas: torch.Tensor,
    alpha: torch.Tensor,
    scheme: str = "sr-6",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample Gaussian source in PyTorch.

    Args:
        center: [3] mean position
        sigmas: [3] standard deviations
        alpha: total intensity
        scheme: sampling scheme

    Returns:
        points: [N, 3] sample positions
        weights: [N] sample weights
    """
    device = center.device
    dtype = center.dtype

    if scheme == "1-point":
        points = center.unsqueeze(0)
        weights = alpha.unsqueeze(0)
        return points, weights

    elif scheme == "sr-6":
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=device, dtype=dtype))
        points = torch.zeros((6, 3), device=device, dtype=dtype)
        for i in range(3):
            points[2 * i, i] = sqrt3 * sigmas[i]
            points[2 * i + 1, i] = -sqrt3 * sigmas[i]
        points = points + center
        weights = torch.ones(6, device=device, dtype=dtype) / 6.0 * alpha
        return points, weights

    elif scheme == "ut-7":
        n = 3
        kappa = 1.0
        gamma = torch.sqrt(torch.tensor(n + kappa, device=device, dtype=dtype))

        points = torch.zeros((7, 3), device=device, dtype=dtype)
        points[0] = center

        for i in range(3):
            points[1 + 2 * i, i] = gamma * sigmas[i]
            points[2 + 2 * i, i] = -gamma * sigmas[i]

        points[1:] = points[1:] + center

        weights = torch.zeros(7, device=device, dtype=dtype)
        weights[0] = kappa / (n + kappa)
        weights[1:] = 1.0 / (2.0 * (n + kappa))
        weights = weights * alpha

        return points, weights

    elif scheme == "7-point":
        points_norm = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            device=device,
            dtype=dtype,
        )
        points = points_norm * sigmas + center

        dist_sq = torch.sum((points - center) ** 2 / (sigmas**2 + 1e-10), dim=1)
        weights = torch.exp(-0.5 * dist_sq)
        weights = weights / weights.sum() * alpha

        return points, weights

    elif scheme == "grid-27":
        lin = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
        points_norm = torch.stack(
            [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1
        )

        points = points_norm * sigmas + center

        dist_sq = torch.sum((points - center) ** 2 / (sigmas**2 + 1e-10), dim=1)
        weights = torch.exp(-0.5 * dist_sq)
        weights = weights / weights.sum() * alpha

        return points, weights

    else:
        raise ValueError(f"Unsupported scheme for torch: {scheme}")


def sample_uniform_torch(
    center: torch.Tensor,
    axes: torch.Tensor,
    alpha: torch.Tensor,
    scheme: str = "7-point",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample uniform ellipsoid source in PyTorch.

    Args:
        center: [3] center position
        axes: [3] semi-axis lengths
        alpha: total intensity
        scheme: sampling scheme

    Returns:
        points: [N, 3] sample positions
        weights: [N] sample weights
    """
    device = center.device
    dtype = center.dtype

    if scheme == "1-point":
        points = center.unsqueeze(0)
        weights = alpha.unsqueeze(0)
        return points, weights

    elif scheme == "7-point":
        points_norm = torch.tensor(
            [
                [0, 0, 0],
                [0.5, 0, 0],
                [-0.5, 0, 0],
                [0, 0.5, 0],
                [0, -0.5, 0],
                [0, 0, 0.5],
                [0, 0, -0.5],
            ],
            device=device,
            dtype=dtype,
        )
        points = points_norm * axes + center
        weights = torch.ones(7, device=device, dtype=dtype) / 7.0 * alpha
        return points, weights

    elif scheme == "stratified-33":
        points_norm = torch.zeros((33, 3), device=device, dtype=dtype)
        points_norm[0] = torch.tensor([0, 0, 0], device=device, dtype=dtype)

        idx = 1
        for i in range(3):
            points_norm[idx, i] = 0.5
            idx += 1
            points_norm[idx, i] = -0.5
            idx += 1

        edge_r = 0.35
        for i in range(3):
            for j in range(i + 1, 3):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        points_norm[idx, i] = s1 * edge_r
                        points_norm[idx, j] = s2 * edge_r
                        idx += 1

        body_r = 0.25
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    points_norm[idx] = torch.tensor(
                        [s1 * body_r, s2 * body_r, s3 * body_r],
                        device=device,
                        dtype=dtype,
                    )
                    idx += 1

        outer_r = 0.7
        for i in range(3):
            points_norm[idx, i] = outer_r
            idx += 1
            points_norm[idx, i] = -outer_r
            idx += 1

        points = points_norm * axes + center

        weights = torch.ones(33, device=device, dtype=dtype)
        weights[0] = 0.06
        weights[1:7] = 0.035
        weights[7:19] = 0.025
        weights[19:27] = 0.018
        weights[27:33] = 0.012
        weights = weights / weights.sum() * alpha

        return points, weights

    elif scheme == "grid-27":
        # 3x3x3 grid with Gaussian-like weighting (center weighted)
        points_norm = torch.zeros((27, 3), device=device, dtype=dtype)
        weights = torch.ones(27, device=device, dtype=dtype)

        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.5  # -0.5, 0, 0.5
                    y = (j - 1) * 0.5
                    z = (k - 1) * 0.5
                    points_norm[idx] = torch.tensor(
                        [x, y, z], device=device, dtype=dtype
                    )
                    # Weight by distance from center (Gaussian-like)
                    dist_sq = x * x + y * y + z * z
                    weights[idx] = torch.exp(
                        torch.tensor(-dist_sq / 0.5, device=device, dtype=dtype)
                    )
                    idx += 1

        points = points_norm * axes + center
        weights = weights / weights.sum() * alpha

        return points, weights

    else:
        raise ValueError(f"Unsupported uniform scheme for torch: {scheme}")


def render_atlas_surface_torch(
    center: torch.Tensor,
    sigmas: torch.Tensor,
    alpha: torch.Tensor,
    surface_coords: torch.Tensor,
    surface_z_values: Optional[torch.Tensor],
    tissue_params: dict,
    sampling_scheme: str = "sr-6",
    kernel_type: str = "green_halfspace",
    geometry_mode: str = "local_depth",
    surface_normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render surface response in PyTorch (differentiable) for Gaussian source.

    Args:
        center: [3] source center
        sigmas: [3] source sigmas
        alpha: source intensity
        surface_coords: [N, 3] surface coordinates
        surface_z_values: [N] surface Z values (for local_depth)
        tissue_params: optical properties
        sampling_scheme: sampling scheme
        kernel_type: kernel type
        geometry_mode: "local_depth" or "local_plane"
        surface_normals: [N, 3] surface normals (for local_plane)

    Returns:
        response: [N] surface response
    """
    device = center.device
    dtype = center.dtype

    diff = diffusion_params_torch(tissue_params, device)
    D = diff["D"]
    mu_eff = diff["mu_eff"]
    zb = diff["zb"]

    source_points, source_weights = sample_gaussian_torch(
        center, sigmas, alpha, sampling_scheme
    )

    n_surf = surface_coords.shape[0]
    n_src = source_points.shape[0]

    if geometry_mode == "local_depth":
        if surface_z_values is not None:
            surface_z = surface_z_values
        else:
            surface_z = surface_coords[:, 2]

        depth = surface_z.unsqueeze(1) - source_points[:, 2].unsqueeze(0)

        dx = surface_coords[:, 0].unsqueeze(1) - source_points[:, 0].unsqueeze(0)
        dy = surface_coords[:, 1].unsqueeze(1) - source_points[:, 1].unsqueeze(0)

        rho_sq = dx**2 + dy**2
        rho = torch.sqrt(torch.clamp(rho_sq, min=1e-12))

    elif geometry_mode == "local_plane":
        if surface_normals is None:
            raise ValueError("surface_normals required for local_plane mode")

        delta = surface_coords.unsqueeze(1) - source_points.unsqueeze(0)

        normals_exp = surface_normals.unsqueeze(1)

        depth = torch.sum(delta * normals_exp, dim=2)

        parallel = delta - depth.unsqueeze(2) * normals_exp
        rho = torch.norm(parallel, dim=2)

    else:
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")

    r1_sq = rho**2 + depth**2
    r1 = torch.sqrt(torch.clamp(r1_sq, min=1e-12))

    if kernel_type == "green_infinite":
        G = torch.exp(-mu_eff * r1) / (4 * torch.pi * D * r1)

    elif kernel_type == "green_halfspace":
        r2_sq = rho**2 + (depth + 2 * zb) ** 2
        r2 = torch.sqrt(torch.clamp(r2_sq, min=1e-12))

        G1 = torch.exp(-mu_eff * r1) / (4 * torch.pi * D * r1)
        G2 = torch.exp(-mu_eff * r2) / (4 * torch.pi * D * r2)

        G = torch.clamp(G1 - G2, min=0.0)

    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")

    response = torch.sum(G * source_weights.unsqueeze(0), dim=1)

    return response


def render_atlas_surface_uniform_torch(
    center: torch.Tensor,
    axes: torch.Tensor,
    alpha: torch.Tensor,
    surface_coords: torch.Tensor,
    surface_z_values: Optional[torch.Tensor],
    tissue_params: dict,
    sampling_scheme: str = "7-point",
    kernel_type: str = "green_halfspace",
    geometry_mode: str = "local_depth",
    surface_normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Render surface response in PyTorch (differentiable) for uniform source.

    Args:
        center: [3] source center
        axes: [3] semi-axis lengths
        alpha: source intensity
        surface_coords: [N, 3] surface coordinates
        surface_z_values: [N] surface Z values (for local_depth)
        tissue_params: optical properties
        sampling_scheme: sampling scheme
        kernel_type: kernel type
        geometry_mode: "local_depth" or "local_plane"
        surface_normals: [N, 3] surface normals (for local_plane)

    Returns:
        response: [N] surface response
    """
    device = center.device
    dtype = center.dtype

    diff = diffusion_params_torch(tissue_params, device)
    D = diff["D"]
    mu_eff = diff["mu_eff"]
    zb = diff["zb"]

    source_points, source_weights = sample_uniform_torch(
        center, axes, alpha, sampling_scheme
    )

    if geometry_mode == "local_depth":
        if surface_z_values is not None:
            surface_z = surface_z_values
        else:
            surface_z = surface_coords[:, 2]

        depth = surface_z.unsqueeze(1) - source_points[:, 2].unsqueeze(0)

        dx = surface_coords[:, 0].unsqueeze(1) - source_points[:, 0].unsqueeze(0)
        dy = surface_coords[:, 1].unsqueeze(1) - source_points[:, 1].unsqueeze(0)

        rho_sq = dx**2 + dy**2
        rho = torch.sqrt(torch.clamp(rho_sq, min=1e-12))

    elif geometry_mode == "local_plane":
        if surface_normals is None:
            raise ValueError("surface_normals required for local_plane mode")

        delta = surface_coords.unsqueeze(1) - source_points.unsqueeze(0)
        normals_exp = surface_normals.unsqueeze(1)
        depth = torch.sum(delta * normals_exp, dim=2)
        parallel = delta - depth.unsqueeze(2) * normals_exp
        rho = torch.norm(parallel, dim=2)

    else:
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")

    r1_sq = rho**2 + depth**2
    r1 = torch.sqrt(torch.clamp(r1_sq, min=1e-12))

    if kernel_type == "green_infinite":
        G = torch.exp(-mu_eff * r1) / (4 * torch.pi * D * r1)

    elif kernel_type == "green_halfspace":
        r2_sq = rho**2 + (depth + 2 * zb) ** 2
        r2 = torch.sqrt(torch.clamp(r2_sq, min=1e-12))

        G1 = torch.exp(-mu_eff * r1) / (4 * torch.pi * D * r1)
        G2 = torch.exp(-mu_eff * r2) / (4 * torch.pi * D * r2)

        G = torch.clamp(G1 - G2, min=0.0)

    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")

    response = torch.sum(G * source_weights.unsqueeze(0), dim=1)

    return response


class DifferentiableAtlasForward(torch.nn.Module):
    """Differentiable forward model for atlas surface optimization."""

    def __init__(
        self,
        surface_coords: np.ndarray,
        surface_z_values: np.ndarray,
        tissue_params: dict,
        sampling_scheme: str = "sr-6",
        kernel_type: str = "green_halfspace",
        geometry_mode: str = "local_depth",
        device: torch.device = None,
    ):
        """Initialize forward model.

        Args:
            surface_coords: [N, 3] surface coordinates
            surface_z_values: [N] surface Z values
            tissue_params: optical properties
            sampling_scheme: sampling scheme
            kernel_type: kernel type
            geometry_mode: geometry mode
            device: torch device
        """
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.tissue_params = tissue_params
        self.sampling_scheme = sampling_scheme
        self.kernel_type = kernel_type
        self.geometry_mode = geometry_mode

        self.surface_coords = torch.tensor(
            surface_coords, dtype=torch.float32, device=device
        )
        self.surface_z_values = torch.tensor(
            surface_z_values, dtype=torch.float32, device=device
        )

    def forward(
        self,
        center: torch.Tensor,
        sigmas: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            center: [3] source center
            sigmas: [3] source sigmas
            alpha: source intensity

        Returns:
            response: [N] surface response
        """
        return render_atlas_surface_torch(
            center=center,
            sigmas=sigmas,
            alpha=alpha,
            surface_coords=self.surface_coords,
            surface_z_values=self.surface_z_values,
            tissue_params=self.tissue_params,
            sampling_scheme=self.sampling_scheme,
            kernel_type=self.kernel_type,
            geometry_mode=self.geometry_mode,
        )


class DifferentiableUniformAtlasForward(torch.nn.Module):
    """Differentiable forward model for atlas surface optimization with uniform source."""

    def __init__(
        self,
        surface_coords: np.ndarray,
        surface_z_values: np.ndarray,
        tissue_params: dict,
        sampling_scheme: str = "7-point",
        kernel_type: str = "green_halfspace",
        geometry_mode: str = "local_depth",
        device: torch.device = None,
    ):
        """Initialize forward model.

        Args:
            surface_coords: [N, 3] surface coordinates
            surface_z_values: [N] surface Z values
            tissue_params: optical properties
            sampling_scheme: sampling scheme
            kernel_type: kernel type
            geometry_mode: geometry mode
            device: torch device
        """
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.tissue_params = tissue_params
        self.sampling_scheme = sampling_scheme
        self.kernel_type = kernel_type
        self.geometry_mode = geometry_mode

        self.surface_coords = torch.tensor(
            surface_coords, dtype=torch.float32, device=device
        )
        self.surface_z_values = torch.tensor(
            surface_z_values, dtype=torch.float32, device=device
        )

    def forward(
        self,
        center: torch.Tensor,
        axes: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            center: [3] source center
            axes: [3] semi-axis lengths
            alpha: source intensity

        Returns:
            response: [N] surface response
        """
        return render_atlas_surface_uniform_torch(
            center=center,
            axes=axes,
            alpha=alpha,
            surface_coords=self.surface_coords,
            surface_z_values=self.surface_z_values,
            tissue_params=self.tissue_params,
            sampling_scheme=self.sampling_scheme,
            kernel_type=self.kernel_type,
            geometry_mode=self.geometry_mode,
        )


class DifferentiableGaussianSourceAtlas(torch.nn.Module):
    """Differentiable Gaussian source parameters for atlas optimization."""

    def __init__(
        self,
        center_init: np.ndarray,
        sigmas_init: tuple,
        alpha_init: float,
        device: torch.device = None,
    ):
        """Initialize source parameters.

        Args:
            center_init: initial center
            sigmas_init: initial sigmas
            alpha_init: initial alpha
            device: torch device
        """
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.center = torch.nn.Parameter(
            torch.tensor(center_init, dtype=torch.float32, device=device)
        )
        self.log_sigmas = torch.nn.Parameter(
            torch.log(torch.tensor(sigmas_init, dtype=torch.float32, device=device))
        )
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(alpha_init, dtype=torch.float32, device=device))
        )

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self.log_sigmas)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def get_params(self) -> dict:
        """Get current parameters as dict."""
        return {
            "center": self.center.detach().cpu().numpy().tolist(),
            "sigmas": self.sigmas.detach().cpu().numpy().tolist(),
            "alpha": float(self.alpha.detach().cpu().numpy()),
        }


class DifferentiableUniformSourceAtlas(torch.nn.Module):
    """Differentiable uniform ellipsoid source parameters for atlas optimization."""

    def __init__(
        self,
        center_init: np.ndarray,
        axes_init: tuple,
        alpha_init: float,
        device: torch.device = None,
    ):
        """Initialize source parameters.

        Args:
            center_init: initial center
            axes_init: initial axes
            alpha_init: initial alpha
            device: torch device
        """
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.center = torch.nn.Parameter(
            torch.tensor(center_init, dtype=torch.float32, device=device)
        )
        self.log_axes = torch.nn.Parameter(
            torch.log(torch.tensor(axes_init, dtype=torch.float32, device=device))
        )
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(alpha_init, dtype=torch.float32, device=device))
        )

    @property
    def axes(self) -> torch.Tensor:
        return torch.exp(self.log_axes)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def get_params(self) -> dict:
        """Get current parameters as dict."""
        return {
            "center": self.center.detach().cpu().numpy().tolist(),
            "axes": self.axes.detach().cpu().numpy().tolist(),
            "alpha": float(self.alpha.detach().cpu().numpy()),
        }
