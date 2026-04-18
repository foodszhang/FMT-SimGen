"""Closed-form forward models for three source types.

Provides unified interface for:
- Point source: infinite medium Green's function
- Ball source: angle-averaged Green's function integral
- Gaussian source: analytic slice + 1D quadrature
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

if TYPE_CHECKING:
    from .config import OpticalParams
    from .sources import SourceSpec

from .green import G_inf, G_bar_angle_averaged

logger = logging.getLogger(__name__)


def closed_form_point_on_surface(
    source_pos_mm: np.ndarray,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    optical: OpticalParams,
) -> np.ndarray:
    dx = surface_coords[:, :, 0] - source_pos_mm[0]
    dy = surface_coords[:, :, 1] - source_pos_mm[1]
    dz = surface_coords[:, :, 2] - source_pos_mm[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r = np.maximum(r, 0.01)
    G = G_inf(r, optical)
    projection = np.zeros_like(G, dtype=np.float32)
    projection[valid_mask] = G[valid_mask].astype(np.float32)
    return projection


def closed_form_ball_on_surface(
    source_pos_mm: np.ndarray,
    radius_mm: float,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    optical: OpticalParams,
    n_quad: int = 10,
) -> np.ndarray:
    H, W = surface_coords.shape[:2]
    dx = surface_coords[:, :, 0] - source_pos_mm[0]
    dy = surface_coords[:, :, 1] - source_pos_mm[1]
    dz = surface_coords[:, :, 2] - source_pos_mm[2]
    d_arr = np.sqrt(dx**2 + dy**2 + dz**2)
    nodes, weights = leggauss(n_quad)
    rp = 0.5 * radius_mm * (nodes + 1.0)
    wp = 0.5 * radius_mm * weights
    ball_volume = (4.0 / 3.0) * np.pi * radius_mm**3
    projection = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue
            d = d_arr[i, j]
            if d < 0.01:
                continue
            Gbar = G_bar_angle_averaged(d, rp, optical)
            integrand = 4 * np.pi * rp**2 * Gbar
            I_unnorm = np.sum(wp * integrand)
            projection[i, j] = I_unnorm / ball_volume
    return projection


def closed_form_gaussian_on_surface(
    source_pos_mm: np.ndarray,
    sigma_mm: np.ndarray,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    optical: OpticalParams,
    n_z_quad: int = 64,
) -> np.ndarray:
    H, W = surface_coords.shape[:2]
    sigma_iso = float(np.mean(sigma_mm))
    dx = surface_coords[:, :, 0] - source_pos_mm[0]
    dy = surface_coords[:, :, 1] - source_pos_mm[1]
    dz = surface_coords[:, :, 2] - source_pos_mm[2]
    rho_xy = np.sqrt(dx**2 + dy**2)
    z_offset = dz
    z_max = 5 * sigma_iso
    nodes, weights = leggauss(n_z_quad)
    zs = 0.5 * z_max * (nodes + 1.0)
    zs = np.concatenate([-zs[::-1], zs])
    ws = 0.5 * z_max * np.concatenate([weights[::-1], weights])
    projection = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            if not valid_mask[i, j]:
                continue
            d_rho = rho_xy[i, j]
            z_src = z_offset[i, j]
            total = 0.0
            for zi, wi in zip(zs, ws):
                r = np.sqrt(d_rho**2 + (z_src - zi) ** 2)
                if r < 0.01:
                    continue
                G = G_inf(r, optical)
                amp = np.exp(-(zi**2) / (2 * sigma_iso**2)) / np.sqrt(
                    2 * np.pi * sigma_iso**2
                )
                total += wi * amp * G
            projection[i, j] = total
    return projection


def forward_closed_source(
    source: SourceSpec,
    surface_coords: np.ndarray,
    valid_mask: np.ndarray,
    optical: OpticalParams,
) -> np.ndarray:
    if source.kind == "point":
        return closed_form_point_on_surface(
            source.center_mm, surface_coords, valid_mask, optical
        )
    elif source.kind == "ball":
        return closed_form_ball_on_surface(
            source.center_mm,
            source.radius_mm,
            surface_coords,
            valid_mask,
            optical,
        )
    elif source.kind == "gaussian":
        return closed_form_gaussian_on_surface(
            source.center_mm,
            source.sigma_mm,
            surface_coords,
            valid_mask,
            optical,
        )
    else:
        raise ValueError(f"Unknown source kind: {source.kind}")
