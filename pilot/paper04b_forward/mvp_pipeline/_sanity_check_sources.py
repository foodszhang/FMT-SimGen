"""Sanity check: verify three source types produce different fluence values."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf, G_bar_angle_averaged

source_pos = np.array([0.0, 0.0, 0.0])
radius = 2.0
sigma = 1.0

obs_point = np.array([5.0, 0.0, 0.0])

d = np.linalg.norm(obs_point - source_pos)
phi_point = G_inf(d, OPTICAL)

n_quad = 10
from numpy.polynomial.legendre import leggauss

nodes, weights = leggauss(n_quad)
rp = 0.5 * radius * (nodes + 1.0)
wp = 0.5 * radius * weights

Gbar = G_bar_angle_averaged(d, rp, OPTICAL)
ball_volume = (4.0 / 3.0) * np.pi * radius**3
integrand = 4 * np.pi * rp**2 * Gbar
phi_ball = np.sum(wp * integrand) / ball_volume

n_z_quad = 64
z_max = 5 * sigma
nodes_z, weights_z = leggauss(n_z_quad)
zs = 0.5 * z_max * (nodes_z + 1.0)
zs = np.concatenate([-zs[::-1], zs])
ws = 0.5 * z_max * np.concatenate([weights_z[::-1], weights_z])

phi_gauss = 0.0
for zi, wi in zip(zs, ws):
    src_pt = np.array([0, 0, zi])
    r = np.linalg.norm(obs_point - src_pt)
    if r > 0.01:
        G = G_inf(r, OPTICAL)
        amp = np.exp(-(zi**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        phi_gauss += wi * amp * G

print("=" * 60)
print("Sanity Check: Three Source Types")
print("=" * 60)
print(f"Observation point: 5mm from source")
print(f"Ball radius: {radius}mm, Gaussian sigma: {sigma}mm")
print("-" * 60)
print(f"Point:    {phi_point:.6e}")
print(f"Ball:     {phi_ball:.6e}")
print(f"Gaussian: {phi_gauss:.6e}")
print("-" * 60)
print(f"Point/Ball ratio:     {phi_point / phi_ball:.6f}")
print(f"Point/Gaussian ratio: {phi_point / phi_gauss:.6f}")
print(f"Ball/Gaussian ratio:  {phi_ball / phi_gauss:.6f}")
print("=" * 60)

if np.allclose([phi_point], [phi_ball], rtol=1e-4):
    print("FAIL: Point and Ball collapsed!")
    sys.exit(1)
elif np.allclose([phi_ball], [phi_gauss], rtol=1e-4):
    print("FAIL: Ball and Gaussian collapsed!")
    sys.exit(1)
else:
    print("PASS: Three sources produce distinct values ✓")
