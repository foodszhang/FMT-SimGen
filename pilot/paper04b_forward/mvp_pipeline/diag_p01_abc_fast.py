"""D-P0.1.A/B/C: Three diagnostics using cached P5-ventral data.

Optimized for speed.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf

CACHE_DIR = Path("pilot/paper04b_forward/results/diag_cache")


def make_loss_fn(vertices, phi_mcx, direct_mask, optical):
    def loss(params):
        source_pos = params[:3]
        r = np.linalg.norm(vertices - source_pos, axis=1)
        forward = G_inf(np.maximum(r, 0.01), optical).astype(np.float32)
        valid = direct_mask & (phi_mcx > 0) & (forward > 0)
        if np.sum(valid) < 50:
            return 1e10
        scale = float(np.sum(phi_mcx[valid]) / np.sum(forward[valid]))
        log_meas = np.log10(phi_mcx[valid] + 1e-20)
        log_fwd = np.log10(scale * forward[valid] + 1e-20)
        return float(np.mean((log_meas - log_fwd) ** 2))

    return loss


def main():
    print("=" * 70)
    print("D-P0.1.A/B/C: Three Diagnostics for P5-ventral")
    print("=" * 70)

    print("\nLoading cached data...")
    vertices = np.load(CACHE_DIR / "p5_ventral_vertices.npy")
    phi_mcx = np.load(CACHE_DIR / "p5_ventral_phi_mcx.npy")
    is_direct_geo = np.load(CACHE_DIR / "p5_ventral_is_direct.npy")
    gt_pos = np.load(CACHE_DIR / "p5_ventral_gt_pos.npy")

    loss_fn = make_loss_fn(vertices, phi_mcx, is_direct_geo, OPTICAL)

    print(f"\nSetup:")
    print(f"  GT position: {gt_pos}")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Direct vertices: {np.sum(is_direct_geo)}")
    print(f"  Valid vertices (direct & phi>0): {np.sum(is_direct_geo & (phi_mcx > 0))}")

    # D-P0.1.A: Loss landscape 1D slices
    print("\n" + "=" * 70)
    print("D-P0.1.A: Loss landscape 1D slices (GT ± 5mm, step 0.2mm)")
    print("=" * 70)

    axis_names = ["X", "Y", "Z"]
    for axis in range(3):
        offsets = np.linspace(-5, 5, 51)
        losses = []
        for dx in offsets:
            pos = gt_pos.copy()
            pos[axis] += dx
            losses.append(loss_fn(pos))

        losses = np.array(losses)
        min_idx = np.argmin(losses)
        min_offset = offsets[min_idx]
        min_loss = losses[min_idx]
        loss_at_gt = losses[25]

        loss_range = losses.max() - losses.min()
        loss_near_gt = losses[23:28].max() - losses[23:28].min()

        print(f"\n{axis_names[axis]}-axis slice:")
        print(f"  loss(GT) = {loss_at_gt:.4f}")
        print(f"  min loss = {min_loss:.4f} at offset = {min_offset:+.2f} mm")
        print(f"  loss range (±5mm) = {loss_range:.4f}")
        print(f"  loss range (±0.4mm around GT) = {loss_near_gt:.4f}")

        if abs(min_offset) < 0.4:
            print(f"  → Minimum near GT (within ±0.4mm)")
        elif abs(min_offset) < 2.0:
            print(f"  → Minimum shifted from GT by {abs(min_offset):.2f}mm")
        else:
            print(f"  → Minimum far from GT ({abs(min_offset):.2f}mm)")

        if loss_near_gt < 0.01:
            print(f"  → FLAT near GT (variation < 0.01)")
        elif loss_near_gt < 0.1:
            print(f"  → Shallow near GT (variation < 0.1)")
        else:
            print(f"  → Has curvature near GT")

    # D-P0.1.B: Gradient at init positions
    print("\n" + "=" * 70)
    print("D-P0.1.B: Gradient magnitude at init positions (sigma=1.0)")
    print("=" * 70)

    print(
        f"\n{'seed':>4} {'init_err_mm':>12} {'loss(init)':>12} {'|grad|':>12} {'status':>20}"
    )
    print("-" * 70)

    for seed in range(5):
        np.random.seed(seed)
        init = gt_pos + np.random.randn(3) * 1.0
        init_err = np.linalg.norm(init - gt_pos)
        loss_val = loss_fn(init)
        grad = approx_fprime(init, loss_fn, epsilon=0.01)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < 1e-5:
            status = "FLAT (grad < 1e-5)"
        elif grad_norm < 1e-3:
            status = "SHALLOW (grad < 1e-3)"
        elif grad_norm < 1e-1:
            status = "MODERATE"
        else:
            status = "STRONG"

        print(
            f"{seed:>4} {init_err:>12.3f} {loss_val:>12.4f} {grad_norm:>12.6f} {status:>20}"
        )

    # D-P0.1.C: Brute-force grid search (coarse then fine)
    print("\n" + "=" * 70)
    print("D-P0.1.C: Brute-force grid search")
    print("=" * 70)

    loss_at_gt = loss_fn(gt_pos)
    print(f"\nloss(GT) = {loss_at_gt:.4f}")

    # Coarse search (step 0.5mm)
    print("\nCoarse search (GT ± 3mm, step 0.5mm)...")
    best_coarse = None
    n_eval = 0
    for x in np.arange(gt_pos[0] - 3, gt_pos[0] + 3.1, 0.5):
        for y in np.arange(gt_pos[1] - 3, gt_pos[1] + 3.1, 0.5):
            for z in np.arange(gt_pos[2] - 3, gt_pos[2] + 3.1, 0.5):
                l = loss_fn(np.array([x, y, z]))
                n_eval += 1
                if best_coarse is None or l < best_coarse[0]:
                    best_coarse = (l, np.array([x, y, z]))

    print(f"  {n_eval} evaluations")
    print(f"  coarse best: {best_coarse[1]}, loss={best_coarse[0]:.4f}")

    # Fine search around coarse best
    print("\nFine search (coarse_best ± 1mm, step 0.2mm)...")
    best_fine = best_coarse
    coarse_pos = best_coarse[1]
    n_eval2 = 0
    for x in np.arange(coarse_pos[0] - 1, coarse_pos[0] + 1.1, 0.2):
        for y in np.arange(coarse_pos[1] - 1, coarse_pos[1] + 1.1, 0.2):
            for z in np.arange(coarse_pos[2] - 1, coarse_pos[2] + 1.1, 0.2):
                l = loss_fn(np.array([x, y, z]))
                n_eval2 += 1
                if l < best_fine[0]:
                    best_fine = (l, np.array([x, y, z]))

    print(f"  {n_eval2} evaluations")

    best_loss, best_pos = best_fine
    err_to_gt = np.linalg.norm(best_pos - gt_pos)

    print(f"\nGrid search results (total {n_eval + n_eval2} evaluations):")
    print(f"  best_pos = [{best_pos[0]:+.2f}, {best_pos[1]:+.2f}, {best_pos[2]:+.2f}]")
    print(f"  loss(best) = {best_loss:.4f}")
    print(f"  loss(GT) = {loss_at_gt:.4f}")
    print(f"  |best - GT| = {err_to_gt:.2f} mm")

    if err_to_gt < 0.5:
        print(f"\n  → CONCLUSION: Minimum is AT GT (within 0.5mm)")
        print(f"     Problem is optimizer, not loss landscape")
    elif err_to_gt < 2.0:
        print(f"\n  → CONCLUSION: Minimum is NEAR GT (within 2mm)")
        print(f"     Model mismatch is moderate")
    else:
        print(f"\n  → CONCLUSION: Minimum is FAR from GT ({err_to_gt:.2f}mm)")
        print(f"     Model mismatch is severe")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Loss at GT: {loss_at_gt:.4f}")
    print(f"Loss at best (grid): {best_loss:.4f}")
    print(
        f"Loss improvement: {loss_at_gt - best_loss:.4f} ({(loss_at_gt - best_loss) / loss_at_gt * 100:.1f}%)"
    )
    print(f"Position error to GT: {err_to_gt:.2f} mm")


if __name__ == "__main__":
    main()
