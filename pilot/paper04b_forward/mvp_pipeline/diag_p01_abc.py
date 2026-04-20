"""D-P0.1.A/B/C: Three diagnostics for M4' inversion issue.

Optimized version with vectorized direct_path check.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
VOLUME_PATH = Path(
    "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")

AIR_LABEL = 0
SOFT_TISSUE_LABELS = {1}


def load_volume():
    return np.fromfile(VOLUME_PATH, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def is_direct_path_vertex_vectorized(
    source_pos_mm, vertices, volume_labels, voxel_size_mm, step_mm=0.1
):
    center = np.array(volume_labels.shape) / 2
    n_verts = len(vertices)
    is_direct = np.ones(n_verts, dtype=bool)

    for i in range(n_verts):
        vertex_pos = vertices[i]
        direction = vertex_pos - source_pos_mm
        distance = np.linalg.norm(direction)
        if distance < 0.01:
            continue

        direction = direction / distance
        n_steps = int(distance / step_mm)

        for j in range(1, n_steps + 1):
            pos_mm = source_pos_mm + j * step_mm * direction
            voxel = np.floor(pos_mm / voxel_size_mm + center).astype(int)

            if not (
                0 <= voxel[0] < volume_labels.shape[0]
                and 0 <= voxel[1] < volume_labels.shape[1]
                and 0 <= voxel[2] < volume_labels.shape[2]
            ):
                break

            label = volume_labels[voxel[0], voxel[1], voxel[2]]
            if label not in {AIR_LABEL} | SOFT_TISSUE_LABELS:
                is_direct[i] = False
                break

    return is_direct


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

    print("\nLoading volume...")
    volume = load_volume()
    print("Extracting surface vertices...")
    vertices = extract_surface_vertices(volume > 0, VOXEL_SIZE_MM)

    gt_pos = np.array([-0.6, 2.4, -3.8])
    print("Loading fluence...")
    fluence = np.load(ARCHIVE_BASE / "S2-Vol-P5-ventral-r2.0" / "fluence.npy")

    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices / VOXEL_SIZE_MM + center).astype(int)
    phi_mcx = np.zeros(len(vertices), dtype=np.float32)
    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if 0 <= vx < 95 and 0 <= vy < 100 and 0 <= vz < 52:
            phi_mcx[i] = fluence[vx, vy, vz]

    print("Computing direct vertices...")
    is_direct_geo = is_direct_path_vertex_vectorized(
        gt_pos, vertices, volume, VOXEL_SIZE_MM
    )

    loss_fn = make_loss_fn(vertices, phi_mcx, is_direct_geo, OPTICAL)

    print(f"\nSetup:")
    print(f"  GT position: {gt_pos}")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Direct vertices: {np.sum(is_direct_geo)}")
    print(f"  Valid vertices (direct & phi>0): {np.sum(is_direct_geo & (phi_mcx > 0))}")

    # D-P0.1.A: Loss landscape 1D slices
    print("\n" + "=" * 70)
    print("D-P0.1.A: Loss landscape 1D slices (GT ± 5mm, step 0.1mm)")
    print("=" * 70)

    axis_names = ["X", "Y", "Z"]
    for axis in range(3):
        offsets = np.linspace(-5, 5, 101)
        losses = []
        for dx in offsets:
            pos = gt_pos.copy()
            pos[axis] += dx
            losses.append(loss_fn(pos))

        losses = np.array(losses)
        min_idx = np.argmin(losses)
        min_offset = offsets[min_idx]
        min_loss = losses[min_idx]
        loss_at_gt = losses[50]

        loss_range = losses.max() - losses.min()
        loss_near_gt = losses[45:56].max() - losses[45:56].min()

        print(f"\n{axis_names[axis]}-axis slice:")
        print(f"  loss(GT) = {loss_at_gt:.4f}")
        print(f"  min loss = {min_loss:.4f} at offset = {min_offset:+.2f} mm")
        print(f"  loss range (±5mm) = {loss_range:.4f}")
        print(f"  loss range (±0.5mm around GT) = {loss_near_gt:.4f}")

        if abs(min_offset) < 0.5:
            print(f"  → Minimum near GT (within ±0.5mm)")
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

    # D-P0.1.C: Brute-force grid search
    print("\n" + "=" * 70)
    print("D-P0.1.C: Brute-force grid search (GT ± 3mm, step 0.2mm)")
    print("=" * 70)

    best = None
    n_eval = 0
    for x in np.arange(gt_pos[0] - 3, gt_pos[0] + 3.1, 0.2):
        for y in np.arange(gt_pos[1] - 3, gt_pos[1] + 3.1, 0.2):
            for z in np.arange(gt_pos[2] - 3, gt_pos[2] + 3.1, 0.2):
                l = loss_fn(np.array([x, y, z]))
                n_eval += 1
                if best is None or l < best[0]:
                    best = (l, np.array([x, y, z]))

    best_loss, best_pos = best
    err_to_gt = np.linalg.norm(best_pos - gt_pos)
    loss_at_gt = loss_fn(gt_pos)

    print(f"\nGrid search results ({n_eval} evaluations):")
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
