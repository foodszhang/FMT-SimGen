"""Investigate NCC discrepancy - why is NCC 0.65 instead of expected 0.94?"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import OPTICAL
from shared.green import G_inf

CACHE_DIR = Path("pilot/paper04b_forward/results/diag_cache")
GT_POS = np.array([-0.6, 2.4, -3.8])


def main():
    print("=" * 70)
    print("NCC Discrepancy Investigation")
    print("=" * 70)

    vertices = np.load(CACHE_DIR / "p5_ventral_vertices.npy")
    phi_mcx = np.load(CACHE_DIR / "p5_ventral_phi_mcx.npy")
    is_direct = np.load(CACHE_DIR / "p5_ventral_is_direct.npy")

    forward = G_inf(np.linalg.norm(vertices - GT_POS, axis=1), OPTICAL).astype(
        np.float32
    )

    valid = is_direct & (phi_mcx > 0) & (forward > 0)

    print(f"\nValid vertices: {np.sum(valid)}")

    phi_valid = phi_mcx[valid]
    fwd_valid = forward[valid]

    scale = np.sum(phi_valid) / np.sum(fwd_valid)
    fwd_scaled = scale * fwd_valid

    print(f"\nScale: {scale:.4e}")

    ncc_linear = np.corrcoef(phi_valid, fwd_scaled)[0, 1]
    print(f"\nNCC (linear scale): {ncc_linear:.4f}")

    log_phi = np.log10(phi_valid + 1e-20)
    log_fwd = np.log10(fwd_scaled + 1e-20)
    ncc_log = np.corrcoef(log_phi, log_fwd)[0, 1]
    print(f"NCC (log scale): {ncc_log:.4f}")

    rank_phi = np.argsort(np.argsort(phi_valid))
    rank_fwd = np.argsort(np.argsort(fwd_scaled))
    ncc_rank = np.corrcoef(rank_phi, rank_fwd)[0, 1]
    print(f"NCC (rank): {ncc_rank:.4f}")

    print("\n" + "=" * 70)
    print("Scatter plot analysis")
    print("=" * 70)

    n = len(phi_valid)
    sample_idx = np.random.choice(n, min(1000, n), replace=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(phi_valid[sample_idx], fwd_scaled[sample_idx], alpha=0.3, s=1)
    ax.set_xlabel("phi_mcx")
    ax.set_ylabel("forward_scaled")
    ax.set_title(f"Linear scale (NCC={ncc_linear:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1]
    ax.scatter(log_phi[sample_idx], log_fwd[sample_idx], alpha=0.3, s=1)
    ax.set_xlabel("log10(phi_mcx)")
    ax.set_ylabel("log10(forward_scaled)")
    ax.set_title(f"Log scale (NCC={ncc_log:.3f})")

    ax = axes[2]
    dist_valid = np.linalg.norm(vertices[valid] - GT_POS, axis=1)
    ax.scatter(
        dist_valid[sample_idx], phi_valid[sample_idx], alpha=0.3, s=1, label="MCX"
    )
    ax.scatter(
        dist_valid[sample_idx], fwd_scaled[sample_idx], alpha=0.3, s=1, label="G_inf"
    )
    ax.set_xlabel("Distance from source (mm)")
    ax.set_ylabel("Fluence")
    ax.set_title("Distance vs Fluence")
    ax.set_yscale("log")
    ax.legend()

    plt.tight_layout()
    output_path = Path(
        "pilot/paper04b_forward/results/forward_audit/ncc_investigation.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    print("\n" + "=" * 70)
    print("Distance bin analysis")
    print("=" * 70)

    bins = [0, 2, 5, 10, 15, 20, 30]
    for i in range(len(bins) - 1):
        mask = (dist_valid >= bins[i]) & (dist_valid < bins[i + 1])
        if np.sum(mask) > 10:
            ncc_bin = np.corrcoef(log_phi[mask], log_fwd[mask])[0, 1]
            print(
                f"  d ∈ [{bins[i]:.0f}, {bins[i + 1]:.0f}) mm: N={np.sum(mask)}, NCC_log={ncc_bin:.3f}"
            )


if __name__ == "__main__":
    main()
