"""Run D2.1 vertex-log-NCC for all 5 positions to verify §4.C values."""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
VOLUME_PATH = (
    REPO_ROOT
    / "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2/mcx_volume_downsampled_2x.bin"
)
ARCHIVE_BASE = (
    REPO_ROOT / "pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2"
)

POSITIONS = {
    "P1-dorsal": {"pos_mm": [-0.6, 2.4, 5.8]},
    "P2-left": {"pos_mm": [-8.0, 2.4, 1.0]},
    "P3-right": {"pos_mm": [6.8, 2.4, 1.0]},
    "P4-dorsal-lat": {"pos_mm": [-6.3, 2.4, 5.8]},
    "P5-ventral": {"pos_mm": [-0.6, 2.4, -3.8]},
}

PAPER_VALUES = {
    "P1-dorsal": 0.9365,
    "P2-left": 0.9498,
    "P3-right": 0.9429,
    "P4-dorsal-lat": 0.9834,
    "P5-ventral": 0.9578,
}

SOFT_TISSUE_LABEL = 1
AIR_LABEL = 0


def load_volume():
    volume = np.fromfile(VOLUME_PATH, dtype=np.uint8)
    return volume.reshape(VOLUME_SHAPE_XYZ)


def extract_surface_vertices(binary_mask, voxel_size_mm):
    from skimage import measure

    verts, _, _, _ = measure.marching_cubes(
        binary_mask.astype(float), level=0.5, spacing=(voxel_size_mm,) * 3
    )
    center = np.array(binary_mask.shape) / 2 * voxel_size_mm
    return verts - center


def is_direct_path_vertex(source_pos_mm, vertex_pos_mm, volume_labels, voxel_size_mm):
    center = np.array(volume_labels.shape) / 2

    def mm_to_voxel(mm):
        return np.floor(mm / voxel_size_mm + center).astype(int)

    direction = vertex_pos_mm - source_pos_mm
    distance = np.linalg.norm(direction)
    if distance < 0.01:
        return True

    direction = direction / distance
    step_mm = 0.1
    n_steps = int(distance / step_mm)

    for i in range(1, min(n_steps, 500) + 1):
        pos_mm = source_pos_mm + i * step_mm * direction
        voxel = mm_to_voxel(pos_mm)

        if not (
            0 <= voxel[0] < volume_labels.shape[0]
            and 0 <= voxel[1] < volume_labels.shape[1]
            and 0 <= voxel[2] < volume_labels.shape[2]
        ):
            break

        label = volume_labels[voxel[0], voxel[1], voxel[2]]
        if label not in {AIR_LABEL, SOFT_TISSUE_LABEL}:
            return False

    return True


def sample_fluence_at_vertices(fluence, vertices_mm, voxel_size_mm):
    center = np.array(fluence.shape) / 2
    verts_voxel = np.floor(vertices_mm / voxel_size_mm + center).astype(int)

    phi = np.zeros(len(vertices_mm), dtype=np.float32)
    valid = np.zeros(len(vertices_mm), dtype=bool)

    for i, (vx, vy, vz) in enumerate(verts_voxel):
        if (
            0 <= vx < fluence.shape[0]
            and 0 <= vy < fluence.shape[1]
            and 0 <= vz < fluence.shape[2]
        ):
            phi[i] = fluence[vx, vy, vz]
            valid[i] = True

    return phi, valid


def compute_metrics(phi_mcx, phi_closed, mask):
    valid = mask & (phi_mcx > 0) & (phi_closed > 0)
    if np.sum(valid) < 10:
        return {"ncc": 0.0, "n_valid": 0}

    mcx_vals = phi_mcx[valid]
    closed_vals = phi_closed[valid]

    log_mcx = np.log10(mcx_vals + 1e-20)
    log_closed = np.log10(closed_vals + 1e-20)
    ncc = np.corrcoef(log_mcx, log_closed)[0, 1]

    return {"ncc": float(ncc), "n_valid": int(np.sum(valid))}


def main():
    from shared.config import OPTICAL
    from shared.green import G_inf

    print("=" * 70)
    print("D2.1 Vertex-Log-NCC Verification for §4.C")
    print("=" * 70)
    print(f"Script: {Path(__file__).name}")
    print(f"Commit: (run git log -1 --oneline)")
    print("=" * 70)

    volume = load_volume()
    binary_mask = volume > 0

    print("Extracting surface vertices...")
    vertices_mm = extract_surface_vertices(binary_mask, VOXEL_SIZE_MM)
    n_vertices = len(vertices_mm)
    print(f"Total vertices: {n_vertices}")

    results = {}

    for pos_name, pos_info in POSITIONS.items():
        print(f"\n{'=' * 60}")
        print(f"Position: {pos_name}")
        print(f"{'=' * 60}")

        source_pos_mm = np.array(pos_info["pos_mm"])
        fluence_path = ARCHIVE_BASE / f"S2-Vol-{pos_name}-r2.0" / "fluence.npy"

        print(f"Source: {source_pos_mm}")
        print(f"Fluence: {fluence_path}")

        fluence = np.load(fluence_path)
        phi_mcx, valid_in_bounds = sample_fluence_at_vertices(
            fluence, vertices_mm, VOXEL_SIZE_MM
        )

        dx = vertices_mm[:, 0] - source_pos_mm[0]
        dy = vertices_mm[:, 1] - source_pos_mm[1]
        dz = vertices_mm[:, 2] - source_pos_mm[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 0.01)
        phi_closed = G_inf(r, OPTICAL).astype(np.float32)

        print("Computing direct-path vertices...")
        is_direct = np.zeros(n_vertices, dtype=bool)
        for i in range(n_vertices):
            is_direct[i] = is_direct_path_vertex(
                source_pos_mm, vertices_mm[i], volume, VOXEL_SIZE_MM
            )

        n_direct = np.sum(is_direct)
        print(
            f"Direct-path: {n_direct} / {n_vertices} ({100 * n_direct / n_vertices:.1f}%)"
        )

        metrics_direct = compute_metrics(
            phi_mcx, phi_closed, valid_in_bounds & is_direct
        )

        paper_val = PAPER_VALUES[pos_name]
        diff = metrics_direct["ncc"] - paper_val

        results[pos_name] = {
            "source_pos_mm": source_pos_mm.tolist(),
            "n_direct": int(n_direct),
            "n_valid": metrics_direct["n_valid"],
            "ncc": metrics_direct["ncc"],
            "paper_value": paper_val,
            "diff": diff,
        }

        status = "PASS" if abs(diff) <= 0.01 else "FAIL"
        print(
            f"NCC: {metrics_direct['ncc']:.4f} (paper: {paper_val:.4f}, diff: {diff:+.4f}) [{status}]"
        )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Position':<15} {'Paper':<10} {'Repro':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)

    all_pass = True
    for pos_name in POSITIONS:
        r = results[pos_name]
        status = "PASS" if abs(r["diff"]) <= 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"{pos_name:<15} {r['paper_value']:<10.4f} {r['ncc']:<10.4f} {r['diff']:+<10.4f} {status}"
        )

    print("=" * 70)
    if all_pass:
        print("ALL PASS: §4.C verified within ±0.01")
    else:
        print("FAIL: Some positions outside ±0.01 tolerance")

    output_dir = Path("pilot/paper04b_forward/results/d2_1_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "verification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import csv

    with open(output_dir / "verification_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Position", "Paper_Value", "Reproduced_Value", "Diff", "Status"]
        )
        for pos_name in POSITIONS:
            r = results[pos_name]
            status = "PASS" if abs(r["diff"]) <= 0.01 else "FAIL"
            writer.writerow(
                [
                    pos_name,
                    f"{r['paper_value']:.4f}",
                    f"{r['ncc']:.4f}",
                    f"{r['diff']:+.4f}",
                    status,
                ]
            )

    print(f"\nOutput: {output_dir / 'verification_table.csv'}")


if __name__ == "__main__":
    main()
