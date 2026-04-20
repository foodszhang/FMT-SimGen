"""Final sanity check: Are organ-labeled vertices truly on outer surface?

Check: For each organ vertex, does its neighborhood contain air (label=0)?
- If YES → vertex is on outer skin surface (air outside, organ inside)
- If NO → vertex might be on internal organ surface (BUG)
"""

import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

VOXEL_SIZE_MM = 0.4
VOLUME_SHAPE_XYZ = (95, 100, 52)
ARCHIVE_BASE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/geometry_audit")


def load_volume():
    volume_path = ARCHIVE_BASE / "mcx_volume_downsampled_2x.bin"
    return np.fromfile(volume_path, dtype=np.uint8).reshape(VOLUME_SHAPE_XYZ)


def main():
    print("=" * 70)
    print("FINAL SANITY CHECK: Air Neighbor Test")
    print("=" * 70)

    atlas = load_volume()
    body_mask = (atlas > 0).astype(np.uint8)

    verts, _, _, _ = measure.marching_cubes(
        body_mask, level=0.5, spacing=(VOXEL_SIZE_MM,) * 3
    )
    center = np.array(atlas.shape) / 2 * VOXEL_SIZE_MM
    vertices = verts - center

    center_idx = np.array(atlas.shape) / 2
    v_vox = np.floor(vertices / VOXEL_SIZE_MM + center_idx).astype(int)
    v_vox = np.clip(v_vox, 0, np.array(atlas.shape) - 1)

    labels_at_v = atlas[v_vox[:, 0], v_vox[:, 1], v_vox[:, 2]]

    organ_mask = ~np.isin(labels_at_v, [0, 1])
    organ_verts_vox = v_vox[organ_mask]

    print(f"\nTotal vertices: {len(vertices)}")
    print(
        f"Organ-labeled vertices: {np.sum(organ_mask)} ({100 * np.sum(organ_mask) / len(vertices):.1f}%)"
    )

    print("\nChecking air neighbors for each organ vertex...")
    has_air_neighbor = []
    has_soft_tissue_neighbor = []

    for v in organ_verts_vox:
        x0, x1 = max(0, v[0] - 1), min(atlas.shape[0], v[0] + 2)
        y0, y1 = max(0, v[1] - 1), min(atlas.shape[1], v[1] + 2)
        z0, z1 = max(0, v[2] - 1), min(atlas.shape[2], v[2] + 2)
        neigh = atlas[x0:x1, y0:y1, z0:z1]
        has_air_neighbor.append(0 in neigh)
        has_soft_tissue_neighbor.append(1 in neigh)

    has_air_neighbor = np.array(has_air_neighbor)
    has_soft_tissue_neighbor = np.array(has_soft_tissue_neighbor)

    n_with_air = np.sum(has_air_neighbor)
    n_without_air = np.sum(~has_air_neighbor)
    n_with_soft = np.sum(has_soft_tissue_neighbor)

    print(f"\nResults:")
    print(
        f"  Organ vertices with air neighbor (TRUE OUTER SURFACE): {n_with_air} ({100 * n_with_air / len(has_air_neighbor):.1f}%)"
    )
    print(
        f"  Organ vertices without air neighbor (SUSPECT INTERNAL): {n_without_air} ({100 * n_without_air / len(has_air_neighbor):.1f}%)"
    )
    print(
        f"  Organ vertices with soft tissue neighbor: {n_with_soft} ({100 * n_with_soft / len(has_soft_tissue_neighbor):.1f}%)"
    )

    pct_with_air = 100 * n_with_air / len(has_air_neighbor)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if pct_with_air >= 95:
        print(f"\n✅ PASS: {pct_with_air:.1f}% organ vertices have air neighbor")
        print("   → Confirmed: These are OUTER SURFACE vertices")
        print("   → G-3/G-4 failures are FALSE ALARMS")
        print("   → Proceed to A-1~A-7 forward audit")
        passed = True
    else:
        print(f"\n❌ FAIL: Only {pct_with_air:.1f}% organ vertices have air neighbor")
        print("   → {n_without_air} vertices might be on INTERNAL organ surfaces")
        print("   → STOP and investigate mesh extraction")
        passed = False

    with open(OUTPUT_DIR / "air_neighbor_check.txt", "w") as f:
        f.write(f"Organ vertices: {np.sum(organ_mask)}\n")
        f.write(
            f"With air neighbor (outer surface): {n_with_air} ({pct_with_air:.1f}%)\n"
        )
        f.write(
            f"Without air neighbor (suspect): {n_without_air} ({100 * n_without_air / len(has_air_neighbor):.1f}%)\n"
        )
        f.write(f"PASS: {passed}\n")

    return {
        "organ_vertices": int(np.sum(organ_mask)),
        "with_air_neighbor": int(n_with_air),
        "without_air_neighbor": int(n_without_air),
        "pct_with_air": pct_with_air,
        "passed": passed,
    }


if __name__ == "__main__":
    main()
