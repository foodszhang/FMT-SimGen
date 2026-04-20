"""Deeper investigation: Why do body-mask vertices have organ labels?

The marching_cubes vertices are at fractional positions on the air/body boundary.
When we round to integer voxel indices, we might hit organ voxels.

This script checks:
1. What happens if we check the OUTSIDE voxel (air side) instead of inside
2. What happens if we check both sides
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
    print("Deep Dive: Vertex Label Lookup Analysis")
    print("=" * 70)

    atlas = load_volume()
    body_mask = (atlas > 0).astype(np.uint8)

    verts, faces, _, _ = measure.marching_cubes(
        body_mask, level=0.5, spacing=(VOXEL_SIZE_MM,) * 3
    )
    center = np.array(atlas.shape) / 2 * VOXEL_SIZE_MM
    vertices = verts - center

    print(f"\nTotal vertices: {len(vertices)}")

    center_idx = np.array(atlas.shape) / 2

    v_vox_float = vertices / VOXEL_SIZE_MM + center_idx
    v_vox_floor = np.floor(v_vox_float).astype(int)
    v_vox_ceil = np.ceil(v_vox_float).astype(int)
    v_vox_round = np.round(v_vox_float).astype(int)

    def count_labels(v_vox, name):
        v_vox_clipped = np.clip(v_vox, 0, np.array(atlas.shape) - 1)
        labels = atlas[v_vox_clipped[:, 0], v_vox_clipped[:, 1], v_vox_clipped[:, 2]]
        air_soft = np.sum(np.isin(labels, [0, 1]))
        organ = np.sum(labels >= 2)
        print(f"\n  {name}:")
        print(f"    Air/Soft tissue: {air_soft} ({100 * air_soft / len(labels):.1f}%)")
        print(f"    Organ (≥2):      {organ} ({100 * organ / len(labels):.1f}%)")
        return air_soft, organ

    count_labels(v_vox_floor, "FLOOR (inside voxel)")
    count_labels(v_vox_ceil, "CEIL (outside voxel)")
    count_labels(v_vox_round, "ROUND (nearest voxel)")

    print("\n" + "=" * 70)
    print("Analysis: Why do surface vertices have organ labels?")
    print("=" * 70)

    v_vox_clipped = np.clip(v_vox_round, 0, np.array(atlas.shape) - 1)
    labels_round = atlas[v_vox_clipped[:, 0], v_vox_clipped[:, 1], v_vox_clipped[:, 2]]

    organ_mask = labels_round >= 2
    organ_vertices = vertices[organ_mask]
    organ_voxels = v_vox_round[organ_mask]

    print(f"\nOrgan-labeled vertices: {np.sum(organ_mask)}")
    print(f"\nSample organ vertex positions (first 10):")
    for i in range(min(10, len(organ_voxels))):
        vx, vy, vz = organ_voxels[i]
        if (
            0 <= vx < atlas.shape[0]
            and 0 <= vy < atlas.shape[1]
            and 0 <= vz < atlas.shape[2]
        ):
            label = atlas[int(vx), int(vy), int(vz)]
            print(f"  voxel ({vx:.0f}, {vy:.0f}, {vz:.0f}): label={label}")

    print("\n" + "=" * 70)
    print("Hypothesis Check: Are organ vertices on organ SURFACE?")
    print("=" * 70)

    organ_surface_count = 0
    for i in range(len(organ_voxels)):
        vx, vy, vz = organ_voxels[i].astype(int)
        if (
            0 <= vx < atlas.shape[0]
            and 0 <= vy < atlas.shape[1]
            and 0 <= vz < atlas.shape[2]
        ):
            label_center = atlas[vx, vy, vz]
            if label_center >= 2:
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nx, ny, nz = vx + dx, vy + dy, vz + dz
                            if (
                                0 <= nx < atlas.shape[0]
                                and 0 <= ny < atlas.shape[1]
                                and 0 <= nz < atlas.shape[2]
                            ):
                                neighbors.append(atlas[nx, ny, nz])
                if 1 in neighbors or 0 in neighbors:
                    organ_surface_count += 1

    print(f"\nOrgan vertices adjacent to soft tissue/air: {organ_surface_count}")
    print(
        f"This is {100 * organ_surface_count / np.sum(organ_mask):.1f}% of organ vertices"
    )

    if organ_surface_count > 0.5 * np.sum(organ_mask):
        print(
            "\n→ CONCLUSION: These vertices are on ORGAN SURFACES (internal boundaries)"
        )
        print("  The marching_cubes on body_mask extracts the OUTER surface only,")
        print("  but the vertex lookup is hitting organ voxels due to:")
        print("  1. Vertex positions are at fractional coordinates")
        print("  2. Rounding to integer voxels can land on either side of boundary")
        print("  3. Near organ surfaces, rounding can hit organ voxels")
    else:
        print("\n→ Organ vertices are NOT on organ surfaces - deeper issue")

    print("\n" + "=" * 70)
    print("PROPOSED FIX")
    print("=" * 70)
    print("""
The issue is that vertices on the body surface can round to organ voxels
when the body surface is adjacent to an organ.

SOLUTION: When looking up labels at vertices, check if the vertex is
actually on the air/body boundary, not inside an organ:

1. For each vertex, check the 3x3x3 neighborhood
2. If the neighborhood contains both air (0) and soft tissue (1), 
   the vertex is on the TRUE outer surface
3. If the neighborhood contains organ labels but no air, the vertex
   might be on an internal organ surface - EXCLUDE these

This would give us the TRUE outer surface vertices.
""")


if __name__ == "__main__":
    main()
