"""Precompute direct mask for P5-ventral to speed up diagnostics."""

import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def main():
    print("Loading volume...")
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

    print("Computing direct vertices (this takes ~2 min)...")
    is_direct_geo = is_direct_path_vertex_vectorized(
        gt_pos, vertices, volume, VOXEL_SIZE_MM
    )

    print(f"Direct vertices: {np.sum(is_direct_geo)}")
    print(f"Valid vertices (direct & phi>0): {np.sum(is_direct_geo & (phi_mcx > 0))}")

    output_dir = Path("pilot/paper04b_forward/results/diag_cache")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "p5_ventral_vertices.npy", vertices)
    np.save(output_dir / "p5_ventral_phi_mcx.npy", phi_mcx)
    np.save(output_dir / "p5_ventral_is_direct.npy", is_direct_geo)
    np.save(output_dir / "p5_ventral_gt_pos.npy", gt_pos)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
